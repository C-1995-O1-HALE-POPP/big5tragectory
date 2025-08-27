"""
predictor_two_stage.py
Two‑stage heuristic dialog evaluation (Paper 4.3 compatible).

Stage 1 (Evidence mining):
  - For each OCEAN trait, list TWO positive (adjust) factors and TWO negative (maintain) factors,
    each with a short reason. The model should think step‑by‑step INTERNALLY but output only
    structured JSON (no chain-of-thought dumps).

Stage 2 (Scoring):
  - Given Stage 1 evidence, score each factor→trait with (dir, str, conf, reason) following
    the decision rules and priors.
  - Produce per‑trait score_probs over {1..5} and a net direction ∈ {-1,0,1}.
  - We compute m_raw = Σ p(s)*s, then m_norm = (m_raw-1)/4 ∈ [0,1].

Outputs from score():
{
  "salience": {...},
  "motive": {
     "O": {"m_raw":..,"m_norm":..,"direction":..,"score_probs":{...},
           "positive":[{"factor":...,"reason":...},...],
           "negative":[...]
     }, ... },
  "factor_signed": {O..N: float in [-1,1]},   # aggregated signed strength for auditing
  "phase1": {...}                              # original evidence for reference
}

State trackers that already consume "motive" will remain compatible.
"""

import os, json, re, time, random, traceback
from typing import Dict, List, Any, Optional
from openai import OpenAI
from loguru import logger
from prompt import generate_prior_prompt, W, BIG5_DEFINITIONS
# ------------------------------
# Constants & Heuristics
# ------------------------------
DIMENSIONS = ["O", "C", "E", "A", "N"]



MENTIONS = {
    "motivation":83,"topic_type":66,"semantic_fit":54,"internal_state":53,
    "expected_impact":46,"feedback":41,"fluency":38,"urgency":31,
    "contextual_setting":18,"relationship":17
}

# Global weights per factor per trait — normalized only across factors that touch that trait
G = {
    m: {
        k: (
            (MENTIONS[m]) /
            sum((MENTIONS[d] for d in MENTIONS if W[d][k] != 0))
        ) if W[m][k] != 0 else 0.0
        for k in DIMENSIONS
    }
    for m in MENTIONS
}

# Strict schemas for Stage 1 & Stage 2
PHASE1_SCHEMA = {
      "ADJUST": [{"factor": "", "reason": ""}, {"factor": "", "reason": ""}],
      "MAINTAIN": [{"factor": "", "reason": ""}, {"factor": "", "reason": ""}]
} 


PHASE2_SCHEMA = {
  "factors": {
    k: {
      "O":{"dir":0,"str":0,"conf":0,"reason":""},
      "C":{"dir":0,"str":0,"conf":0,"reason":""},
      "E":{"dir":0,"str":0,"conf":0,"reason":""},
      "A":{"dir":0,"str":0,"conf":0,"reason":""},
      "N":{"dir":0,"str":0,"conf":0,"reason":""},
    } for k in W.keys()
  },
  "salience": { d: {"val":0, "explain": ""} for d in DIMENSIONS },
  "scores": {
    d: {
      "direction": 0,
      "score_probs": {"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0},
      "brief": ""
    } for d in DIMENSIONS
  }
}

def get_phase2_schema(phase1_full: dict) -> dict:
    res = {}
    for dim, factors in phase1_full.items():
        res[dim]= {k: {"dir":0,"str":0,"pos":0,"reason":""} for k in factors.get("positive",[]) if k != "" }
    return res

def get_phase_2_priors(phase1_full: dict) -> str:
    dic = {}
    for dim, factors in phase1_full.items():
        for item in factors.get("positive",[]):
            if item["factor"] == "": continue
            if not dic[item["factor"]]:
                dic[item["factor"]] = [dim]
            elif dim not in dic[item["factor"]]:
                dic[item["factor"]].append(dim)
    return "\n".join([generate_prior_prompt(k, v) for k,v in dic.items()])

# ------------------------------
# Logger
# ------------------------------
LOG_LEVEL = os.getenv("QWEN_SCORER_LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <7}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "{message}"
)
logger.debug("Global factor weights (G): " + json.dumps(G, ensure_ascii=False))

# ------------------------------
# OpenAI-compatible client
# ------------------------------
class llmClient:
    """OpenAI 兼容调用 """
    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.2,
                 timeout: int = 60):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") \
                   or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info(f"QwenBailianClient init: model={model}, base_url={base_url}, temperature={temperature}")

    def chat_stream(self,
                    messages: list,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    enable_thinking: bool = False):
        """流式生成：yield 文本增量"""
        t0 = time.time()
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
            messages=messages,
            stream=True,
            # extra_body={"enable_thinking": enable_thinking},
        )
        total = ""
        for chunk in stream:
            # 兼容某些实现末尾会给空 choices 的情况
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            # 某些实现会在流中就带 finish_reason
            if getattr(choice, "finish_reason", None):
                break
            delta = getattr(choice, "delta", None)
            if delta and getattr(delta, "content", None):
                total += delta.content
                yield delta.content  # 返回增量
        dt = (time.time() - t0) * 1000
        logger.debug(f"[Qwen stream] latency={dt:.1f}ms, resp_len={len(total)}")

    def chat_once(self, messages: list,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       enable_thinking: bool = False) -> str:
        """非流式一次性返回全文（兜底）"""
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
            messages=messages,
            # extra_body={"enable_thinking": enable_thinking},
        )
        dt = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        logger.debug(f"[Qwen once] latency={dt:.1f}ms, tokens≈{getattr(resp.usage,'total_tokens',None)}, resp_len={len(text)}")
        return text
    
    def change_model(self, model: str):
        self.model = model
        logger.info(f"LLM change model to {model}")
    
    def change_temperature(self, temperature: float):
        self.temperature = temperature
        logger.info(f"LLM change temperature to {temperature}")

# ------------------------------
# Predictor (two-stage)
# ------------------------------
class HeuristicMotivePredictor:
    """
    score(context_turns, P_t, P0, meta) -> dict (see module docstring).
    """
    def __init__(self, llm: llmClient, beta: float = 1.3,
                 use_global_factor_weight: bool = True,
                 max_retries: int = 3, retry_delay: float = 1.2,
                 backoff: float = 2.0, jitter: float = 0.25):
        self.llm = llm
        self.beta = float(beta)
        self.use_g = bool(use_global_factor_weight)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.backoff = float(backoff)
        self.jitter = float(jitter)

    # ---------- Stage 1 ----------
    @staticmethod
    def _build_phase1_prompt(context_turns: List[str], trait: str, P_t: Dict[str, float],
                             P0: Optional[Dict[str, float]] = None,
                             meta: Optional[Dict[str, Any]] = None) -> str:
        if trait not in DIMENSIONS:
            raise ValueError(f"Trait '{trait}' not recognized. Valid traits: {DIMENSIONS}")
        ctx = "\n".join(context_turns)
        schema = json.dumps(PHASE1_SCHEMA, ensure_ascii=False, indent=2)
        return f"""
You are an analyst extracting EVIDENCE for whether to ADJUST or MAINTAIN the ASSISTANT's persona expression for a given trait.

Your task is to provide TWO strongest factors supporting ADJUST and TWO supporting MAINTAIN.
If the number of available factors is insufficient, ensure that at least one factor supports ADJUST and at least one factor supports MAINTAIN.
Leave empty string for factor/reason if not available.

The factor you may choose from are:
{", ".join(sorted(W.keys()))}

The trait you should focus on is: 
{trait} ({BIG5_DEFINITIONS[trait]}).

Think through the context carefully and analyze based on the assistant's Current Persona P_t{" and Baseline Persona P0" if P0 is not None else ""}, but OUTPUT ONLY structured JSON.

[Context Turns]
{ctx}

[Current Persona P_t]
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]\n" + json.dumps(P0, ensure_ascii=False) if P0 is not None else ""}

{"[Task Meta]\n" + json.dumps(meta, ensure_ascii=False) if meta is not None else ""}

[Priors: egulators per factor for ADJUST]
{[generate_prior_prompt(i, [trait]) for i in W.keys()]}

[Output STRICT JSON Schema]
{schema}

[Important]
- Choose concrete factor names 
- Reasons must be short, evidence-grounded, and avoid speculative leaps.
""".strip()

    def _call_llm(self, prompt: str) -> str:
        delay = self.retry_delay
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                text = self.llm.chat_once(messages=[{"role": "user", "content": prompt}])
                # basic JSON presence check
                if not re.search(r"\{.*\}", text, flags=re.S):
                    raise ValueError("No JSON in output.")
                return text
            except Exception as e:
                last_exc = e
                if attempt == self.max_retries:
                    break
                sleep_s = max(0.2, delay * (self.backoff ** (attempt - 1)) * (1.0 + random.uniform(-self.jitter, self.jitter)))
                time.sleep(sleep_s)
        raise RuntimeError(f"LLM call failed after retries: {last_exc}")

    def _parse_phase1(self, text: str) -> dict:
        data = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
        if "phase1" not in data or not isinstance(data["phase1"], dict):
            raise ValueError("phase1 JSON missing.")
        # sanitize: ensure required structure
        clean = {"phase1": {}}
        for d in DIMENSIONS:
            block = data["phase1"].get(d, {})
            pos = block.get("positive", [])
            neg = block.get("negative", [])
            def _fix(lst):
                res = []
                for it in lst[:2]:
                    if not isinstance(it, dict): continue
                    fac = str(it.get("factor", "")).strip()
                    if fac not in W: continue
                    res.append({"factor": fac, "reason": str(it.get("reason",""))[:300]})
                # pad to length 2
                while len(res) < 2:
                    res.append({"factor": "", "reason": ""})
                return res[:2]
            clean["phase1"][d] = {"positive": _fix(pos), "negative": _fix(neg)}
        return clean

    # ---------- Stage 2 ----------
    @staticmethod
    def _build_phase2_prompt(context_turns: List[str], P_t: Dict[str, float],
                             phase1: dict,
                             P0: Optional[Dict[str, float]] = None,
                             meta: Optional[Dict[str, Any]] = None) -> str:
        ctx = "\n".join(context_turns)
        schema = json.dumps(get_phase2_schema(phase1_full=phase1), ensure_ascii=False, indent=2)
        return f"""
Given the conversation and the Stage‑1 evidence (positive vs negative factors per trait), perform STRUCTURED SCORING:

(A) For each factor k ∈ {list(W.keys())} and each trait d ∈ {DIMENSIONS} with W[k][d]!=0,
    provide (dir,str,conf,reason) following the decision rules and priors.
(B) Produce per‑trait 'scores' with:
    - direction ∈ {{-1,0,1}} (net effect now),
    - score_probs over "1".."5" (must sum to 1 ±1e-3),
    - brief one‑line justification.
(C) Provide per‑trait salience {{val, explain}}.

Think through the evidence but OUTPUT ONLY the STRICT JSON.

[Turns]
{ctx}

[Current Persona P_t]
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]\n" + json.dumps(P0, ensure_ascii=False) if P0 is not None else ""}

{"[Task Meta]\n" + json.dumps(meta, ensure_ascii=False) if meta is not None else ""}

[Stage‑1 Evidence]
{json.dumps(phase1, ensure_ascii=False, indent=2)}

[Decision rules for direction/sign]
- dir = +1  -> AMPLIFY/encourage expression of the trait.
- dir = -1  -> SUPPRESS/attenuate expression.
- dir = 0   -> insufficient or mixed evidence; or the trait not applicable for this factor.
- Strength guideline: strong≈0.7–1.0, moderate≈0.4–0.7, weak≈0.1–0.4; str=0 if dir=0.
- 'conf' reflects evidence reliability/clarity (0–1); penalize for indirect or conflicting cues.
- It is allowed to output dir=0 or dir=-1 when evidence for +1 is weak.

[Priors (abbrev.)]


[Output STRICT JSON Schema]
{schema}

[Important]
- Only output JSON. Ensure probabilities form a valid distribution for each trait.
""".strip()

    def _parse_phase2(self, text: str) -> dict:
        data = json.loads(re.search(r"\{.*\}", text, flags=re.S).group(0))
        # Validate salience
        if "salience" not in data or not isinstance(data["salience"], dict):
            raise ValueError("Missing 'salience'.")
        for d in DIMENSIONS:
            v = data["salience"].get(d, {"val": 0.0, "explain": ""})
            if isinstance(v, dict):
                v["val"] = float(v.get("val", 0.0))
            else:
                data["salience"][d] = {"val": float(v), "explain": ""}

        # Validate factors
        if "factors" not in data or not isinstance(data["factors"], dict):
            raise ValueError("Missing 'factors'.")
        # Allow sparse; we'll fill defaults
        for k in W.keys():
            block = data["factors"].get(k, {})
            out = {}
            for d in DIMENSIONS:
                it = block.get(d, {})
                out[d] = {
                    "dir": int(it.get("dir", 0)),
                    "str": float(it.get("str", 0.0)),
                    "conf": float(it.get("conf", 0.0)),
                    "reason": str(it.get("reason", ""))[:300],
                }
            data["factors"][k] = out

        # Validate scores
        if "scores" not in data or not isinstance(data["scores"], dict):
            raise ValueError("Missing 'scores'.")
        scores = {}
        for d in DIMENSIONS:
            sblock = data["scores"].get(d, {})
            direction = int(sblock.get("direction", 0))
            if direction not in (-1, 0, 1):
                direction = 0
            probs = sblock.get("score_probs", {})
            try:
                probs = {str(k): float(probs[str(k)]) for k in range(1, 6)}
            except Exception:
                probs = {"1":1.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0}
            ssum = sum(probs.values())
            if ssum <= 0: ssum = 1.0
            probs = {k: v/ssum for k,v in probs.items()}
            scores[d] = {"direction": direction, "score_probs": probs, "brief": str(sblock.get("brief",""))[:200]}
        data["scores"] = scores
        return data

    # ---------- Aggregation ----------
    @staticmethod
    def _calc_motive_from_probs(probs: Dict[str, float]) -> Dict[str, float]:
        m_raw = sum(int(k) * probs[k] for k in ("1", "2", "3", "4", "5"))
        m_norm = max(0.0, min(1.0, (m_raw - 1.0) / 4.0))  # map 1..5 to 0..1
        return {"m_raw": m_raw, "m_norm": m_norm}

    def _aggregate_factors_signed(self, factors: dict) -> Dict[str, float]:
        total = {k: 0.0 for k in DIMENSIONS}
        for factor in W:
            per = factors.get(factor, {})
            for k in DIMENSIONS:
                if W[factor][k] == 0:
                    continue
                slot = per.get(k, {})
                dir_k = int(slot.get("dir", 0))
                str_k = float(slot.get("str", 0.0))
                conf_k = float(slot.get("conf", 0.0))
                gk = G[factor][k] if self.use_g else 1.0
                total[k] += gk * W[factor][k] * dir_k * str_k * conf_k
        # squash to [-1,1]
        final = {}
        for k, v in total.items():
            if abs(v) < 1e-6:
                final[k] = 0.0
            else:
                m = min(abs(v) * self.beta, 1.0)
                final[k] = m if v > 0 else -m
        return final

    # ---------- Public API ----------
    def score(self, context_turns: List[str], P_t: Dict[str, float],
              P0: Optional[Dict[str, float]] = None,
              meta: Optional[Dict[str, Any]] = None) -> dict:
        # Stage 1
        p1_prompt = self._build_phase1_prompt(context_turns, P_t, P0, meta)
        p1_text = self._call_llm(p1_prompt)
        phase1 = self._parse_phase1(p1_text)

        # Stage 2
        p2_prompt = self._build_phase2_prompt(context_turns, P_t, phase1, P0, meta)
        p2_text = self._call_llm(p2_prompt)
        parsed = self._parse_phase2(p2_text)

        # Build motive block
        motive = {}
        for d in DIMENSIONS:
            calc = self._calc_motive_from_probs(parsed["scores"][d]["score_probs"])
            motive[d] = {
                "m_raw": round(calc["m_raw"], 6),
                "m_norm": round(calc["m_norm"], 6),
                "direction": parsed["scores"][d]["direction"],
                "score_probs": parsed["scores"][d]["score_probs"],
                "positive": phase1["phase1"][d]["positive"],
                "negative": phase1["phase1"][d]["negative"],
            }

        factor_signed = self._aggregate_factors_signed(parsed["factors"])

        out = {
            "salience": parsed["salience"],
            "motive": motive,
            "factor_signed": factor_signed,
            "phase1": phase1  # keep Stage‑1 evidence for auditing
        }
        logger.info("[predictor] two-stage result: " + json.dumps(out, ensure_ascii=False))
        return out


if __name__ == "__main__":
    # Simple smoke test (needs OPENAI_API_KEY or DASHSCOPE_API_KEY)
    try:
        llm = llmClient(model=os.getenv("SCORER_MODEL", "gpt-4o-mini"))
        predictor = HeuristicMotivePredictor(llm, max_retries=1)
        P_t = {"O":0.55,"C":0.65,"E":0.35,"A":0.70,"N":0.40}
        ctx = ["[ASSISTANT] let's plan the week...", "[USER] ok, deadline is near..."]
        res = predictor.score(ctx, P_t)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Self-test skipped or failed:", e)