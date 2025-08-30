# predictor_two_stage.py
"""
Two-stage heuristic dialog evaluation (Paper 4.3 compatible).

Stage 1 (Evidence mining, per trait):
  - For the target OCEAN trait, list TWO ADJUST factors and TWO MAINTAIN factors,
    each with a short reason. The model should think step-by-step INTERNALLY
    but output only structured JSON (no chain-of-thought dumps).

Stage 2 (Scoring, per trait):
  - Hide the Stage-1 ADJUST/MAINTAIN labels. Provide the factor list for THIS trait.
  - For each factor→trait output (dir ∈ {-1,1}, score ∈ {1..5}, conf ∈ [0,1], reason).
  - Build per-trait score_probs over {1..5} from confidence-weighted histogram.
  - Compute m_raw = Σ p(s)*s, then m_norm = (m_raw-1)/4 ∈ [0,1].
  - Net direction ∈ {-1,0,1} is sign of Σ(dir*score*conf).

Outputs from score():
{
  "motive": {
     "O": {"m_raw":..,"m_norm":..,"direction":..,"score_probs":{...},
           "ADJUST":[{"factor":...,"reason":...},...],
           "MAINTAIN":[...]
     }, ... },
  "factor_signed": {O..N: float in [-1,1]},   # aggregated signed strength for auditing
  "phase1": {...}                              # original evidence for reference (ADJUST/MAINTAIN)
}

State trackers that already consume "motive" will remain compatible.
"""

import os, json, re, time, random, traceback
from typing import Dict, List, Any, Optional, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed  # 可按需切回并发

from openai import OpenAI
from loguru import logger
from prompt import generate_prior_prompt, W, BIG5_DEFINITIONS  # <- 需提供：W[factor][trait]权重矩阵、维度定义、先验生成函数
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
# ------------------------------
# Constants & Heuristics
# ------------------------------
DIMENSIONS = ["O", "C", "E", "A", "N"]
RPM_LIMIT = 1200
semaphore = Semaphore(RPM_LIMIT // 60)
# 这些“提及度”用于构造全局权重 G（只在需要时使用）
MENTIONS = {
    "motivation":83,"topic_type":66,"semantic_fit":54,"internal_state":53,
    "expected_impact":46,"feedback":41,"fluency":38,"urgency":31,
    "contextual_setting":18,"relationship":17
}

LO_TRAIT = 0.2
HI_TRAIT = 0.8

WEAK_TRAIT_NORM = 0.33
STRONG_TRAIT_NORM = 0.66

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

# Strict schema for Stage 1 (per trait)
PHASE1_SCHEMA = {
    "ADJUST": [{"factor": "", "reason": ""}, {"factor": "", "reason": ""}],
    "MAINTAIN": [{"factor": "", "reason": ""}, {"factor": "", "reason": ""}]
}

# ------------------------------
# Logger
# ------------------------------
LOG_LEVEL = os.getenv("QWEN_SCORER_LOG_LEVEL", "DEBUG")
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
        self.temperature = float(temperature)
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") \
                   or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            raise RuntimeError("DASHSCOPE/OPENAI API KEY is missing.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info(f"LLM init: model={model}, base_url={base_url}, temperature={temperature}")

    def chat_once(self, messages: list,
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature if temperature is None else float(temperature),
            max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
            messages=messages,
        )
        dt = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        logger.debug(f"[chat_once] {dt:.1f}ms, len={len(text)}")
        return text

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

    def change_model(self, model: str):
        self.model = model
        logger.info(f"LLM change model to {model}")
    
    def change_temperature(self, temperature: float):
        self.temperature = temperature
        logger.info(f"LLM change temperature to {temperature}")
# ------------------------------
# Utils
# ------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _hist_probs_from_factor_scores(evals: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Build score_probs over {1..5} using confidence-weighted histogram of Stage-2 'score'.
    """
    bins = {str(i): 0.0 for i in range(1, 6)}
    for e in evals:
        s = int(clamp(int(e.get("score", 0)), 1, 5))
        c = float(clamp(float(e.get("conf", 0.0)), 0.0, 1.0))
        bins[str(s)] += max(0.0, c)
    total = sum(bins.values())
    if total <= 0:
        bins["1"] = 1.0
        total = 1.0
    return {k: v / total for k, v in bins.items()}

def _expected_from_probs(probs: Dict[str, float]) -> Tuple[float, float]:
    m_raw = sum(int(k) * probs[k] for k in ("1", "2", "3", "4", "5"))
    m_norm = clamp((m_raw - 1.0) / 4.0, 0.0, 1.0)
    return m_raw, m_norm

# ------------------------------
# Factor key normalization (ALL-CAPS compatibility)
# ------------------------------
def _normalize_factor_key(s: str) -> str:
    """统一转换为全大写，并将非字母数字转为下划线后去除首尾下划线。"""
    return re.sub(r"[^A-Z0-9]", "_", str(s or "").upper()).strip("_")

def _w_upper_map() -> Dict[str, str]:
    """构建 W 的大写映射表： { 'URGENCY': 'urgency', ... }"""
    return { _normalize_factor_key(k): k for k in W.keys() }

# ------------------------------
# Predictor (two-stage)
# ------------------------------
class HeuristicMotivePredictor:
    """
    score(context_turns, P_t, P0, meta) -> dict (see module docstring).
    两阶段均以“单一 trait”为单位，在外层并发跑五个维度。
    """
    def __init__(self, llm: llmClient, beta: float = 1.3,
                 use_global_factor_weight: bool = True,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 backoff: float = 1.8, jitter: float = 0.25,
                 eps: float = 0.25):
        self.llm = llm
        self.beta = float(beta)
        self.use_g = bool(use_global_factor_weight)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.backoff = float(backoff)
        self.jitter = float(jitter)
        self.eps = eps  # 用于判定 signed_sum 是否为零

    # ---------- Robust LLM JSON parsing with retries ----------
    def _parse_json(self, text: str) -> dict:
        """
        Extract the first JSON object from text safely.
        """
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in LLM output.")
        try:
            return json.loads(m.group(0))
        except Exception as e:
            raise ValueError(f"JSON decode failed: {e}")

    def _ask_json(self, prompt: str) -> dict:
        """
        Call LLM with retries and return parsed JSON.
        """
        with semaphore:
            delay = self.retry_delay
            last_exc = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self.llm.chat_once(messages=[{"role": "user", "content": prompt}])
                    data = self._parse_json(text)
                    return data
                except Exception as e:
                    last_exc = e
                    if attempt == self.max_retries:
                        break
                    sleep_s = max(0.2, delay * (self.backoff ** (attempt - 1)) * (1.0 + random.uniform(-self.jitter, self.jitter)))
                    time.sleep(sleep_s)
            raise RuntimeError(f"LLM call/parse failed after retries: {last_exc}")

    # ---------- Stage 1 ----------
    @staticmethod
    def _build_phase1_prompt(context_turns: List[str], trait: str, P_t: Dict[str, float],
                             P0: Optional[Dict[str, float]] = None,
                             meta: Optional[Dict[str, Any]] = None) -> str:
        if trait not in DIMENSIONS:
            raise ValueError(f"Trait '{trait}' not recognized. Valid traits: {DIMENSIONS}")
        ctx = "\n".join(context_turns)
        schema = json.dumps(PHASE1_SCHEMA, ensure_ascii=False, indent=2)
        # 列出大写版本，要求模型只用这些 key（全大写）
        allowed_upper = [k.upper() for k in sorted(W.keys())]
        priors = [generate_prior_prompt(i, [trait]) for i in W.keys()]

        return f"""
You are an analyst extracting EVIDENCE for whether to ADJUST or MAINTAIN the ASSISTANT's persona expression for a given trait.

STRICT RULES:
- You MUST choose factor keys EXACTLY from this ALL-CAPS allowed set:
  {allowed_upper}
- Do NOT invent new factor names. If none apply, use empty string "".
- Reasons must be short, evidence-grounded, and non-speculative.

Task:
- Provide TWO strongest factors for ADJUST and TWO for MAINTAIN (if insufficient, ensure at least one each; use "" if none).

Trait to focus:
{trait} ({BIG5_DEFINITIONS[trait]}).

Persona scale: traits in [0,1]; <{WEAK_TRAIT_NORM} weak, {WEAK_TRAIT_NORM}–{STRONG_TRAIT_NORM} moderate, >{STRONG_TRAIT_NORM} strong.

[Context Turns]
{ctx}

[Current Persona P_t]
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]\n" + json.dumps(P0, ensure_ascii=False) if P0 is not None else ""}

{"[Task Meta]\n" + json.dumps(meta, ensure_ascii=False) if meta is not None else ""}

[Priors: regulators per factor for ADJUST]
{json.dumps(priors, ensure_ascii=False, indent=2)}

[STRICT OUTPUT JSON Schema; factor name must be from the ALL-CAPS list above])]
{schema}

""".strip()

    @staticmethod
    def _parse_phase1_one_trait(raw: dict) -> Dict[str, List[Dict[str, str]]]:
        """
        raw: {"ADJUST":[{factor,reason}...], "MAINTAIN":[...]}
        -> {"ADJUST":[...2], "MAINTAIN":[...2]}  (pad with empty if needed; filter invalid factors)
        现在：接受模型返回的“全大写因子名”，并映射回 W 的真实 key。
        """
        W_UPPER = _w_upper_map()

        def _fix(lst):
            res = []
            for it in (lst or [])[:2]:
                if not isinstance(it, dict):
                    continue
                raw_fac = str(it.get("factor", "")).strip()
                fac_upper = _normalize_factor_key(raw_fac)
                if fac_upper not in W_UPPER:
                    continue
                fac_actual = W_UPPER[fac_upper]  # 回到原始 W 的 key（一般是小写 snake_case）
                res.append({"factor": fac_actual, "reason": str(it.get("reason",""))[:300]})
            while len(res) < 2:
                res.append({"factor":"", "reason":""})
            return res[:2]

        return {
            "ADJUST": _fix(raw.get("ADJUST")),
            "MAINTAIN": _fix(raw.get("MAINTAIN"))
        }

    # ---------- Stage 2 helpers ----------
    @staticmethod
    def _get_phase2_factors(phase1_trait: dict, adjust_only: bool = False) -> List[str]:
        factors = []
        for kv in (phase1_trait.get("ADJUST") or []):
            f = kv.get("factor","")
            if f and f not in factors:
                factors.append(f)
        if not adjust_only:
            for kv in (phase1_trait.get("MAINTAIN") or []):
                f = kv.get("factor","")
                if f and f not in factors:
                    factors.append(f)
        return factors

    @staticmethod
    def _get_phase2_schema(phase1_trait: dict, trait: str) -> dict:
        return {
            trait: {
                factor: {"dir":0, "score":0, "conf":0.0, "reason":""}
                for factor in HeuristicMotivePredictor._get_phase2_factors(phase1_trait)
            }
        }

    @staticmethod
    def _get_phase2_priors(phase1_trait: dict, trait: str) -> str:
        pri = [generate_prior_prompt(f, [trait]) for f in HeuristicMotivePredictor._get_phase2_factors(phase1_trait)]
        return "\n".join(pri)

    @staticmethod
    def _build_phase2_prompt(context_turns: List[str], P_t: Dict[str, float],
                             phase1_trait: dict, trait: str,
                             P0: Optional[Dict[str, float]] = None,
                             meta: Optional[Dict[str, Any]] = None,
                             history: Optional[List] = None) -> str:
        """
        Stage-2（单一 trait）：隐藏 ADJUST/MAINTAIN 标签，仅提供该 trait 的 factor 列表。
        对每个 factor→trait 输出 (dir ∈ {-1,1}, score ∈ {1..5}, conf ∈ [0,1], reason)。
        """
        ctx = "\n".join(context_turns)
        schema = json.dumps(HeuristicMotivePredictor._get_phase2_schema(phase1_trait, trait), ensure_ascii=False, indent=2)
        priors = HeuristicMotivePredictor._get_phase2_priors(phase1_trait, trait)
        return f"""
You are the Stage-2 evaluator (single trait).

Trait to evaluate:
{trait} ({BIG5_DEFINITIONS[trait]})

Task: For EACH factor below, judge its influence on THIS trait in the CURRENT CONTEXT.
For every factor, output:
- dir ∈ {{-1, 1}}:  -1 = suppress/attenuate expression;  +1 = amplify/encourage expression
- score ∈ {{1,2,3,4,5}}: motivation strength NOW
    1 = very weak/irrelevant; 3 = moderate/mixed; 5 = very strong/compelling
- conf ∈ [0,1]: confidence (penalize vague/indirect/conflicting evidence)
- reason: one short, evidence-grounded justification

Persona scale: trait values in [0,1]: 
- <{WEAK_TRAIT_NORM} weak, {WEAK_TRAIT_NORM}–{STRONG_TRAIT_NORM} moderate, >{STRONG_TRAIT_NORM} strong.

You should pay attention to the current personality dimension value (P_t). 
If it is too high or too low (for example, below {LO_TRAIT} or above {HI_TRAIT}), make appropriate maintenance towards the middle value to leave enough space for personality changes in future conversations.

More importantly, you should consider your previous rounds of adjustments to avoid overcorrection. 
For example, if you've already increased a dimension multiple times (more than three times), in future adjustments, consider decreasing that dimension's value first, rather than maintaining or increasing it.
Your operation history is shown below. 1 indicates increases in dimension values, while -1 indicate decreases. The earliest rounds are prioritized.

[Operation History]
{json.dumps(history, ensure_ascii=False)}

[Context Turns]
{ctx}

[Current Persona P_t]
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]\n" + json.dumps(P0, ensure_ascii=False) if P0 is not None else ""}

{"[Task Meta]\n" + json.dumps(meta, ensure_ascii=False) if meta is not None else ""}

[Factors to evaluate]
{json.dumps(HeuristicMotivePredictor._get_phase2_factors(phase1_trait), ensure_ascii=False, indent=2)}

[Priors (factor→trait hints)]
{priors}

[STRICT OUTPUT JSON SCHEMA]
{schema}

[Important]
- Output ONLY JSON following the schema exactly.
- Every listed factor must appear.
- Keep reasons short and evidence-based; no speculation.
- ADJUST/MAINTAIN labels from Stage-1 are hidden in this stage.
""".strip()

    @staticmethod
    def _parse_phase2_one_trait(raw: dict, trait: str, phase1_trait: dict) -> Dict[str, Dict[str, Any]]:
        """
        raw: { "<trait>": { "<factor>": {dir, score, conf, reason}, ... } }
        -> normalized dict for that trait.
        """
        expect = HeuristicMotivePredictor._get_phase2_factors(phase1_trait)
        block = raw.get(trait, {}) if isinstance(raw, dict) else {}
        out = {}
        for f in expect:
            it = block.get(f, {}) if isinstance(block, dict) else {}
            d = int(it.get("dir", 0))
            d = -1 if d < 0 else (1 if d > 0 else 0)   # clamp to {-1,0,1}
            sc = int(it.get("score", 0))
            sc = int(clamp(sc, 1, 5)) if d != 0 else 1  # if dir=0, fallback to minimal strength
            cf = float(clamp(float(it.get("conf", 0.0)), 0.0, 1.0))
            rs = str(it.get("reason", ""))[:300]
            out[f] = {"dir": d, "score": sc, "conf": cf, "reason": rs}
        return out

    # ---------- Aggregation ----------
    def _aggregate_factors_signed(self, merged_factors: dict) -> Dict[str, float]:
        """
        merged_factors: {factor: {trait: {dir, score, conf, reason}, ...}, ...}
        使用 W/G 权重、dir/score/conf 计算每个 trait 的总签名强度，并压缩到 [-1,1]。
        """
        total = {k: 0.0 for k in DIMENSIONS}
        for factor, per_trait in merged_factors.items():
            for k in DIMENSIONS:
                if W[factor][k] == 0:
                    continue
                slot = per_trait.get(k, {})
                dir_k = int(slot.get("dir", 0))
                sc_k = float(slot.get("score", 0.0))
                conf_k = float(slot.get("conf", 0.0))
                gk = G[factor][k] if self.use_g else 1.0
                total[k] += gk * W[factor][k] * dir_k * sc_k * conf_k
        # squash to [-1,1]
        final = {}
        for k, v in total.items():
            if abs(v) < 1e-9:
                final[k] = 0.0
            else:
                m = min(abs(v) * self.beta / 5.0, 1.0)  # /5 使强度与(1..5)尺度相称
                final[k] = m if v > 0 else -m
        return final

    # ---------- Public API ----------
    def score(self, context_turns: List[str], P_t: Dict[str, float],
              P0: Optional[Dict[str, float]] = None,
              meta: Optional[Dict[str, Any]] = None,
              history: Optional[Dict[str, List]] = None) -> dict:
        """
        顺序跑 O,C,E,A,N 五个维度（如需并发，参考注释的 ThreadPoolExecutor 版本）。
        每个维度独立走 Stage-1 (ADJUST/MAINTAIN evidence) + Stage-2 (factor 评分)。
        """
        def _run_for_trait(trait: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            # Stage 1
            p1_prompt = self._build_phase1_prompt(context_turns, trait, P_t, P0, meta)
            p1_raw = self._ask_json(p1_prompt)
            p1_trait = self._parse_phase1_one_trait(p1_raw)
            adjust_factors = self._get_phase2_factors(p1_trait, adjust_only=True)

            # Stage 2
            p2_prompt = self._build_phase2_prompt(context_turns, P_t, p1_trait, trait, P0, meta, history.get(trait)) # type: ignore
            p2_raw = self._ask_json(p2_prompt)
            p2_trait = self._parse_phase2_one_trait(p2_raw, trait, p1_trait)

            # 聚合得出该 trait 的动机
            eval_list = list(p2_trait[i] for i in adjust_factors if i in p2_trait)
            probs = _hist_probs_from_factor_scores(eval_list)
            m_raw, m_norm = _expected_from_probs(probs)

            signed_sum = sum(e["dir"] * e["score"] * e["conf"] for e in eval_list)
            direction = 1 if signed_sum > self.eps else (-1 if signed_sum < -self.eps else 0)

            motive_trait = {
                "m_raw": round(m_raw, 6),
                "m_norm": round(m_norm, 6),
                "direction": direction,
                "score_probs": {k: round(float(v), 6) for k, v in probs.items()},
                "ADJUST": p1_trait["ADJUST"],
                "MAINTAIN": p1_trait["MAINTAIN"],
            }
            return trait, motive_trait, p1_trait, p2_trait

        motive: Dict[str, Any] = {}
        phase1_all = {"phase1": {}}
        factors_merged: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(_run_for_trait, d) for d in DIMENSIONS]
            for fut in as_completed(futures):
                trait, motive_trait, p1_trait, p2_trait = fut.result()
                motive[trait] = motive_trait
                phase1_all["phase1"][trait] = {
                    "ADJUST": p1_trait["ADJUST"],
                    "MAINTAIN": p1_trait["MAINTAIN"]
                }
                for f, rec in p2_trait.items():
                    factors_merged.setdefault(f, {})
                    factors_merged[f][trait] = rec

        # 顺序版本
        # for d in DIMENSIONS:
        #     try:
        #         trait, motive_trait, p1_trait, p2_trait = _run_for_trait(d)
        #         motive[trait] = motive_trait
        #         phase1_all["phase1"][trait] = {
        #             "ADJUST": p1_trait["ADJUST"],
        #             "MAINTAIN": p1_trait["MAINTAIN"]
        #         }
        #         for f, rec in p2_trait.items():
        #             factors_merged.setdefault(f, {})
        #             factors_merged[f][trait] = rec
        #     except Exception as e:
        #         logger.error(f"Trait {d} processing failed: {e}\n{traceback.format_exc()}")

        factor_signed = self._aggregate_factors_signed(factors_merged)

        out = {
            "motive": motive,
            "factor_signed": factor_signed,
            "phase1": phase1_all
        }
        return out


if __name__ == "__main__":
    try:
        llm = llmClient(model=os.getenv("SCORER_MODEL", "gpt-4o-mini"))
        predictor = HeuristicMotivePredictor(llm, max_retries=2)
        P_t = {"O":0.55,"C":0.65,"E":0.35,"A":0.70,"N":0.40}
        ctx = ["[ASSISTANT] let's plan the week...", "[USER] ok, deadline is near..."]
        res = predictor.score(ctx, P_t)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Self-test skipped or failed:", e)
