# predictor.py
import os, json, re, time, random, traceback
from typing import Dict, List, Any, Optional
from openai import OpenAI
from loguru import logger

# ------------------------------
# 基本常量
# ------------------------------
DIMENSIONS = ["O", "C", "E", "A", "N"]

W = {
    "motivation":         {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "topic_type":         {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "semantic_fit":       {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "internal_state":     {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "expected_impact":    {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":0.0},
    "feedback":           {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "fluency":            {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":0.0},
    "urgency":            {"O":0.0,"C":1.0,"E":0.0,"A":1.0,"N":0.0},
    "contextual_setting": {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "relationship":       {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
}

MENTIONS = {
    "motivation":83,"topic_type":66,"semantic_fit":54,"internal_state":53,
    "expected_impact":46,"feedback":41,"fluency":38,"urgency":31,
    "contextual_setting":18,"relationship":17
}
# _gmax = max(MENTIONS.values())
# G = {f: MENTIONS[f]/_gmax for f in MENTIONS}  # 0~1

G = {m: {k: (MENTIONS[m] if W[m][k] != 0 else 0)/ sum((MENTIONS[d] if W[d][k] != 0 else 0 for d in MENTIONS)) for k in DIMENSIONS} for m in MENTIONS} 
STRICT_SCHEMA_MIN = {
  "factors": {
    k: {"explain":"", "evidence":[], "per_trait":{
      "O":{"dir":0,"str":0,"conf":0},
      "C":{"dir":0,"str":0,"conf":0},
      "E":{"dir":0,"str":0,"conf":0},
      "A":{"dir":0,"str":0,"conf":0},
      "N":{"dir":0,"str":0,"conf":0},
    }} for k in W.keys()
  },
  "salience": {  # 你定制的带解释形式
      d : {"val":0, "explain": ""} for d in DIMENSIONS
  },
}

# ------------------------------
# Logger 配置（默认 INFO；需要可改环境变量）
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

# ------------------------------
# 百炼 OpenAI 兼容客户端
# ------------------------------
logger.debug(f"Global factor weights (G): {json.dumps(G, ensure_ascii=False, indent=2)}")

class llmClient:
    """OpenAI 兼容调用百炼 Qwen3-8B"""
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
# 打分器（含重试与日志）
# ------------------------------
class HeuristicMotivePredictor:
    """
    只做“上下文序列打分”的纯打分器：
      context_turns -> 构造 Prompt -> 调 LLM（带重试/退避）-> 解析严格 JSON -> 聚合十因子 -> 返回每维最终分

    参数：
      llm:        llmClient 实例（需提供 .chat_once(messages|prompt_str) 接口或兼容）
      beta:       放大系数，抑制过弱贡献（最终幅度 |m|<=1）
      use_global_factor_weight: 是否使用全局因子权重 G
      max_retries / retry_delay / backoff / jitter: 重试控制
    """
    def __init__(self, llm: llmClient, beta: float = 1.3,
                 use_global_factor_weight: bool = True,
                 max_retries: int = 3, retry_delay: float = 1.2,
                 backoff: float = 2.0, jitter: float = 0.25):
        self.llm = llm
        self.beta = beta
        self.use_g = use_global_factor_weight
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff = backoff
        self.jitter = jitter

    # -------- Prompt 构造（仅基于上下文序列 + P_t/P0/meta） -------
    @staticmethod
    def _build_user_prompt(context_turns: List[str], P_t: Dict[str, float],
                           P0: Optional[Dict[str, float]] = None,
                           meta: Optional[Dict[str, Any]] = None) -> str:
        ctx = "\n".join(context_turns)
        schema = json.dumps(STRICT_SCHEMA_MIN, ensure_ascii=False, indent=2)
        return f"""
You are a careful analyst that scores conversational factors impacting persona expression (OCEAN) of an ASSISTANT based on the conversation context and current persona state.
You will be given a conversation context (chronological, oldest first) and the current persona state P_t (and optionally a baseline P0).
Your job is to extract EVIDENCE for each factor, decide DIRECTION (dir ∈ {{-1,0,1}}) and STRENGTH (str ∈ [0,1]) for only the traits mentioned in the Priors, then estimate per-trait salience ({{val, explain}}).

[Turns]:
{ctx}

[Current Persona P_t]:
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]:" if P0 is not None else ""}
{json.dumps(P0 if P0 is not None else None, ensure_ascii=False)}

{"[Task Meta]:" if meta is not None else ""}
{json.dumps(meta if meta is not None else None, ensure_ascii=False)}

[Decision rules for direction/sign]
- dir = +1  → the factor tends to AMPLIFY/encourage expression of the trait given the evidence.
- dir = -1  → the factor tends to SUPPRESS/attenuate expression of the trait given the evidence (prefer this over +1 when the majority of signals point to inhibition, constraint, fatigue, conflict, or avoidance).
- dir = 0   → insufficient or mixed evidence; or the trait is not mentioned for this factor.
- Strength mapping guideline (not a hard rule): strong evidence → str≈0.7–1.0, moderate → 0.4–0.7, weak → 0.1–0.4; put str=0 if dir=0.
- Confidence 'conf' reflects evidence reliability/clarity (0–1). Penalize conf when evidence is indirect or contradictory.
- If evidence is neutral or mixed, choose dir=0.

[Priors: up/down regulators per factor]
- motivation  (C, A, E, N):
  • C↑ when seeking recognition/avoiding errors; C↓ with complacency, lack of stakes, fuzzy goals, chaotic priorities.
  • A↓ with face-saving/defensiveness/territorial behavior; A↑ with prosocial duty or cooperative goals.
  • E↑ with public visibility/social reward; E↓ with low visibility/solo, shame/withdrawal cues.
  • N↑ with fear of failure/rumination; N↓ with reassurance/clear safety net.

- topic_type  (O, E):
  • O/E↑ with creative/playful/novel tasks; 
  • E↓ with dull/formal/rote tasks; O↓ with rigid SOP/anti-novelty/bureaucratic grind.

- semantic_fit  (E, O, C):
  • E↑ when content resonates with own experience/self-disclosure; E↓ when impersonal/3rd-person/alienating.
  • O↑ for open-ended/ambiguous/idea-heavy material; O↓ for purely mechanical, highly constrained specs.
  • C↑ when material is structured and rule-bound; C↓ when it’s inconsistent/contradictory/noisy.

- internal_state  (N, C, E):
  • N↑ with anxiety/vulnerability/overwhelm; N↓ with calm/regulated/grounded states.
  • C↓ with tiredness/overload/cognitive depletion; C↑ with rested/alert/energetic states.
  • E↓ with social fatigue/withdrawal; E↑ with energized/engaged mood.

- expected_impact  (A, C, N):
  • A↑ with trust/support/prosocial payoff; A↓ with threat/blame/zero-sum framing.
  • C↑ with productive leverage/clear efficacy; C↓ with low control/pointless busywork.
  • N↑ when outcomes feel risky/irreversible; N↓ when safety margins and reversibility are explicit.

- feedback  (N, A, E, C):
  • N↑ with harsh criticism/anger/hostility; N↓ with validation/reassurance/specific guidance.
  • A↑ with kindness/benefit-of-doubt; A↓ with sarcasm/contempt/stonewalling.
  • E↓ with shaming/humiliation; E↑ with encouraging tone.
  • C↑ with actionable checklists; C↓ with vague/conflicting asks.

- fluency  (E, C):
  • E↑ with energetic rhythm/back-and-forth momentum; E↓ with monotone/withdrawn/fragmented flow.
  • C↓ with chaotic/hesitant/derailed flow; C↑ with structured cadence and turn-taking norms.

- urgency  (C, A, N):
  • C↑ with real deadlines/clear stakes; C↓ with false alarms/learned helplessness/priority thrash.
  • A↓ with time-pressure misunderstandings leading to blame; A↑ when alignment reduces friction.
  • N↑ with time scarcity + uncertainty; N↓ with buffered timelines and contingency plans.

- contextual_setting  (A, E, N):
  • A↑ with group harmony/psychological safety; A↓ with overt conflict/competitive frames.
  • E↓ with formal/unknown/large-audience settings; E↑ with familiar, small, informal settings.
  • N↑ with scrutiny/high stakes; N↓ with low-stakes practice or backstage coordination.

- relationship  (E, N, A):
  • E↑ with familiar teammates/rapport; E↓ with strangers/hostile ties.
  • N↑ when the space is safe to disclose vulnerability; N↓ when boundaries are respected and support norms are clear.
  • A↑ with trust history/reciprocity; A↓ with breaches, perceived exploitation, or status threats.

[Output STRICT JSON Schema]
{schema}

[Notes]
- For each factor, only adjust the traits listed above; keep others at dir=0,str=0,conf=0.
- Provide a brief 'explain' and include short 'evidence' snippets quoted/paraphrased from the context.
- Output STRICT JSON ONLY. No extra text.
""".strip()


    # -------- LLM 调用（带重试/退避 + 进度日志） --------
    def _chat_with_retries(self, user_prompt: str) -> str:
        delay = self.retry_delay
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[predictor] LLM attempt {attempt}/{self.max_retries}, prompt_len={len(user_prompt)}")
                t0 = time.time()
                text = self.llm.chat_once(messages=[{"role": "user", "content": user_prompt}])
                dt = (time.time() - t0) * 1000
                logger.info(f"[predictor] LLM ok in {dt:.1f}ms, resp_head={(text[:120].replace(chr(10),' '))!r}")
                # 解析校验（若不符合预期会抛 ValueError 触发重试）
                self._extract_json(text)
                return text
            except Exception as e:
                last_exc = e
                reason = f" | reason={e}" if isinstance(e, ValueError) else ""
                logger.warning(f"[predictor] attempt {attempt} failed{reason}")
                logger.debug(traceback.format_exc())
                if attempt == self.max_retries:
                    break
                # 指数退避 + 抖动
                sleep_s = delay * (self.backoff ** (attempt - 1))
                sleep_s *= (1.0 + random.uniform(-self.jitter, self.jitter))
                sleep_s = max(0.2, sleep_s)
                logger.info(f"[predictor] retrying in {sleep_s:.2f}s ...")
                time.sleep(sleep_s)
        raise RuntimeError(f"Scoring failed after retries: {last_exc}")

    # -------- 解析严格 JSON --------
    def _extract_json(self, text: str) -> dict:
        # 抓第一段 JSON（允许模型偶尔加提示词/解释）
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in model output.")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("Extracted JSON is not an object.")

        # 基本字段检查
        if "factors" not in obj or "salience" not in obj:
            raise ValueError("JSON must contain 'factors' and 'salience'.")

        # salience 结构容错：允许 {val, explain} 或标量
        sal = obj["salience"]
        if not isinstance(sal, dict):
            raise ValueError("'salience' must be an object.")
        for d in DIMENSIONS:
            v = sal.get(d, {"val": 0.0, "explain": ""})
            if isinstance(v, dict):
                if "val" not in v:
                    raise ValueError(f"salience['{d}'] missing 'val'.")
                try:
                    v["val"] = float(v["val"])
                except Exception:
                    raise ValueError(f"salience['{d}'].val must be numeric.")
            else:  # 标量兜底
                try:
                    sal[d] = {"val": float(v), "explain": ""}
                except Exception:
                    raise ValueError(f"salience['{d}'] must be numeric or {{val, explain}}.")
        obj["salience"] = sal
        return obj

    # -------- 聚合十因子 → 每维最终分 --------
    def _aggregate(self, raw: dict) -> dict:
        logger.debug(f"[predictor] raw JSON: {json.dumps(raw, ensure_ascii=False)[:1500]}")
        total = {k: 0.0 for k in DIMENSIONS}
        for factor in W:
            per = raw.get("factors", {}).get(factor, {}).get("per_trait", {})
            for k in DIMENSIONS:
                if W[factor][k] == 0:
                    continue
                slot = per.get(k, {})
                dir_k = int(slot.get("dir", 0))
                str_k = float(slot.get("str", 0.0))
                conf_k = float(slot.get("conf", 0.0))
                gk = G[factor][k] if self.use_g else 1.0
                total[k] += gk * W[factor][k] * dir_k * str_k * conf_k

        # 压幅 & 带符号
        final = {}
        for k, v in total.items():
            if abs(v) < 1e-6:
                final[k] = 0.0
            else:
                m = min(abs(v) * self.beta, 1.0)
                final[k] = m if v > 0 else -m

        out = {"salience": raw.get("salience", {}), "final": final}
        logger.info("[predictor] aggregate result: " + json.dumps(out, ensure_ascii=False, indent=2))
        return out

    # -------- 对外主流程（唯一入口） --------
    def score(self, context_turns: List[str], P_t: Dict[str, float],
              P0: Optional[Dict[str, float]] = None, meta: Optional[Dict[str, Any]] = None) -> dict:
        """
        传入：对话上下文序列（list[str]，从旧到新）、当前人格 P_t（0~1），可选 P0/meta
        返回：{
          "salience": { O..N: {"val": float, "explain": str} },
          "final":    { O..N: float in [-1,1] }   # 十因子聚合后的方向带幅度
        }
        """
        user_prompt = self._build_user_prompt(context_turns, P_t, P0, meta)
        logger.debug(f"[predictor] prompt_len={len(user_prompt)}")
        text = self._chat_with_retries(user_prompt)
        logger.debug("[predictor] raw_head: " + text[:400].replace("\n", " ") + ("..." if len(text) > 400 else ""))
        raw = self._extract_json(text)
        return self._aggregate(raw)

if __name__ == "__main__":
    # 初始化 LLM 客户端（需提前在环境里 export OPENAI_API_KEY / OPENAI_BASE_URL）
    llm = llmClient(model="gpt-4o-mini")
    predictor = HeuristicMotivePredictor(
        llm,
        max_retries=4,
        retry_delay=1.0,
        backoff=2.0,
        jitter=0.3
    )

    # 当前人格向量 P_t
    P_t = {"O": 0.55, "C": 0.65, "E": 0.35, "A": 0.70, "N": 0.40}

    # 输入上下文（对话片段，顺序 oldest-first）
    context = [
        "[ASSISTANT] let's plan the week...",
        "[USER] ok, deadline is near..."
    ]

    # 调用打分
    result = predictor.score(context, P_t)

    # 打印结果
    logger.success("最终打分结果：\n" + json.dumps(result, ensure_ascii=False, indent=2))
