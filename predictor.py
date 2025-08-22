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

class QwenBailianClient:
    """OpenAI 兼容调用百炼 Qwen3-8B"""
    def __init__(self, model: str = "qwen3-8b",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.2,
                 timeout: int = 60):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") \
                   or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info(f"QwenBailianClient init: model={model}, base_url={base_url}, temperature={temperature}")

    def chat_once(self, user: str) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role":"user","content":user}],
            extra_body={"enable_thinking": False},
        )
        dt = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        logger.debug(f"LLM latency={dt:.1f}ms, tokens≈{getattr(resp.usage,'total_tokens',None)}, "
                     f"resp_len={len(text)}")
        logger.debug(f"LLM response: " + (text[:400].replace('\n', ' ')) + ("..." if len(text) > 400 else ""))
        return text

# ------------------------------
# 打分器（含重试与日志）
# ------------------------------
class HeuristicMotivePredictor:
    """
    拼 Prompt → 调 LLM（重试/退避）→ 解析 JSON → 10 因子聚合 → 
    
    llm: QwenBailianClient 实例
    beta: 放大系数，防止过弱的因子影响
    use_global_factor_weight: 是否使用全局因子权重 G
    max_retries: 最大重试次数
    retry_delay: 初始重试延迟（秒）
    backoff: 重试指数退避系数
    jitter: 抖动范围（0~1），防止重试同步
    """
    def __init__(self, llm: QwenBailianClient, beta: float = 1.3,
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

    # -------- Prompt -------

    @staticmethod
    def _build_user_prompt(context_turns: List[str], P_t: Dict[str,float],
                           P0: Optional[Dict[str,float]] = None,
                           meta: Optional[Dict] = None) -> str:
        ctx = "\n".join(context_turns)
        schema = json.dumps(STRICT_SCHEMA_MIN, ensure_ascii=False, indent=2)
        return f"""
You are a careful analyst that scores conversational factors impacting persona expression (OCEAN) of an ASSISTANT based on the conversation context and current persona state.
You will be given a conversation context between ASSISTANT and USER and the persona of current ASSISTANT. Please analyze the context from the perspective of the ASSISTANT.
Your task is to analyze the following conversation context and current persona state of the USER, then score 10 named factors that the ASSISTANT would consider important for responding in the current turn.
Also produce per-trait salience in [0,1], which means how much each dimension of traits ASSISTANT should expressed in future dialogue turn, and give your reason about this value. 
[Context Order]: oldest-first
[Turns]:
{ctx}

[Current Persona P_t]:
{json.dumps(P_t, ensure_ascii=False)}

{"[Baseline Persona P0]:" if P0 is not None else ""}
{json.dumps(P0 if P0 is not None else None, ensure_ascii=False)}

{"[Task Meta]:" if meta is not None else ""}
{json.dumps(meta if meta is not None else None, ensure_ascii=False)}

[Ten Factors to Score]:
1) motivation       2) topic_type     3) semantic_fit     4) internal_state
5) expected_impact  6) feedback       7) fluency          8) urgency
9) contextual_setting  10) relationship

[Priors]:
- motivation: C↑ seek recognition/avoid errors; A↓ when face-saving
- topic_type: O/E↑ creative/playful; E↓ dull/formal
- semantic_fit: E↑ resonates with own experience; O↓ distant/uninterested
- internal_state: N↑ anxious/vulnerable; C↓ tired/overloaded
- expected_impact: A↑ trust/support; C↑ productive steer
- feedback: N↑ anger/criticism; A↑ kindness
- fluency: E↑ energetic rhythm; C↓ chaotic/hesitant
- urgency: C↑ deadline pressure; A↓ mistakes/misunderstanding
- contextual_setting: A↑ group harmony; E↓ formal/unknown
- relationship: E↑ with familiar; N↑ when safe to disclose vulnerability

[Output STRICT JSON Schema]:
{schema}

[Note]:
- For each factor, only adjust dimensions mentioned in Priors; keep others at dir=0,str=0,conf=0
- Fill 'explain' briefly and 'evidence' with snippets..
- Salience: provide {{val, explain}} per trait.
- Output STRICT JSON ONLY. No extra text.
""".strip()

    # -------- 调用与重试 --------
    def _chat_with_retries(self, user: str) -> str:
        delay = self.retry_delay
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"LLM request attempt {attempt}/{self.max_retries}")
                text = self.llm.chat_once(user)
                # 简单健壮性检查
                self._extract_json(text)
                return text
            except Exception as e:
                last_exc = e
                snippet = ""
                if isinstance(e, ValueError):
                    snippet = f" | reason={e}"
                logger.warning(f"Attempt {attempt} failed{snippet}")
                logger.debug(traceback.format_exc())
                if attempt == self.max_retries:
                    break
                # 指数退避 + 抖动
                sleep_s = delay * (self.backoff ** (attempt - 1))
                sleep_s *= (1.0 + random.uniform(-self.jitter, self.jitter))
                sleep_s = max(0.2, sleep_s)
                logger.info(f"Retrying in {sleep_s:.2f}s ...")
                time.sleep(sleep_s)
        raise RuntimeError(f"Qwen3 scoring failed after retries: {last_exc}")

    # -------- 解析与聚合 --------
    def _extract_json(self, text: str) -> dict:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("No JSON object found in model output.")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("Extracted JSON is not an object.")
        for m in MENTIONS.values():
            for k in DIMENSIONS:
                if k not in obj.get("salience", {}):
                    raise ValueError(f"Salience missing dimension '{k}'.")
                if not isinstance(obj["salience"][k], dict):
                    raise ValueError(f"Salience for '{k}' is not a dict.")
        return obj

    def _aggregate(self, raw: dict) -> dict:
        # s = {k: 0.0 for k in DIMENSIONS}
        # conf_bucket = {k: [] for k in DIMENSIONS}

        # # 记录每个因子的有效条目数量，便于排查
        # eff_counts = {}

        # for f, payload in raw["factors"].items():
        #     per = payload.get("per_trait", {})
        #     gf = (G.get(f, 1.0) if self.use_g else 1.0)
        #     cnt = 0
        #     for k in DIMENSIONS:
        #         if k not in per:  # 容错
        #             continue
        #         dir_k = int(per[k].get("dir", 0))
        #         str_k = float(per[k].get("str", 0.0))
        #         conf_k = float(per[k].get("conf", 0.0))
        #         s[k] += gf * W[f][k] * dir_k * str_k
        #         if dir_k != 0:
        #             conf_bucket[k].append(conf_k)
        #             cnt += 1
        #     eff_counts[f] = cnt

        # final = {}
        # for k in DIMENSIONS:
        #     val = s[k]
        #     dir_k = 0 if abs(val) < 1e-6 else (1 if val > 0 else -1)
        #     # beta 放大系数，防止过弱；不超过 1
        #     m_k = min(abs(val) * self.beta, 1.0)
        #     conf_k = sum(conf_bucket[k])/len(conf_bucket[k]) if conf_bucket[k] else 0.5
        #     final[k] = {"dir": dir_k, "m": m_k, "conf": round(conf_k, 3)}

        # # 打点总结
        # logger.info("Aggregate summary: " +
        #             ", ".join(f"{k}:dir={final[k]['dir']},m={final[k]['m']:.2f}"
        #                       for k in DIMENSIONS))
        # logger.debug("Per-factor effective counts: " + json.dumps(eff_counts, ensure_ascii=False))

        # # 兼容你自定义的 salience 结构（val+explain）
        # sal = raw.get("salience", {d: {"val": 0.0, "explain": ""} for d in DIMENSIONS})
        # # 兜底：若模型不按 schema 返回简单标量
        # for d in DIMENSIONS:
        #     v = sal.get(d, 0.0)
        #     if isinstance(v, dict) and "val" in v:
        #         continue
        #     sal[d] = {"val": float(v) if isinstance(v, (int, float)) else 0.0, "explain": ""}

        # return {
        #     "factors": raw["factors"],
        #     "salience": sal,
        #     "final": {"per_trait": final}
        # }
        logger.debug(f"Raw data for aggregation: {json.dumps(raw, ensure_ascii=False)}")
        final = {k: 0.0 for k in DIMENSIONS}
        for factor in W:
            for k in DIMENSIONS:
                if W[factor][k] == 0:
                    continue
                dir_k = int(raw["factors"].get(factor, {}).get("per_trait", {}).get(k, {}).get("dir", 0))
                str_k = float(raw["factors"].get(factor, {}).get("per_trait", {}).get(k, {}).get("str", 0.0))
                conf_k = float(raw["factors"].get(factor, {}).get("per_trait", {}).get(k, {}).get("conf", 0.0))
                gk = G[factor][k] if self.use_g else 1.0
                final[k] += gk * W[factor][k] * dir_k * str_k * conf_k
        for k in DIMENSIONS:
            val = final[k]
            dir_k = 0 if abs(val) < 1e-6 else (1 if val > 0 else -1)
            m_k = min(abs(val) * self.beta, 1.0) * dir_k
            final[k] = m_k
        result = {"salience": raw.get("salience", {}), "final": final}
        logger.info("Aggregation result: " + json.dumps(result, ensure_ascii=False, indent=2))
        return result
    # -------- 对外主流程 --------
    def score(self, context_turns: List[str], P_t: Dict[str,float],
              P0: Optional[Dict[str,float]] = None, meta: Optional[Dict] = None) -> dict:
        usr = self._build_user_prompt(context_turns, P_t, P0, meta)
        logger.debug(f"Prompt preview (chars): user={len(usr)}")

        text = self._chat_with_retries(usr)
        # 为了日志安全，截断显示
        logger.debug("Raw model text head: " + text[:400].replace("\n", " ") + ("..." if len(text) > 400 else ""))

        raw = self._extract_json(text)
        # 基础字段存在性检查（简化版）
        if "factors" not in raw:
            raise ValueError("JSON lacks 'factors' key.")
        return self._aggregate(raw)

    # —— 兼容你的 demo：返回 direction/strength/salience —— #
    def predict_for_demo(self, utterance: str, Pt: Dict[str,float]) -> Dict[str, dict]:
        aggregated = self.score([utterance], Pt, P0=None, meta=None)
        sal = aggregated["salience"]
        out = {}
        for k in DIMENSIONS:
            f = aggregated["final"]["per_trait"][k]
            out[k] = {
                "direction": int(f["dir"]),
                "strength": float(f["m"]),
                "salience": float(sal.get(k, {}).get("val", 0.0))
            }
        return out

if __name__ == "__main__":
    # 初始化
    llm = QwenBailianClient(model="qwen3-8b")  # 环境变量里配置 DASHSCOPE_API_KEY / _BASE_URL
    predictor = HeuristicMotivePredictor(llm, max_retries=4, retry_delay=1.0, backoff=2.0, jitter=0.3)

    # 评分（多轮对话可传最近若干轮，顺序与 prompt 声明一致）
    P_t = {"O":0.55,"C":0.65,"E":0.35,"A":0.70,"N":0.40}
    result = predictor.score(["[ASSISTANT] let's plan the week...", "[USER] ok, deadline is near..."], P_t)

    logger.success(f"Scored result: {json.dumps(result, ensure_ascii=False, indent=2)}")
