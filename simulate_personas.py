# -*- coding: utf-8 -*-
"""
simulate_personas.py
====================

- 从 HuggingFace `Cynaptics/persona-chat` 采样 persona 句与首句（无本地 fallback）。
- 用户“场景动态/稳定”与助手“人格动态/静态”两开关 → 四种组合。
- 同一随机种子统一预采样 N 组对话规格，在四种组合中复用，保证可比对齐。
- 助手人格的动态调整使用 PersonaStateTracker。
- 用户由 PromptedUserAgent 生成消息；首句强制为预采样的 dataset 首句。

本版本新增（Orthogonal Switch + Neutral Tone + Anti-Echo for User）：
- **正交转场（orthogonal switch）**：在指定回合切到与既往完全无关的主题；**由程序合成**该回合用户消息（白名单、2–3句、结尾单问、语气中性），确保真正脱离原话题且不触发安全模板。
- **用户侧 Anti-Echo**：用户生成后做相似度闸（Jaccard），过近则**二次重采样**（最多2次）；还向 system 注入最近 3 条助手片段的 **DO-NOT-COPY** 黑名单，降低“镜像/复述”概率；失败则走模板 fallback。
- **助手侧 Anti-Repeat**：沿用最近回复片段黑名单 + 相似度闸 + 二次生成。
- 每次对话生成 uuid；导出 OCEAN 轨迹 JSON 与 PNG 曲线；index 导出。

运行：
  python simulate_personas.py
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid
import os

# ----- matplotlib（无显示环境安全）-----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset  # must be available

# ============== Logging & Progress ==============
try:
    from loguru import logger
except Exception:
    import logging as _logging
    class _FallbackLogger:
        def __getattr__(self, name: str):
            fn = getattr(_logging.getLogger(__name__), name, None)
            if fn is None:
                def noop(*a, **k): pass
                return noop
            return fn
    logger = _FallbackLogger()

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, total=None, desc=None):
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else None
        for i, x in enumerate(iterable, 1):
            if total:
                logger.info(f"{desc or 'Progress'}: {i}/{total} ({i/total*100:.1f}%)")
            yield x


# ============== Project Modules ==============
from prompt import big5_system_prompts_en, SYSTEM_PROMPT  # 仅引入需要的
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker, DIMENSIONS


# ============== I/O & Plot Utils ==============
OUTPUT_ROOT = os.path.join(".", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "persona_plots")
TRACES_DIR = os.path.join(OUTPUT_ROOT, "persona_traces")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TRACES_DIR, exist_ok=True)

def _ensure_dirs_for_combo(combo_tag: str) -> Tuple[str, str]:
    plots = os.path.join(PLOTS_DIR, combo_tag)
    traces = os.path.join(TRACES_DIR, combo_tag)
    os.makedirs(plots, exist_ok=True)
    os.makedirs(traces, exist_ok=True)
    return plots, traces

def save_persona_trace_json(trace: List[Dict[str, float]], out_path: str) -> None:
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[trace] failed to save {out_path}: {e}")

def plot_persona_trace(trace: List[Dict[str, float]], title: str, out_path: str) -> None:
    if not trace:
        logger.error(f"[plot] empty trace for {out_path}")
        return
    ts = [d["t"] for d in trace]
    plt.figure(figsize=(9, 5))
    for dim in DIMENSIONS:
        ys = [d[dim] for d in trace]
        plt.plot(ts, ys, label=dim, linewidth=2)
    plt.ylim(-0.05, 1.05)
    plt.xlim(min(ts), max(ts))
    plt.xlabel("Turn", fontsize=11)
    plt.ylabel("Trait value (0–1)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=5, frameon=False)
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=180)
    except Exception as e:
        logger.error(f"[plot] failed to save {out_path}: {e}")
    finally:
        plt.close()


# ============== Dataset Loading (NO FALLBACK) ==============
_DATASET_LOADED: bool = False
_DATASET_PERSONA_CACHE: List[str] = []
_DATASET_FIRST_MSG_CACHE: List[str] = []

def _extract_first_message(row) -> Optional[str]:
    dlg = row.get("dialogue")
    if isinstance(dlg, list) and dlg:
        if isinstance(dlg[0], str):
            return dlg[0].strip()
        if isinstance(dlg[0], dict):
            for k in ("text", "utterance", "content"):
                if k in dlg[0] and isinstance(dlg[0][k], str):
                    return dlg[0][k].strip()
    for key in ("first_utterance", "first_message"):
        if isinstance(row.get(key), str):
            return row[key].strip()
    return None

def _load_dataset_for_sampling(max_rows: int = 8000) -> None:
    global _DATASET_LOADED, _DATASET_PERSONA_CACHE, _DATASET_FIRST_MSG_CACHE
    if _DATASET_LOADED:
        return
    logger.info("[dataset] loading Cynaptics/persona-chat (train, streaming=True)")
    ds_iter = load_dataset("Cynaptics/persona-chat", split="train", streaming=True)
    persona_lines: List[str] = []
    first_msgs: List[str] = []
    for i, row in enumerate(ds_iter, 1):
        plist = row.get("persona_b") or row.get("persona_a") or row.get("persona") or []
        if isinstance(plist, (list, tuple)):
            for p in plist:
                if isinstance(p, str) and p.strip():
                    persona_lines.append(p.strip())
        fm = _extract_first_message(row)
        if fm:
            first_msgs.append(fm)
        if i >= max_rows:
            break
    _DATASET_PERSONA_CACHE = list({s for s in persona_lines if s})
    _DATASET_FIRST_MSG_CACHE = list({s for s in first_msgs if s})
    if not _DATASET_PERSONA_CACHE or not _DATASET_FIRST_MSG_CACHE:
        raise RuntimeError("Failed to populate dataset caches (persona/first message).")
    _DATASET_LOADED = True
    logger.info(f"[dataset] cached persona={len(_DATASET_PERSONA_CACHE)}, first_msgs={len(_DATASET_FIRST_MSG_CACHE)}")


def sample_persona_lines_from_dataset(n: int) -> List[str]:
    if not _DATASET_LOADED:
        _load_dataset_for_sampling()
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return []
    if n > len(_DATASET_PERSONA_CACHE):
        raise ValueError(f"Cannot sample {n} persona lines; only {len(_DATASET_PERSONA_CACHE)} available.")
    return random.sample(_DATASET_PERSONA_CACHE, n)

def sample_first_message_from_dataset() -> str:
    if not _DATASET_LOADED:
        _load_dataset_for_sampling()
    if not _DATASET_FIRST_MSG_CACHE:
        raise RuntimeError("No first messages in dataset cache.")
    return random.choice(_DATASET_FIRST_MSG_CACHE)


# ============== Simple Anti-Repeat / Anti-Echo Utils ==============
def _tokenize(s: str) -> List[str]:
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def jaccard_sim(a: str, b: str) -> float:
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def build_do_not_repeat_snippets(history: List[Dict[str, str]], k: int = 4, max_len: int = 220) -> List[str]:
    """抽取最近 k 条【助手】回复，供助手侧避免复用。"""
    snippets: List[str] = []
    cnt = 0
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if len(text) > max_len:
            text = text[:max_len]
        snippets.append(text)
        cnt += 1
        if cnt >= k:
            break
    return list(reversed(snippets))

def build_do_not_copy_from_assistant(history: List[Dict[str, str]], k: int = 3, max_len: int = 220) -> List[str]:
    """抽取最近 k 条【助手】回复片段，供用户侧避免复述/镜像。"""
    snippets: List[str] = []
    count = 0
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        if len(txt) > max_len:
            txt = txt[:max_len]
        snippets.append(txt)
        count += 1
        if count >= k:
            break
    return list(reversed(snippets))


# ============== Dynamic System Prompt (OCEAN-bucketed) ==============
def _nearest_key(d: dict[float, str], v: float) -> float:
    keys = list(d.keys())
    if not keys:
        return round(v, 1)
    return min(keys, key=lambda k: abs(k - v))

def generate_dynamic_system_prompt(
    base_text: str,
    enable_base: bool,
    vals: dict[str, float],
    table: dict[str, dict[float, str]],
) -> str:
    parts = []
    if enable_base and base_text.strip():
        parts.append(base_text.strip())
    for trait in ["O", "C", "E", "A", "N"]:
        v = vals.get(trait, None)
        if v is None:
            continue
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{trait} must be in [0,1], got {v}")
        bucket = round(v, 1)
        if trait not in table or bucket not in table[trait]:
            if trait not in table:
                continue
            bucket = _nearest_key(table[trait], bucket)
        parts.append(table[trait][bucket])
    return " ".join(parts).strip()


# ============== Orthogonal Switch 话题白名单（温和且非敏感） ==============
ORTHOGONAL_TOPICS = (
    "typography and layout systems",
    "architecture and urban design",
    "workflow tools and note-taking methods",
    "minimalism and home organization",
    "board games and strategy mechanics",
    "gardening and indoor plants",
    "photography composition techniques",
    "coffee brewing methods",
    "budget travel logistics (non-emergency)",
    "classical literature you’ve been reading",
)


# ============== Agents（助手） ==============
@dataclass
class Agent:
    """Assistant agent (can be dynamic or static)."""

    name: str
    dynamic: bool
    persona_lines: List[str] = field(default_factory=list)
    P0: Dict[str, float] = field(default_factory=lambda: {k: round(random.uniform(0.0, 1.0), 2) for k in DIMENSIONS})
    predictor: Optional[HeuristicMotivePredictor] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        for k in DIMENSIONS:
            v = self.P0.get(k)
            if v is None or not (0.0 <= v <= 1.0):
                raise ValueError(f"P0[{k}] must be in [0,1], got {v}")
        if self.dynamic:
            if self.predictor is None:
                self.predictor = HeuristicMotivePredictor(
                    llmClient(), beta=1.3, use_global_factor_weight=True, eps=0.15
                )
            self.state_tracker = PersonaStateTracker(
                P0=self.P0,
                predictor=self.predictor,
                # 积极但稳定
                target_step=0.12,
                lambda_decay=0.80,
                alpha_cap=0.55,
                gate_m_norm=0.20,
                gate_min_dims=1,
                cooldown_k=1,
                passive_reg_alpha=0.02,
                passive_reg_use_decay=True,
                global_drift=0.005,
            )
        else:
            self.state_tracker = None

    def get_current_state(self) -> Dict[str, float]:
        if self.dynamic and self.state_tracker is not None:
            return self.state_tracker.get_current_state()
        return dict(self.P0)

    def current_persona_values(self) -> Dict[str, float]:
        return self.get_current_state()

    def _anti_repeat_addendum(self, history: List[Dict[str, str]]) -> str:
        snippets = build_do_not_repeat_snippets(history, k=4, max_len=220)
        if not snippets:
            return ""
        joined = "\n".join(f"- {s}" for s in snippets)
        return (
            "DO-NOT-REPEAT SNIPPETS (assistant side):\n"
            f"{joined}\n"
            "Avoid reusing these phrases, templates, or near-paraphrases. Provide NEW content only.\n"
        )

    def system_prompt(self, history: List[Dict[str, str]]) -> str:
        P_vals = self.current_persona_values()
        base_text = (SYSTEM_PROMPT + " " + " ".join(self.persona_lines)).strip() if self.persona_lines else SYSTEM_PROMPT

        dyn_prompt = generate_dynamic_system_prompt(
            base_text=base_text,
            enable_base=True,
            vals=P_vals,
            table=big5_system_prompts_en,
        )

        extra = [self._anti_repeat_addendum(history)]
        return (dyn_prompt + "\n\n" + "\n".join([e for e in extra if e])).strip()

    def update_persona(self, context: List[Dict[str, str]]) -> None:
        if not self.dynamic or self.state_tracker is None:
            return
        try:
            self.state_tracker.step(context)
        except Exception as e:
            self.logger.error(f"Persona update failed for {self.name}: {e}")

    def _too_similar_to_recent(self, text: str, history: List[Dict[str, str]], sim_threshold: float = 0.55) -> bool:
        for m in reversed(history):
            if m.get("role") != "assistant":
                continue
            prev = (m.get("content") or "").strip()
            if not prev:
                continue
            if jaccard_sim(text, prev) >= sim_threshold:
                return True
        return False

    def respond(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        system_prompt = self.system_prompt(messages)
        payload = [{"role": "system", "content": system_prompt}]
        payload += [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

        try:
            llm = self.predictor.llm if (self.predictor and getattr(self.predictor, "llm", None)) else llmClient()
            response_text = (llm.chat_once(messages=payload, temperature=temperature) or "").strip()

            # 相似度拦截 → 二次请求“给新增点”
            if response_text and self._too_similar_to_recent(response_text, messages):
                stricter = system_prompt + (
                    "\n\nSTRICT UPDATE (assistant side):\n"
                    "- Provide ONLY NEW angles or details.\n"
                    "- Avoid paraphrasing previous assistant messages or templates.\n"
                    "- Keep a neutral, steady tone; end with exactly one question.\n"
                )
                payload[0]["content"] = stricter
                retry = (llm.chat_once(messages=payload, temperature=max(0.3, temperature - 0.2)) or "").strip()
                if retry:
                    response_text = retry
            return response_text
        except Exception as e:
            user_text = messages[-1].get("content", "") if messages else ""
            self.logger.error(f"LLM call failed for {self.name}: {e}; echo fallback.")
            return "I'm sorry, I'm having trouble generating a reply right now. You said: " + user_text


# ============== 用户代理（PromptedUserAgent） ==============
@dataclass
class PromptedUserAgent:
    """Prompt-driven user agent. Supports 'stable' and 'shifting' topics."""

    name: str
    scenario: str
    total_turns: int
    persona_lines: List[str] = field(default_factory=list)
    P0: Dict[str, float] = field(default_factory=lambda: {k: round(random.uniform(0.0, 1.0), 2) for k in DIMENSIONS})
    predictor: Optional[HeuristicMotivePredictor] = None
    first_message_override: Optional[str] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _turn_index: int = 0

    # ==== 显式“转场”设置 ====
    scenario_shift_turn: Optional[int] = None
    _last_context_switch_flag: bool = False

    def _is_shifting_enabled(self) -> bool:
        return self.scenario.lower() == "shifting"

    def _shift_turn(self) -> int:
        if self.scenario_shift_turn is not None and self.scenario_shift_turn >= 1:
            return self.scenario_shift_turn
        half = self.total_turns // 2 if self.total_turns > 0 else 1
        return max(1, half + 1)

    # ---------- 用户侧 Anti-Echo 支持 ----------
    def _too_similar_to_recent(self, text: str, history: List[Dict[str, str]], thr: float = 0.50) -> bool:
        """避免与最近助手/用户文本过近似（镜像/复述）。"""
        # 检查最近 1~3 条助手文本
        checked = 0
        for m in reversed(history):
            if m.get("role") == "assistant":
                prev = (m.get("content") or "").strip()
                if prev and jaccard_sim(text, prev) >= thr:
                    return True
                checked += 1
                if checked >= 3:
                    break
        # 检查最近一条用户文本，避免自我复述
        for m in reversed(history):
            if m.get("role") == "user":
                prev = (m.get("content") or "").strip()
                if prev and jaccard_sim(text, prev) >= thr:
                    return True
                break
        return False

    def _user_side_anti_echo_addendum(self, history: List[Dict[str, str]]) -> str:
        snippets = build_do_not_copy_from_assistant(history, k=3, max_len=200)
        if not snippets:
            return ""
        joined = "\n".join(f"- {s}" for s in snippets)
        return (
            "DO-NOT-COPY-FROM-ASSISTANT (user side):\n"
            f"{joined}\n"
            "Do NOT quote, paraphrase, or mirror the above lines. Introduce NEW details and a DIFFERENT angle.\n"
            "Start with a different opening verb/noun than any shown above.\n"
        )

    def _stricter_user_system(self, base_sys: str, history: List[Dict[str, str]]) -> str:
        hard_rules = (
            "\nSTRICT REWRITE RULES (user side):\n"
            "- Do NOT reuse bigrams/phrases from the last assistant message.\n"
            "- Add one NEW fact/example or a concrete preference.\n"
            "- Keep 1–2 sentences. End with exactly ONE question.\n"
            "- Neutral tone; no templates; no meta comments.\n"
        )
        return base_sys + "\n" + self._user_side_anti_echo_addendum(history) + hard_rules

    # ---------- 指令文本 ----------
    def _instruction_for_turn(self) -> str:
        sc = self.scenario.lower()
        if sc not in {"stable", "shifting"}:
            raise ValueError(f"Unknown scenario '{self.scenario}'")

        base_rules = (
            "You are participating in a friendly conversation. "
            "Stay neutral, concrete, and conversational. "
            "Do not repeat the same topic across turns, and do not restate earlier content from either side. "
            "Keep utterances 1–3 sentences. Avoid templates."
        )

        if (not self._is_shifting_enabled()) or (self._turn_index + 1) != self._shift_turn():
            # 正常回合：轻松日常话题
            return (
                base_rules + " "
                "Talk about light topics (music, hobbies, food, travel, daily life, work, creativity). "
                "End with exactly one relevant follow-up question."
            )
        else:
            safe_orthogonal = "; ".join(ORTHOGONAL_TOPICS)
            return (
                base_rules + " "
                "ORTHOGONAL CONTEXT SWITCH: Start a new topic that is COMPLETELY UNRELATED to any previous messages. "
                f"Choose a calm, non-sensitive domain such as: {safe_orthogonal}. "
                "Keep the same steady tone; do NOT escalate or dramatize emotions; "
                "avoid health/emergency/medical/self-harm/violence/crime/illegal advice; "
                "avoid reassurance templates. 1–3 sentences; finish with exactly one concrete question."
            )

    def _system_prompt(self, history: List[Dict[str, str]]) -> str:
        instr = self._instruction_for_turn()
        base_text = (instr + " " + " ".join(self.persona_lines)).strip() if self.persona_lines else instr
        try:
            sys_prompt = generate_dynamic_system_prompt(
                base_text=base_text, enable_base=True, vals=self.P0, table=big5_system_prompts_en
            )
        except Exception:
            sys_prompt = base_text
        # 在用户侧也加入“不要复述助手”的黑名单片段
        return sys_prompt + "\n" + self._user_side_anti_echo_addendum(history)

    def _make_shift_prefix(self) -> str:
        cues = (
            "Switching topics entirely:",
            "New, unrelated topic:",
            "Let me pivot to something different:",
            "Changing to a different domain:",
        )
        return random.choice(cues) + " "

    # ---------- 正交转场用户消息合成器 ----------
    def _compose_orthogonal_user_msg(self, topic: str) -> str:
        """
        构造“完全不同场景”的中性转场消息（2–3句，末尾1个具体问题）。
        直接返回用户要说的话，避免LLM在此回合回到原话题。
        """
        lead = self._make_shift_prefix()
        templates = [
            "{lead}I want to pivot to {topic}. It’s unrelated to what we discussed, but I find it quietly interesting. "
            "What’s your take on it?",
            "{lead}Lately I’ve been exploring {topic}. It’s been a calm change of pace and gives me new ideas. "
            "Have you looked into this area at all?",
            "{lead}Let me switch to {topic}. I like keeping the tone practical and steady while learning something new. "
            "Is this something you’ve experimented with?",
        ]
        return random.choice(templates).format(lead=lead, topic=topic)

    def pop_context_switch_flag(self) -> bool:
        f = self._last_context_switch_flag
        self._last_context_switch_flag = False
        return f

    def respond(self, history: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Turn 0: 首条强制 dataset 首句
        if self._turn_index == 0 and isinstance(self.first_message_override, str) and self.first_message_override.strip():
            self._turn_index += 1
            return self.first_message_override.strip()

        is_shift_turn_now = self._is_shifting_enabled() and (self._turn_index + 1) == self._shift_turn()
        self._last_context_switch_flag = bool(is_shift_turn_now)

        # ★ 在“正交转场回合”直接合成用户消息，确保真正跳出原话题 ★
        if is_shift_turn_now:
            topic = random.choice(ORTHOGONAL_TOPICS)
            msg = self._compose_orthogonal_user_msg(topic)
            self._turn_index += 1
            return msg

        # —— 常规回合：LLM 生成 + 用户侧 Anti-Echo（检测→重采样→fallback） ——
        base_sys = self._system_prompt(history)
        payload = [{"role": "system", "content": base_sys}] + history
        try:
            llm = self.predictor.llm if (self.predictor and getattr(self.predictor, "llm", None)) else llmClient()

            # 第一次采样
            text = (llm.chat_once(messages=payload, temperature=temperature) or "").strip()

            # 反复读闸：若过相似，则最多重采样两次
            attempts = 0
            while text and self._too_similar_to_recent(text, history, thr=0.50) and attempts < 2:
                attempts += 1
                stricter = self._stricter_user_system(base_sys, history)
                payload[0]["content"] = stricter
                text = (llm.chat_once(messages=payload, temperature=max(0.5, temperature)) or "").strip()

            if text:
                self._turn_index += 1
                return text

        except Exception as e:
            self.logger.error(f"User LLM call failed for {self.name}: {e}; fallback to templates.")

        # fallback（LLM 不可用或多次相似失败）：给一个“同主题新角度”的短句+单问
        msg = self._fallback_prompt(is_shift=False)
        self._turn_index += 1
        return msg

    def _fallback_prompt(self, is_shift: bool) -> str:
        stable = [
            "I’ve been narrowing down one practical detail—timing. Do you prefer morning or evening for most plans?",
            "I keep iterating on small workflow tweaks. What’s one tool you’ve adopted recently that actually stuck?",
            "I’m curious about low-effort upgrades that pay off. What’s a tiny change you’ve made that improved your day?",
            "I’m collecting examples of good checklists. Do you keep a personal checklist that works reliably?",
        ]
        orthogonal = [
            "Have you ever explored typography grids and how they shape readable layouts?",
            "Do you keep a note-taking system—like Zettelkasten or PARA—for your projects?",
            "What’s your approach to organizing a small space in a minimalist way?",
            "Which coffee brewing method do you prefer—pour-over, AeroPress, or espresso?",
        ]
        seq = orthogonal if is_shift else stable
        idx = (self._turn_index) % len(seq)
        self._turn_index += 1
        return seq[idx]


# ============== Simulation Core ==============
def simulate_dialogue(
    user_agent: PromptedUserAgent,
    assistant_agent: Agent,
    num_turns: int = 15,
    temperature: float = 0.7,
) -> Tuple[List[Dict[str, str]], List[Dict[str, float]], Optional[int]]:
    """
    返回:
      conversation: [{'role':..,'content':.., ['context_switch': true]}, ...]
      persona_trace: [{'t':1,'O':..,'C':..,'E':..,'A':..,'N':..}, ...]  # 每个助手回合一次
      context_switch_turn: int or None  # 用户在第几回合显式转场（以用户回合计）
    """
    conversation: List[Dict[str, str]] = []
    history_for_llm: List[Dict[str, str]] = []
    persona_trace: List[Dict[str, float]] = []
    context_switch_turn: Optional[int] = None

    for turn in range(1, num_turns + 1):
        # user
        user_msg = user_agent.respond(history_for_llm.copy(), temperature=temperature)
        user_item: Dict[str, object] = {"role": "user", "content": user_msg}
        if user_agent.pop_context_switch_flag():
            user_item["context_switch"] = True
            context_switch_turn = context_switch_turn or turn
        conversation.append(user_item)  # type: ignore[arg-type]
        history_for_llm.append({"role": "user", "content": user_msg})

        # assistant persona update（使用最新上下文进行一次 step）
        assistant_agent.update_persona(history_for_llm.copy())

        # 记录此回合用于生成回复时的当前人格（即更新后）
        cur = assistant_agent.get_current_state()
        persona_trace.append({"t": turn, **{k: float(cur[k]) for k in DIMENSIONS}})

        # assistant reply
        reply = assistant_agent.respond(history_for_llm.copy(), temperature=temperature)
        conversation.append({"role": "assistant", "content": reply})
        history_for_llm.append({"role": "assistant", "content": reply})

    return conversation, persona_trace, context_switch_turn


# ============== Specs Pre-sampling (same seed across 4 combos) ==============
@dataclass
class DialogueSpec:
    user_lines: List[str]
    assistant_lines: List[str]
    user_P0: Dict[str, float]
    assistant_P0: Dict[str, float]
    first_message: str

def presample_dialogue_specs(
    n_dialogues: int,
    persona_lines_per_agent: int,
    seed: int,
) -> List[DialogueSpec]:
    if seed is not None:
        random.seed(seed)
    _load_dataset_for_sampling()
    specs: List[DialogueSpec] = []
    for _ in range(n_dialogues):
        user_lines = sample_persona_lines_from_dataset(persona_lines_per_agent)
        assistant_lines = sample_persona_lines_from_dataset(persona_lines_per_agent)
        fm = sample_first_message_from_dataset()
        specs.append(
            DialogueSpec(
                user_lines=user_lines,
                assistant_lines=assistant_lines,
                user_P0={k: round(random.uniform(0.0, 1.0), 2) for k in DIMENSIONS},
                assistant_P0={k: round(random.uniform(0.0, 1.0), 2) for k in DIMENSIONS},
                first_message=fm,
            )
        )
    return specs


# ============== Single-condition dataset ==============
def simulate_dataset(
    specs: List[DialogueSpec],
    use_dynamic_scenario: bool,
    use_dynamic_persona: bool,
    turns_per_dialogue: int = 15,
    temperature: float = 0.7,
    combo_tag: str = "stable_static",
    index_accumulator: List[Dict[str, str]] | None = None,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    scenario = "shifting" if use_dynamic_scenario else "stable"
    logger.info(f"[simulate_dataset] scenario={scenario}, assistant_dynamic={use_dynamic_persona}, count={len(specs)}")

    plots_dir, traces_dir = _ensure_dirs_for_combo(combo_tag)

    for idx, _ in enumerate(tqdm(range(len(specs)), total=len(specs), desc=f"{scenario}|{'dyn' if use_dynamic_persona else 'static'}")):
        s = specs[idx]
        dialogue_id = str(uuid.uuid4())

        user_agent = PromptedUserAgent(
            name=f"User-{idx}",
            scenario=scenario,
            total_turns=turns_per_dialogue,
            persona_lines=s.user_lines,
            P0=s.user_P0,
            first_message_override=s.first_message,
        )
        assistant_agent = Agent(
            name=f"Assistant-{idx}",
            dynamic=use_dynamic_persona,
            persona_lines=s.assistant_lines,
            P0=s.assistant_P0,
            predictor=None if use_dynamic_persona else None,
        )

        conv, trace, ctx_switch_turn = simulate_dialogue(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            num_turns=turns_per_dialogue,
            temperature=temperature,
        )

        trace_path = os.path.join(traces_dir, f"{dialogue_id}.json")
        save_persona_trace_json(trace, trace_path)

        plot_title = f"{combo_tag} | {dialogue_id}"
        plot_path = os.path.join(plots_dir, f"{dialogue_id}.png")
        plot_persona_trace(trace, plot_title, plot_path)

        result_item = {
            "dialogue_id": dialogue_id,
            "scenario": scenario,
            "dynamic": use_dynamic_persona,
            "assistant_P0": s.assistant_P0,
            "user_P0": s.user_P0,
            "conversation": conv,
            "user_persona": s.user_lines,
            "assistant_persona": s.assistant_lines,
            "first_message": s.first_message,
            "trace_file": os.path.relpath(trace_path, OUTPUT_ROOT),
            "plot_file": os.path.relpath(plot_path, OUTPUT_ROOT),
            "context_switch_turn": ctx_switch_turn,
        }
        results.append(result_item)

        if index_accumulator is not None:
            index_accumulator.append({
                "dialogue_id": dialogue_id,
                "combo": combo_tag,
                "trace_file": result_item["trace_file"],
                "plot_file": result_item["plot_file"],
            })

    return results


# ============== Full 4-combo experiment ==============
def simulate_experiment_4combos(
    n_per_combo: int = 50,
    persona_lines_per_agent: int = 3,
    turns_per_dialogue: int = 15,
    temperature: float = 0.7,
    seed: int = 42,
) -> Dict[str, List[Dict[str, object]]]:
    logger.info(f"[simulate_experiment_4combos] n_per_combo={n_per_combo}, seed={seed}")
    specs = presample_dialogue_specs(n_dialogues=n_per_combo, persona_lines_per_agent=persona_lines_per_agent, seed=seed)

    outputs: Dict[str, List[Dict[str, object]]] = {}
    combos = [
        ("stable_static",   False, False),
        ("stable_dynamic",  False, True),
        ("shifting_static", True,  False),
        ("shifting_dynamic",True,  True),
    ]

    index_records: List[Dict[str, str]] = []

    for tag, sc_flag, dyn_flag in combos:
        outputs[tag] = simulate_dataset(
            specs=specs,
            use_dynamic_scenario=sc_flag,
            use_dynamic_persona=dyn_flag,
            turns_per_dialogue=turns_per_dialogue,
            temperature=temperature,
            combo_tag=tag,
            index_accumulator=index_records,
        )

    try:
        with open(os.path.join(OUTPUT_ROOT, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index_records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[index] failed to save outputs/index.json: {e}")

    return outputs


# ============== CLI demo ==============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # 示例：每组 2 条，可调大到 50 复现实验
        result = simulate_experiment_4combos(
            n_per_combo=1,
            persona_lines_per_agent=3,
            turns_per_dialogue=15,
            temperature=0.7,
            seed=1234,
        )
        total_out = os.path.join(OUTPUT_ROOT, "simulated_persona_dialogues_4combos.json")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(total_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {total_out}\nIndex at {os.path.join(OUTPUT_ROOT, 'index.json')}")
    except Exception as e:
        print(f"Simulation failed: {e}")
