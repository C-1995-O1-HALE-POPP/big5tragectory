# -*- coding: utf-8 -*-
"""
simulate_personas.py  —— 并发版本
====================

- 从 HuggingFace `Cynaptics/persona-chat` 采样 persona 句与首句（无本地 fallback）。
- 用户“场景动态/稳定”与助手“人格动态/静态”两开关 → 四种组合。
- 同一随机种子统一预采样 N 组对话规格，在四种组合中复用，保证可比对齐。
- 助手人格的动态调整使用 PersonaStateTracker。
- 用户由 PromptedUserAgent 生成消息；首句强制为预采样的 dataset 首句。

新增（Orthogonal Switch + Neutral Tone + Anti-Echo for User）同原版。

并发改造要点：
- 组合内对话任务用 ThreadPoolExecutor 并发执行（可配置 max_workers）。
- 可选对 4 个组合也并发（parallelize_combos）。
- LLM 调用用全局信号量 LLM_GATE（默认 4 并发，可改环境变量 LLM_CONCURRENCY）。
- 绘图保存 PNG 用 PLOT_LOCK 串行保护。
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import uuid
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ============== 全局并发控制 ==============
import os as _os
_LLM_CONCURRENCY_DEFAULT = int(_os.getenv("LLM_CONCURRENCY", "4"))
LLM_GATE = threading.Semaphore(max(1, _LLM_CONCURRENCY_DEFAULT))
PLOT_LOCK = threading.Lock()

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
    with PLOT_LOCK:
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
            base_text=base_text, enable_base=True, vals=P_vals, table=big5_system_prompts_en
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
            with LLM_GATE:
                response_text = (llm.chat_once(messages=payload, temperature=temperature) or "").strip()
            if response_text and self._too_similar_to_recent(response_text, messages):
                stricter = system_prompt + (
                    "\n\nSTRICT UPDATE (assistant side):\n"
                    "- Provide ONLY NEW angles or details.\n"
                    "- Avoid paraphrasing previous assistant messages or templates.\n"
                    "- Keep a neutral, steady tone; end with exactly one question.\n"
                )
                payload[0]["content"] = stricter
                with LLM_GATE:
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
    name: str
    scenario: str
    total_turns: int
    persona_lines: List[str] = field(default_factory=list)
    P0: Dict[str, float] = field(default_factory=lambda: {k: round(random.uniform(0.0, 1.0), 2) for k in DIMENSIONS})
    predictor: Optional[HeuristicMotivePredictor] = None
    first_message_override: Optional[str] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _turn_index: int = 0

    scenario_shift_turn: Optional[int] = None
    _last_context_switch_flag: bool = False

    def _is_shifting_enabled(self) -> bool:
        return self.scenario.lower() == "shifting"

    def _shift_turn(self) -> int:
        if self.scenario_shift_turn is not None and self.scenario_shift_turn >= 1:
            return self.scenario_shift_turn
        half = self.total_turns // 2 if self.total_turns > 0 else 1
        return max(1, half + 1)

    def _too_similar_to_recent(self, text: str, history: List[Dict[str, str]], thr: float = 0.50) -> bool:
        checked = 0
        for m in reversed(history):
            if m.get("role") == "assistant":
                prev = (m.get("content") or "").strip()
                if prev and jaccard_sim(text, prev) >= thr:
                    return True
                checked += 1
                if checked >= 3:
                    break
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
            return (
                base_rules + " "
                "Talk about light topics (music, hobbies, food, travel, daily life, work, creativity). "
                "End with exactly one relevant follow-up question."
                ""
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
        return sys_prompt + "\n" + self._user_side_anti_echo_addendum(history)

    def _make_shift_prefix(self) -> str:
        cues = (
            "Switching topics entirely:",
            "New, unrelated topic:",
            "Let me pivot to something different:",
            "Changing to a different domain:",
        )
        return random.choice(cues) + " "

    def _compose_orthogonal_user_msg(self, topic: str) -> str:
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
        if self._turn_index == 0 and isinstance(self.first_message_override, str) and self.first_message_override.strip():
            self._turn_index += 1
            return self.first_message_override.strip()

        is_shift_turn_now = self._is_shifting_enabled() and (self._turn_index + 1) == self._shift_turn()
        self._last_context_switch_flag = bool(is_shift_turn_now)

        if is_shift_turn_now:
            topic = random.choice(ORTHOGONAL_TOPICS)
            msg = self._compose_orthogonal_user_msg(topic)
            self._turn_index += 1
            return msg

        base_sys = self._system_prompt(history)
        payload = [{"role": "system", "content": base_sys}] + history
        try:
            llm = llmClient()
            with LLM_GATE:
                text = (llm.chat_once(messages=payload, temperature=temperature) or "").strip()

            attempts = 0
            while text and self._too_similar_to_recent(text, history, thr=0.50) and attempts < 2:
                attempts += 1
                stricter = self._stricter_user_system(base_sys, history)
                payload[0]["content"] = stricter
                with LLM_GATE:
                    text = (llm.chat_once(messages=payload, temperature=max(0.5, temperature)) or "").strip()

            if text:
                self._turn_index += 1
                return text

        except Exception as e:
            self.logger.error(f"User LLM call failed for {self.name}: {e}; fallback to templates.")

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
    conversation: List[Dict[str, str]] = []
    history_for_llm: List[Dict[str, str]] = []
    persona_trace: List[Dict[str, float]] = []
    context_switch_turn: Optional[int] = None

    for turn in range(1, num_turns + 1):
        user_msg = user_agent.respond(history_for_llm.copy(), temperature=temperature)
        user_item: Dict[str, object] = {"role": "user", "content": user_msg}
        if user_agent.pop_context_switch_flag():
            user_item["context_switch"] = True
            context_switch_turn = context_switch_turn or turn
        conversation.append(user_item)  # type: ignore[arg-type]
        history_for_llm.append({"role": "user", "content": user_msg})

        assistant_agent.update_persona(history_for_llm.copy())

        cur = assistant_agent.get_current_state()
        persona_trace.append({"t": turn, **{k: float(cur[k]) for k in DIMENSIONS}})

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

# ============== 单个对话任务（供线程池调用） ==============
def _run_one_dialogue_task(
    idx: int,
    s: DialogueSpec,
    turns_per_dialogue: int,
    temperature: float,
    combo_tag: str,
    plots_dir: str,
    traces_dir: str,
) -> Dict[str, object]:
    dialogue_id = str(uuid.uuid4())

    user_agent = PromptedUserAgent(
        name=f"User-{idx}",
        scenario=("shifting" if "shifting" in combo_tag else "stable"),
        total_turns=turns_per_dialogue,
        persona_lines=s.user_lines,
        P0=s.user_P0,
        first_message_override=s.first_message,
    )
    assistant_agent = Agent(
        name=f"Assistant-{idx}",
        dynamic=("dynamic" in combo_tag),
        persona_lines=s.assistant_lines,
        P0=s.assistant_P0,
        predictor=None,
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

    return {
        "dialogue_id": dialogue_id,
        "scenario": ("shifting" if "shifting" in combo_tag else "stable"),
        "dynamic": ("dynamic" in combo_tag),
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

# ============== 单组合数据集（并发版本） ==============
def simulate_dataset(
    specs: List[DialogueSpec],
    use_dynamic_scenario: bool,
    use_dynamic_persona: bool,
    turns_per_dialogue: int = 15,
    temperature: float = 0.7,
    combo_tag: str = "stable_static",
    index_accumulator: List[Dict[str, str]] | None = None,
    max_workers_dialogues: int = 8,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    scenario = "shifting" if use_dynamic_scenario else "stable"
    tag = f"{scenario}_{'dynamic' if use_dynamic_persona else 'static'}"
    if combo_tag != tag:
        combo_tag = tag

    logger.info(f"[simulate_dataset] scenario={scenario}, assistant_dynamic={use_dynamic_persona}, "
                f"count={len(specs)}, workers={max_workers_dialogues}")

    plots_dir, traces_dir = _ensure_dirs_for_combo(combo_tag)

    with ThreadPoolExecutor(max_workers=max_workers_dialogues) as ex:
        futures = []
        for idx, s in enumerate(specs):
            futures.append(ex.submit(
                _run_one_dialogue_task, idx, s, turns_per_dialogue, temperature, combo_tag, plots_dir, traces_dir
            ))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{combo_tag}"):
            try:
                item = fut.result()
                results.append(item)
                if index_accumulator is not None:
                    index_accumulator.append({
                        "dialogue_id": item["dialogue_id"],
                        "combo": combo_tag,
                        "trace_file": item["trace_file"],
                        "plot_file": item["plot_file"],
                    })
            except Exception as e:
                logger.error(f"[simulate_dataset] one dialogue failed: {e}")

    return results

# ============== Full 4-combo experiment（可并发组合） ==============
def simulate_experiment_4combos(
    n_per_combo: int = 50,
    persona_lines_per_agent: int = 3,
    turns_per_dialogue: int = 15,
    temperature: float = 0.7,
    seed: int = 42,
    max_workers_dialogues: int = 8,
    parallelize_combos: bool = False,  # 设 True 可并发 4 个组合（注意 API 限流）
    max_workers_combos: int = 4,
) -> Dict[str, List[Dict[str, object]]]:
    logger.info(f"[simulate_experiment_4combos] n_per_combo={n_per_combo}, seed={seed}")
    specs = presample_dialogue_specs(
        n_dialogues=n_per_combo, persona_lines_per_agent=persona_lines_per_agent, seed=seed
    )

    outputs: Dict[str, List[Dict[str, object]]] = {}
    combos = [
        ("stable_static",   False, False),
        ("stable_dynamic",  False, True),
        ("shifting_static", True,  False),
        ("shifting_dynamic",True,  True),
    ]
    index_records: List[Dict[str, str]] = []

    def _run_combo(tag: str, sc_flag: bool, dyn_flag: bool):
        res = simulate_dataset(
            specs=specs,
            use_dynamic_scenario=sc_flag,
            use_dynamic_persona=dyn_flag,
            turns_per_dialogue=turns_per_dialogue,
            temperature=temperature,
            combo_tag=tag,
            index_accumulator=index_records,
            max_workers_dialogues=max_workers_dialogues,
        )
        return tag, res

    if parallelize_combos:
        with ThreadPoolExecutor(max_workers=max_workers_combos) as ex:
            futures = [ex.submit(_run_combo, *c) for c in combos]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="combos"):
                try:
                    tag, res = fut.result()
                    outputs[tag] = res
                except Exception as e:
                    logger.error(f"[simulate_experiment_4combos] combo failed: {e}")
    else:
        for c in combos:
            tag, res = _run_combo(*c)
            outputs[tag] = res

    try:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(os.path.join(OUTPUT_ROOT, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index_records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[index] failed to save outputs/index.json: {e}")

    return outputs

# ============== CLI demo ==============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # 示例：每组 50 条，可调整
        result = simulate_experiment_4combos(
            n_per_combo=50,
            persona_lines_per_agent=3,
            turns_per_dialogue=15,
            temperature=0.7,
            seed=1234,
            max_workers_dialogues=8,   # 组合内并发对话数
            parallelize_combos=False,  # 如需四组合并发，改为 True（配合 LLM_CONCURRENCY）
            max_workers_combos=4,
        )
        total_out = os.path.join(OUTPUT_ROOT, "simulated_persona_dialogues_4combos.json")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(total_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {total_out}\nIndex at {os.path.join(OUTPUT_ROOT, 'index.json')}")
    except Exception as e:
        print(f"Simulation failed: {e}")
