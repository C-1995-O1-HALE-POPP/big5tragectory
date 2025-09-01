# -*- coding: utf-8 -*-
"""
simulate_personas.py  —— 并发版本（User 无 Big5，话题转换必带情感突变）
=====================================================================

- 从 HuggingFace `Cynaptics/persona-chat` 采样 persona 句与首句（无本地 fallback）。
- 用户“场景动态/稳定”与助手“人格动态/静态”两开关 → 四种组合。
- 同一随机种子统一预采样 N 组对话规格，在四种组合中复用，保证可比对齐。
- 助手人格由 generate_persona_system_prompt(persona_id=...) 控制（无需传 personas 列表）。
- 用户（PromptedUserAgent）仅自然接话；当执行“正交话题切换”时，**总是**伴随“情感突变”。

集成：Emotion Mode Prompt Utils（仅修改“用户智能体”以实现持久情绪模式）
- 用户侧 `emotion_mode`（如 "sadness"/"anxiety"），一旦切换设定即对后续轮次持久生效，通过 system prompt 强约束。
- system prompt 使用 prompt.py 的工具函数拼接情绪段落；必要时自动补一条短情感句，保证“恰好一个情绪线索”更稳定。
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
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib  # 用于稳定哈希种子

# ----- matplotlib（无显示环境安全）-----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

# ============== Project Modules ==============
from prompt import (
    generate_persona_system_prompt,
    generate_persona_traits,
    # Emotion Mode Prompt Utils
    build_user_base_rules,
    build_user_system_prompt_with_emotion,
    generate_emotion_sentence,
    coarse_valence,
)
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker, DIMENSIONS  # 仅助手用到 DIMENSIONS

LLM = llmClient()
PREDICTOR = HeuristicMotivePredictor(
    LLM, beta=2.0, use_global_factor_weight=True, eps=0.15
)

# ============== 全局并发控制 ==============
import os as _os
_LLM_CONCURRENCY_DEFAULT = int(_os.getenv("LLM_CONCURRENCY", "4"))
LLM_GATE = threading.Semaphore(max(1, _LLM_CONCURRENCY_DEFAULT))
PLOT_LOCK = threading.Lock()

# ============== 稳定随机种子工具（修复 tuple seed 报错） ==============
def _stable_seed(*parts) -> int:
    """将任意可打印的 parts 稳定映射为 32bit 正整数种子（跨平台可复现）"""
    s = "|".join(map(str, parts))
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**32)

# ============== Persona IDs（内置） ==============
DEFAULT_PERSONA_IDS = ["01", "02", "03", "04", "05", "06", "07", "08"]

def normalize_persona_ids(persona_ids_arg: Optional[str],
                          num_personas: Optional[int],
                          seed: int) -> List[str]:
    base_pool = list(DEFAULT_PERSONA_IDS)
    if persona_ids_arg:
        raw = [x.strip() for x in persona_ids_arg.split(",") if x.strip()]
        seen, cleaned = set(), []
        for x in raw:
            if x in base_pool and x not in seen:
                cleaned.append(x); seen.add(x)
        if not cleaned:
            raise ValueError(f"--persona-ids gave no valid ids (valid: {base_pool})")
        return cleaned
    k = num_personas if (isinstance(num_personas, int) and num_personas > 0) else len(base_pool)
    if k > len(base_pool):
        raise ValueError(f"--num-personas={k} exceeds available pool {len(base_pool)}")
    rng = random.Random(seed or 0)
    selected = sorted(rng.sample(base_pool, k))  # 排序仅为输出稳定
    return selected

def pick_persona_id_from_pool(pool: List[str], seed: int, index: int) -> str:
    if not pool:
        raise ValueError("persona pool is empty")
    rr = index % len(pool)
    rng = random.Random(_stable_seed("persona_pick", seed, index, len(pool)))
    if rng.random() < 0.15:
        return rng.choice(pool)
    return pool[rr]

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

# ============== Emotional Events（用于话题切换时的情感突变） ==============
EMOTION_EVENTS = [
    {"id": "lose_love", "valence": "negative",
     "templates": [
        "I just broke up and I’m feeling gutted.",
        "My relationship just collapsed—heart in pieces, to be honest.",
        "I got dumped and the bottom kind of fell out of my day.",
        "I’m reeling from a breakup; everything feels too loud right now.",
     ]},
    {"id": "won_lottery", "valence": "positive",
     "templates": [
        "I just won a lottery prize and I’m buzzing hard.",
        "I hit an unexpected windfall and I’m practically floating.",
        "I won some money—adrenaline’s spiking in the best way.",
        "I’m celebrating a lucky break; it feels surreal.",
     ]},
    {"id": "work_praise", "valence": "positive",
     "templates": [
        "My boss publicly praised me and I’m riding the high.",
        "I nailed a brutal task and I’m fiercely proud.",
        "I got recognition at work—confidence is peaking.",
        "That win at work was electric; still grinning.",
     ]},
    {"id": "deadline_crunch", "valence": "negative",
     "templates": [
        "A deadline got yanked forward and my stress needle snapped.",
        "I’m drowning in time pressure; shoulders are locked up.",
        "The schedule slipped, and frustration is spiking.",
        "Everything is on fire timeline-wise; I’m tense.",
     ]},
    {"id": "mixed_news", "valence": "mixed",
     "templates": [
        "I got bittersweet news—good spark with a sharp edge.",
        "It’s a weird day: win in one hand, worry in the other.",
        "Something great landed… with strings that tug the other way.",
        "I’m split—happy and uneasy at the same time.",
     ]},
    {"id": "health_scare_minor", "valence": "negative",
     "templates": [
        "I had a minor health scare and it rattled me.",
        "A quick clinic visit spiked my anxiety, even if it’s okay now.",
        "Something felt off earlier; I’m still a bit shaken.",
        "Got a precautionary call from the doc; nerves jangling.",
     ]},
    {"id": "reunion_good", "valence": "positive",
     "templates": [
        "I reconnected with an old friend and my chest feels light.",
        "Ran into someone I’ve missed for years—pure warmth.",
        "An overdue reunion just happened; joy’s overflowing a bit.",
        "Old friend, new spark—today glows.",
     ]},
]

def pick_emotion_event(rng: random.Random) -> dict:
    return rng.choice(EMOTION_EVENTS)

def render_emotion_event(ev: dict, rng: random.Random) -> str:
    return rng.choice(ev["templates"])

# ============== Agents（助手） ==============
@dataclass
class Agent:
    name: str
    dynamic: bool
    persona_id: str
    P0: Optional[Dict[str, float]] = None
    Pt: Optional[Dict[str, float]] = None
    predictor: Optional[HeuristicMotivePredictor] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        if not self.persona_id or not isinstance(self.persona_id, str):
            raise ValueError("persona_id must be a non-empty string")
        if self.P0 is None:
            self.P0 = generate_persona_traits(self.persona_id)
        for k in DIMENSIONS:
            v = self.P0.get(k)
            if v is None or not (0.0 <= v <= 1.0):
                raise ValueError(f"P0[{k}] must be in [0,1], got {v}")
        if self.dynamic:
            if self.predictor is None:
                self.predictor = PREDICTOR
            self.state_tracker = PersonaStateTracker(
                P0=self.P0,
                predictor=self.predictor,
                target_step=0.3,
                lambda_decay=0.80,
                alpha_cap=1.0,
                gate_m_norm=0.10,
                gate_min_dims=1,
                cooldown_k=1,
                passive_reg_alpha=0.002,
                passive_reg_use_decay=True,
                global_drift=0.001,
            )
        else:
            self.state_tracker = None
        self.Pt = self.P0 if self.P0 else None

    def get_current_state(self) -> Dict[str, float]:
        if self.dynamic and self.state_tracker is not None:
            return self.state_tracker.get_current_state()
        return dict(self.P0)  # type: ignore

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
        base = generate_persona_system_prompt(
            persona_id=self.persona_id,
            Pt=self.get_current_state(),
            include_base_task_line=True,
            include_big5_details=True,
        )
        extra = self._anti_repeat_addendum(history)
        return (base + ("\n\n" + extra if extra else "")).strip()

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

# ============== 用户代理（PromptedUserAgent，无 Big5，转场必带情感） ==============
@dataclass
class PromptedUserAgent:
    name: str
    scenario: str
    total_turns: int
    persona_lines: List[str] = field(default_factory=list)  # 仅作为语气素材，可为空
    first_message_override: Optional[str] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _turn_index: int = 0

    scenario_shift_turn: Optional[int] = None
    _last_context_switch_flag: bool = False
    _last_emotion_meta: Optional[dict] = None  # 记录最近的情感事件

    # 持久情绪模式：如 "sadness" / "anxiety" / "joy" / "neutral"
    emotion_mode: Optional[str] = None

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

    # ====== 使用 Emotion Mode Prompt Utils 构建 system prompt ======
    def _system_prompt(self, history: List[Dict[str, str]]) -> str:
        anti = build_do_not_copy_from_assistant(history, k=3, max_len=200)
        base = build_user_base_rules(lang="en")
        return build_user_system_prompt_with_emotion(
            base_rules=base,
            emotion_mode=self.emotion_mode,   # "sadness"/"anxiety"/"joy"/"neutral"/None
            anti_echo_snippets=anti,
            persona_lines=self.persona_lines,
            lang="en",
        )

    def _make_shift_prefix(self) -> str:
        cues = (
            "Switching topics entirely:",
            "New, unrelated topic:",
            "Let me pivot to something different:",
            "Changing to a different domain:",
        )
        return random.choice(cues) + " "

    def _compose_shift_with_emotion(self, topic: str) -> Tuple[str, Optional[dict]]:
        """
        组合：情感事件 + 正交话题切换（必含情感），控制在 1–3 句内，并以一个问题结尾。
        返回 (文本, 事件元数据)
        """
        lead = self._make_shift_prefix()
        rng = random.Random(_stable_seed("emotion", self._turn_index, topic))
        ev = pick_emotion_event(rng)
        emotion_txt = render_emotion_event(ev, rng)
        meta = {"emotion_event_id": ev["id"], "valence": ev["valence"]}

        questions = [
            "What’s your take on it?",
            "How would you approach it?",
            "Any quick thoughts on that?",
        ]
        q = rng.choice(questions)

        text = f"{lead}{emotion_txt} {q}"
        return text, meta

    def pop_context_switch_flag(self) -> bool:
        f = self._last_context_switch_flag
        self._last_context_switch_flag = False
        return f

    def pop_emotion_meta(self) -> Optional[dict]:
        m = self._last_emotion_meta
        self._last_emotion_meta = None
        return m

    # —— 便捷：使用 coarse_valence（来自 prompt.py）作为粗情感检测 —— #
    def _coarse_valence(self, s: str) -> Tuple[str, float]:
        return coarse_valence(s)

    def _ensure_one_emotion_cue(self, text: str) -> str:
        """
        若处于 emotion_mode，但文本不含明显情绪线索，则补一条轻量句子（不加感叹号/emoji）。
        仅在普通回合使用；切换回合已自带情感句。
        """
        if not self.emotion_mode:
            return text
        v, _ = self._coarse_valence(text)
        # 如果已经不是中性（或已有线索），就不强加
        if v in ("negative", "positive", "mixed"):
            return text
        # 追加一条“subtle”风格的短语句
        cue = generate_emotion_sentence(self.emotion_mode, style="subtle", lang="en")
        # 避免过长：若已有问句，则把 cue 放在问句前；否则放末尾
        if "?" in text:
            parts = text.rsplit("?", 1)
            return (parts[0].strip() + ". " + cue.strip() + "? " + parts[1].strip()).strip()
        return (text.rstrip(". ") + ". " + cue).strip()

    def respond(self, history: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # 第一句：若有 dataset 首句，直接用之
        if self._turn_index == 0 and isinstance(self.first_message_override, str) and self.first_message_override.strip():
            self._turn_index += 1
            return self.first_message_override.strip()

        # 是否进入“正交话题 + 情感突变”回合
        is_shift_turn_now = self._is_shifting_enabled() and (self._turn_index + 1) == self._shift_turn()
        self._last_context_switch_flag = bool(is_shift_turn_now)

        if is_shift_turn_now:
            topic = random.choice(ORTHOGONAL_TOPICS)
            msg, meta = self._compose_shift_with_emotion(topic)

            # 一次性设定持久情绪模式（仅根据你需要的映射绑定）
            if meta and isinstance(meta, dict):
                ev_id = meta.get("emotion_event_id")
                if ev_id == "lose_love":
                    self.emotion_mode = "sadness"
                elif ev_id == "deadline_crunch":
                    self.emotion_mode = "anxiety"
                elif ev_id == "health_scare_minor":
                    self.emotion_mode = "anxiety"
                elif ev_id in ("won_lottery", "work_praise", "reunion_good"):
                    self.emotion_mode = "joy"
                elif ev_id == "mixed_news":
                    self.emotion_mode = "neutral"
                else:
                    self.emotion_mode = "neutral"

            self._last_emotion_meta = meta
            self._turn_index += 1
            return msg

        # 常规自然接话（system prompt 会根据 emotion_mode 持续加压）
        base_sys = self._system_prompt(history)
        payload = [{"role": "system", "content": base_sys}] + history
        try:
            llm = LLM
            with LLM_GATE:
                text = (llm.chat_once(messages=payload, temperature=temperature) or "").strip()

            attempts = 0
            while text and self._too_similar_to_recent(text, history, thr=0.50) and attempts < 2:
                attempts += 1
                stronger = base_sys + (
                    "\nSTRICT REWRITE RULES (user side):\n"
                    "- Do NOT reuse bigrams/phrases from the last assistant message.\n"
                    "- Add one NEW fact/example or a concrete preference.\n"
                    "- Keep 1–2 sentences. End with at most ONE question.\n"
                    "- No meta comments.\n"
                )
                payload[0]["content"] = stronger
                with LLM_GATE:
                    text = (llm.chat_once(messages=payload, temperature=max(0.5, temperature)) or "").strip()

            # 若处于情绪模式但未体现线索，补一条短情感句
            if self.emotion_mode and text:
                text = self._ensure_one_emotion_cue(text)

            self._turn_index += 1
            return text if text else "Could you clarify a bit? I might have missed a detail."

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
    num_turns: int = 10,
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
            emo = user_agent.pop_emotion_meta()
            if emo:
                # 附带一次粗判，便于离线分析
                val, inten = user_agent._coarse_valence(user_msg)
                emo["detected_valence"] = val
                emo["detected_intensity"] = round(float(inten), 3)
                user_item["emotion_event"] = emo
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
    first_message: str
    assistant_persona_id: str

def presample_dialogue_specs(
    n_dialogues: int,
    persona_lines_per_agent: int,
    seed: int,
    persona_pool: List[str],
) -> List[DialogueSpec]:
    if seed is not None:
        random.seed(seed)
    _load_dataset_for_sampling()

    specs: List[DialogueSpec] = []
    for i in range(n_dialogues):
        user_lines = sample_persona_lines_from_dataset(persona_lines_per_agent)
        fm = sample_first_message_from_dataset()
        persona_id = pick_persona_id_from_pool(persona_pool, seed or 0, i)
        specs.append(
            DialogueSpec(
                user_lines=user_lines,
                first_message=fm,
                assistant_persona_id=persona_id,
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
        first_message_override=s.first_message,
    )
    assistant_agent = Agent(
        name=f"Assistant-{idx}",
        dynamic=("dynamic" in combo_tag),
        persona_id=s.assistant_persona_id,
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

    plot_title = f"{combo_tag} | {dialogue_id} | persona={s.assistant_persona_id}"
    plot_path = os.path.join(plots_dir, f"{dialogue_id}.png")
    plot_persona_trace(trace, plot_title, plot_path)

    return {
        "dialogue_id": dialogue_id,
        "scenario": ("shifting" if "shifting" in combo_tag else "stable"),
        "dynamic": ("dynamic" in combo_tag),
        "assistant_persona_id": s.assistant_persona_id,
        "conversation": conv,
        "user_persona": s.user_lines,
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
    turns_per_dialogue: int = 10,
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
                        "assistant_persona_id": item.get("assistant_persona_id", ""),
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
    parallelize_combos: bool = False,
    max_workers_combos: int = 4,
    persona_ids_arg: Optional[str] = None,
    num_personas: Optional[int] = None,
) -> Dict[str, List[Dict[str, object]]]:
    logger.info(f"[simulate_experiment_4combos] n_per_combo={n_per_combo}, seed={seed}")

    persona_pool = normalize_persona_ids(persona_ids_arg, num_personas, seed)
    logger.info(f"[personas] pool={persona_pool}")

    specs = presample_dialogue_specs(
        n_dialogues=n_per_combo,
        persona_lines_per_agent=persona_lines_per_agent,
        seed=seed,
        persona_pool=persona_pool,
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

# ============== CLI ==============
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("simulate_personas 4-combo concurrent simulator (User without Big5; emotional shift on topic switch)")
    p.add_argument("--n-per-combo", type=int, default=25, help="每个组合的对话数量")
    p.add_argument("--persona-lines-per-agent", type=int, default=3, help="每个用户代理从数据集中抽取的 persona 描述句数量（只影响语气）")
    p.add_argument("--turns", type=int, default=10, help="每段对话的轮数（user+assistant 计为一轮的 user 发言数）")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-workers-dialogues", type=int, default=8, help="单组合内并发对话线程数")
    p.add_argument("--parallelize-combos", action="store_true", help="并发运行四个组合（注意配合 LLM_CONCURRENCY）")
    p.add_argument("--max-workers-combos", type=int, default=4)

    # 人设选择（助手侧）
    p.add_argument("--persona-ids", type=str, default=None,
                   help='显式指定人设 ID 列表，形如 "01,03,05"；若提供则优先使用。')
    p.add_argument("--num-personas", type=int, default=None,
                   help="从内置池中抽取的人设数量（受 seed 控制）；未提供则默认抽取全部。")

    return p

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()

    result = simulate_experiment_4combos(
        n_per_combo=args.n_per_combo,
        persona_lines_per_agent=args.persona_lines_per_agent,
        turns_per_dialogue=args.turns,
        temperature=args.temperature,
        seed=args.seed,
        max_workers_dialogues=args.max_workers_dialogues,
        parallelize_combos=args.parallelize_combos,
        max_workers_combos=args.max_workers_combos,
        persona_ids_arg=args.persona_ids,
        num_personas=args.num_personas,
    )

    total_out = os.path.join(OUTPUT_ROOT, "simulated_persona_dialogues_4combos.json")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(total_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved to {total_out}\nIndex at {os.path.join(OUTPUT_ROOT, 'index.json')}")

if __name__ == "__main__":
    main()
