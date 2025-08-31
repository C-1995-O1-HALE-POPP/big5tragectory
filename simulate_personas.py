# -*- coding: utf-8 -*-
"""
simulate_personas.py
====================

- 从 HuggingFace `Cynaptics/persona-chat` 采样 persona 句与首句（无本地 fallback）。
- 用户“场景动态/稳定”与助手“人格动态/静态”两开关 → 四种组合。
- 同一个随机种子先统一预采样 50 组对话规格，在四种组合中复用，保证可比对齐。
- 助手人格的动态调整使用 PersonaStateTracker。
- 用户由 PromptedUserAgent 生成消息（稳定/转场），首句强制为预采样的 dataset 首句。

新增特性：
- 为每次对话生成 uuid（四种组合各自独立）。
- 逐回合记录助手人格 OCEAN 轨迹并导出到独立 JSON 文件。
- 为每次对话绘制一张人格维度波动图（PNG）。
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
import uuid
import os
from math import isfinite

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
from prompt import big5_system_prompts_en, SYSTEM_PROMPT, generate_system_prompt  # noqa: F401
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker, DIMENSIONS


# ============== I/O & Plot Utils ==============
OUTPUT_ROOT = os.path.join(".", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "persona_plots")
TRACES_DIR = os.path.join(OUTPUT_ROOT, "persona_traces")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TRACES_DIR, exist_ok=True)

def _ensure_dirs_for_combo(combo_tag: str) -> Tuple[str, str]:
    """Return (plots_subdir, traces_subdir) for a combo, ensuring creation."""
    plots = os.path.join(PLOTS_DIR, combo_tag)
    traces = os.path.join(TRACES_DIR, combo_tag)
    os.makedirs(plots, exist_ok=True)
    os.makedirs(traces, exist_ok=True)
    return plots, traces

def save_persona_trace_json(trace: List[Dict[str, float]], out_path: str) -> None:
    """Save per-turn OCEAN trace to JSON: [{'t': 1, 'O':..,'C':..,'E':..,'A':..,'N':..}, ...]."""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[trace] failed to save {out_path}: {e}")

def plot_persona_trace(trace: List[Dict[str, float]], title: str, out_path: str) -> None:
    """Plot OCEAN over turns; one line per trait."""
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
    """Try several schema variants to get the first user-like message."""
    dlg = row.get("dialogue")
    if isinstance(dlg, list) and dlg:
        # Common case: list of strings
        if isinstance(dlg[0], str):
            return dlg[0].strip()
        # Some variants: list of dicts with text fields
        if isinstance(dlg[0], dict):
            for k in ("text", "utterance", "content"):
                if k in dlg[0] and isinstance(dlg[0][k], str):
                    return dlg[0][k].strip()
    # Some forks nest dialogues another way
    for key in ("first_utterance", "first_message"):
        if isinstance(row.get(key), str):
            return row[key].strip()
    return None

def _load_dataset_for_sampling(max_rows: int = 8000) -> None:
    """Load `Cynaptics/persona-chat` (train, streaming) and cache persona/first-msg.

    Raises on any failure. No local fallbacks.
    """
    global _DATASET_LOADED, _DATASET_PERSONA_CACHE, _DATASET_FIRST_MSG_CACHE
    if _DATASET_LOADED:
        return
    logger.info("[dataset] loading Cynaptics/persona-chat (train, streaming=True)")
    ds_iter = load_dataset("Cynaptics/persona-chat", split="train", streaming=True)
    persona_lines: List[str] = []
    first_msgs: List[str] = []
    for i, row in enumerate(ds_iter, 1):
        # persona sentences: prefer persona_b, fallback persona_a / persona
        plist = row.get("persona_b") or row.get("persona_a") or row.get("persona") or []
        if isinstance(plist, (list, tuple)):
            for p in plist:
                if isinstance(p, str) and p.strip():
                    persona_lines.append(p.strip())
        # first message
        fm = _extract_first_message(row)
        if fm:
            first_msgs.append(fm)
        if i >= max_rows:
            break
    # Deduplicate & basic clean
    _DATASET_PERSONA_CACHE = list({s for s in persona_lines if s})
    _DATASET_FIRST_MSG_CACHE = list({s for s in first_msgs if s})
    if not _DATASET_PERSONA_CACHE or not _DATASET_FIRST_MSG_CACHE:
        raise RuntimeError("Failed to populate dataset caches (persona/first message).")
    _DATASET_LOADED = True
    logger.info(f"[dataset] cached persona={len(_DATASET_PERSONA_CACHE)}, first_msgs={len(_DATASET_FIRST_MSG_CACHE)}")


def sample_persona_lines_from_dataset(n: int) -> List[str]:
    """Sample n unique persona sentences from dataset cache (no fallback)."""
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
    """Sample one first message from dataset cache (no fallback)."""
    if not _DATASET_LOADED:
        _load_dataset_for_sampling()
    if not _DATASET_FIRST_MSG_CACHE:
        raise RuntimeError("No first messages in dataset cache.")
    return random.choice(_DATASET_FIRST_MSG_CACHE)


# ============== Utilities ==============
def random_ocean() -> Dict[str, float]:
    return {dim: round(random.uniform(0.0, 1.0), 2) for dim in DIMENSIONS}


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
        if v is None or not isfinite(v):
            continue
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{trait} must be in [0,1], got {v}")
        bucket = round(v, 1)
        if bucket not in table[trait]:
            bucket = _nearest_key(table[trait], bucket)
        parts.append(table[trait][bucket])
    return " ".join(parts).strip()


# ============== Agents ==============
@dataclass
class Agent:
    """Assistant agent (can be dynamic or static)."""

    name: str
    dynamic: bool
    persona_lines: List[str] = field(default_factory=list)
    P0: Dict[str, float] = field(default_factory=random_ocean)
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

                # 主动更新更敢
                target_step=0.12,   # ↑ 单步目标位移（从 0.08 提到 0.12）
                lambda_decay=0.80,  # ↑ 回归因子更慢衰，d_t 更大
                alpha_cap=0.55,     # ↑ 放宽每轮最大权重

                # Gate 放宽（你日志里 C/A/N 经常 m=0.75/0.88 但被冷却挡住）
                gate_m_norm=0.20,   # ↓ 触发阈值降低
                gate_min_dims=1,    # ↓ 只需一个维度满足即可
                cooldown_k=1,       # ↓ 相邻轮也可再次更新

                # 被动回归和漂移降到“背景噪声”级
                passive_reg_alpha=0.02,  # ↓ 避免把主动变化吃回去
                passive_reg_use_decay=True,
                global_drift=0.005,      # ↓ 极小化末尾回归

            )
        else:
            self.state_tracker = None

    def get_current_state(self) -> Dict[str, float]:
        if self.dynamic and self.state_tracker is not None:
            return self.state_tracker.get_current_state()
        return dict(self.P0)

    def current_persona_values(self) -> Dict[str, float]:
        return self.get_current_state()

    def system_prompt(self) -> str:
        P_vals = self.current_persona_values()
        base_text = (SYSTEM_PROMPT + " " + " ".join(self.persona_lines)).strip() if self.persona_lines else SYSTEM_PROMPT
        return generate_dynamic_system_prompt(
            base_text=base_text,
            enable_base=True,
            vals=P_vals,
            table=big5_system_prompts_en,
        )

    def update_persona(self, context: List[Dict[str, str]]) -> None:
        if not self.dynamic or self.state_tracker is None:
            return
        try:
            self.state_tracker.step(context)
        except Exception as e:
            self.logger.error(f"Persona update failed for {self.name}: {e}")

    def respond(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        system_prompt = self.system_prompt()
        payload = [{"role": "system", "content": system_prompt}]
        payload += [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        try:
            llm = self.predictor.llm if (self.predictor and getattr(self.predictor, "llm", None)) else llmClient()
            response_text = llm.chat_once(messages=payload, temperature=temperature)
            return (response_text or "").strip()
        except Exception as e:
            user_text = messages[-1].get("content", "") if messages else ""
            self.logger.error(f"LLM call failed for {self.name}: {e}; echo fallback.")
            return "I'm sorry, I'm having trouble generating a reply right now. You said: " + user_text


@dataclass
class PromptedUserAgent:
    """Prompt-driven user agent. Supports 'stable' and 'shifting' topics."""

    name: str
    scenario: str
    total_turns: int
    persona_lines: List[str] = field(default_factory=list)
    P0: Dict[str, float] = field(default_factory=random_ocean)
    predictor: Optional[HeuristicMotivePredictor] = None
    first_message_override: Optional[str] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _turn_index: int = 0

    def _instruction_for_turn(self) -> str:
        sc = self.scenario.lower()
        if sc not in {"stable", "shifting"}:
            raise ValueError(f"Unknown scenario '{self.scenario}'")
        half = self.total_turns // 2 if self.total_turns > 0 else 0
        if sc == "stable" or self._turn_index < half:
            return ("You are participating in a friendly conversation. "
                    "Talk about light-hearted topics (hobbies, foods, travel, music, daily life). "
                    "Be engaging and ask a follow-up question.")
        else:
            return ("Shift the conversation to a more serious tone. "
                    "Express stress/anxiety/fatigue about work or deadlines, seek advice while staying conversational.")

    def _system_prompt(self) -> str:
        instr = self._instruction_for_turn()
        base_text = (instr + " " + " ".join(self.persona_lines)).strip() if self.persona_lines else instr
        try:
            return generate_dynamic_system_prompt(
                base_text=base_text, enable_base=True, vals=self.P0, table=big5_system_prompts_en
            )
        except Exception:
            return base_text

    def respond(self, history: List[Dict[str, str]], temperature: float = 0.7) -> str:
        # Turn 0: force dataset first message if provided
        if self._turn_index == 0 and isinstance(self.first_message_override, str) and self.first_message_override.strip():
            self._turn_index += 1
            return self.first_message_override.strip()

        payload = [{"role": "system", "content": self._system_prompt()}] + history
        try:
            llm = self.predictor.llm if (self.predictor and getattr(self.predictor, "llm", None)) else llmClient()
            text = llm.chat_once(messages=payload, temperature=temperature)
            if text:
                self._turn_index += 1
                return text.strip()
        except Exception as e:
            self.logger.error(f"User LLM call failed for {self.name}: {e}; fallback to templates.")

        # simple fallback prompts (only if LLM not usable)
        return self._fallback_prompt()

    def _fallback_prompt(self) -> str:
        stable = [
            "Hi! How are you today?",
            "What's your favorite hobby?",
            "What do you like to do on weekends?",
            "Do you enjoy travelling?",
            "Tell me about your favourite food.",
        ]
        stress = [
            "I'm feeling anxious about a deadline.",
            "Do you ever feel overwhelmed?",
            "How do you handle stress?",
            "It's been a rough week.",
        ]
        sc = self.scenario.lower()
        half = self.total_turns // 2 if self.total_turns > 0 else 0
        if sc == "stable" or self._turn_index - 1 < half:
            idx = (self._turn_index - 1) % len(stable)
            msg = stable[idx]
        else:
            idx = (self._turn_index - 1 - half) % len(stress)
            msg = stress[idx]
        self._turn_index += 1
        return msg


# ============== Simulation Core ==============
def simulate_dialogue(
    user_agent: PromptedUserAgent,
    assistant_agent: Agent,
    num_turns: int = 15,
    temperature: float = 0.7,
) -> Tuple[List[Dict[str, str]], List[Dict[str, float]]]:
    """
    返回:
      conversation: [{'role':..,'content':..}, ...]
      persona_trace: [{'t':1,'O':..,'C':..,'E':..,'A':..,'N':..}, ...]  # 每个助手回合一次
    """
    conversation: List[Dict[str, str]] = []
    history_for_llm: List[Dict[str, str]] = []
    persona_trace: List[Dict[str, float]] = []

    for turn in range(1, num_turns + 1):
        # user
        user_msg = user_agent.respond(history_for_llm.copy(), temperature=temperature)
        conversation.append({"role": "user", "content": user_msg})
        history_for_llm.append({"role": "user", "content": user_msg})

        # assistant persona update (使用最新上下文进行一次 step)
        assistant_agent.update_persona(history_for_llm.copy())

        # 记录此回合用于生成回复时的当前人格（即更新后）
        cur = assistant_agent.get_current_state()
        persona_trace.append({"t": turn, **{k: float(cur[k]) for k in DIMENSIONS}})

        # assistant reply
        reply = assistant_agent.respond(history_for_llm.copy(), temperature=temperature)
        conversation.append({"role": "assistant", "content": reply})
        history_for_llm.append({"role": "assistant", "content": reply})

    return conversation, persona_trace


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
    """Presample persona lines/P0/first-message for n dialogues (dataset only)."""
    if seed is not None:
        random.seed(seed)
    _load_dataset_for_sampling()  # raises if unavailable
    specs: List[DialogueSpec] = []
    for _ in range(n_dialogues):
        user_lines = sample_persona_lines_from_dataset(persona_lines_per_agent)
        assistant_lines = sample_persona_lines_from_dataset(persona_lines_per_agent)
        fm = sample_first_message_from_dataset()
        specs.append(
            DialogueSpec(
                user_lines=user_lines,
                assistant_lines=assistant_lines,
                user_P0=random_ocean(),
                assistant_P0=random_ocean(),
                first_message=fm,
            )
        )
    return specs


# ============== Single-condition dataset (kept for flexibility) ==============
def simulate_dataset(
    specs: List[DialogueSpec],
    use_dynamic_scenario: bool,     # False → stable; True → shifting
    use_dynamic_persona: bool,      # False → static assistant; True → dynamic assistant
    turns_per_dialogue: int = 15,
    temperature: float = 0.7,
    combo_tag: str = "stable_static",
    index_accumulator: List[Dict[str, str]] | None = None,
) -> List[Dict[str, object]]:
    """Run a dataset for one specific (scenario, persona) condition. 额外执行：保存轨迹 & 绘图。"""
    results: List[Dict[str, object]] = []
    scenario = "shifting" if use_dynamic_scenario else "stable"
    logger.info(f"[simulate_dataset] scenario={scenario}, assistant_dynamic={use_dynamic_persona}, count={len(specs)}")

    plots_dir, traces_dir = _ensure_dirs_for_combo(combo_tag)

    for idx, _ in enumerate(tqdm(range(len(specs)), total=len(specs), desc=f"{scenario}|{'dyn' if use_dynamic_persona else 'static'}")):
        s = specs[idx]
        # 对话级 uuid（四种组合内各自独立）
        dialogue_id = str(uuid.uuid4())

        # user
        user_agent = PromptedUserAgent(
            name=f"User-{idx}",
            scenario=scenario,
            total_turns=turns_per_dialogue,
            persona_lines=s.user_lines,
            P0=s.user_P0,
            first_message_override=s.first_message,
        )
        # assistant
        assistant_agent = Agent(
            name=f"Assistant-{idx}",
            dynamic=use_dynamic_persona,
            persona_lines=s.assistant_lines,
            P0=s.assistant_P0,
            predictor=None if use_dynamic_persona else None,  # 让 Agent 在 dynamic=True 时自行构建 predictor
        )

        # simulate
        conv, trace = simulate_dialogue(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            num_turns=turns_per_dialogue,
            temperature=temperature,
        )

        # 保存轨迹 JSON（单条对话一个文件）
        trace_path = os.path.join(traces_dir, f"{dialogue_id}.json")
        save_persona_trace_json(trace, trace_path)

        # 绘图 PNG（单条对话一张图）
        plot_title = f"{combo_tag} | {dialogue_id}"
        plot_path = os.path.join(plots_dir, f"{dialogue_id}.png")
        plot_persona_trace(trace, plot_title, plot_path)

        # 结果条目（含 uuid 与文件路径，路径相对 outputs/ 便于移植）
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
        }
        results.append(result_item)

        # 记录到全局索引
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
    """
    Generate four comparable datasets sharing the exact same pre-sampled specs:

      1) scenario=stable  , assistant=static
      2) scenario=stable  , assistant=dynamic
      3) scenario=shifting, assistant=static
      4) scenario=shifting, assistant=dynamic

    All four reuse the **same** DialogueSpec list (sampled once with `seed`).
    """
    logger.info(f"[simulate_experiment_4combos] n_per_combo={n_per_combo}, seed={seed}")
    specs = presample_dialogue_specs(n_dialogues=n_per_combo, persona_lines_per_agent=persona_lines_per_agent, seed=seed)

    outputs: Dict[str, List[Dict[str, object]]] = {}
    combos = [
        ("stable_static",   False, False),
        ("stable_dynamic",  False, True),
        ("shifting_static", True,  False),
        ("shifting_dynamic",True,  True),
    ]

    # 索引累积器（输出到 outputs/index.json）
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

    # 将索引写入 outputs/index.json（方便快速浏览）
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
        # 运行 4 组对照，每组 50 条（共 200），同一 seed→相同 specs，公平可比
        result = simulate_experiment_4combos(
            n_per_combo=50,
            persona_lines_per_agent=3,
            turns_per_dialogue=15,
            temperature=0.7,
            seed=1234,  # 固定随机种子 → 四组共享相同预采样
        )
        # 导出总结果（保持原路径习惯，但移动到 outputs/ 目录下更整洁）
        total_out = os.path.join(OUTPUT_ROOT, "simulated_persona_dialogues_4combos.json")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(total_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {total_out}\nIndex at {os.path.join(OUTPUT_ROOT, 'index.json')}")
    except Exception as e:
        print(f"Simulation failed: {e}")
