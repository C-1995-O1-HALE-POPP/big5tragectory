# -*- coding: utf-8 -*-
"""
simulate_personas.py
====================

- 从 HuggingFace `Cynaptics/persona-chat` 采样 persona 句与首句（无本地 fallback）。
- 用户“场景动态/稳定”与助手“人格动态/静态”两开关 → 四种组合。
- 同一个随机种子先统一预采样 50 组对话规格，在四种组合中复用，保证可比对齐。
- 助手人格的动态调整使用 PersonaStateTracker。
- 用户由 PromptedUserAgent 生成消息（稳定/转场），首句强制为预采样的 dataset 首句。
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable

import os
from math import isfinite
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
from prompt import big5_system_prompts_en, SYSTEM_PROMPT, generate_system_prompt  # noqa: F401 (generate_system_prompt unused)
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker, DIMENSIONS


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
                target_step=0.25,
                lambda_decay=0.80,
                alpha_cap=0.55,
                gate_m_norm=0.20,
                gate_min_dims=1,
                cooldown_k=2,
                passive_reg_alpha=0.02,
                passive_reg_use_decay=True,
                global_drift=0.005,
            )
        else:
            self.state_tracker = None

    def current_persona_values(self) -> Dict[str, float]:
        if self.dynamic and self.state_tracker is not None:
            return self.state_tracker.get_current_state()
        return dict(self.P0)

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
) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    history_for_llm: List[Dict[str, str]] = []
    for _ in range(num_turns):
        # user
        user_msg = user_agent.respond(history_for_llm.copy(), temperature=temperature)
        conversation.append({"role": "user", "content": user_msg})
        history_for_llm.append({"role": "user", "content": user_msg})
        # update assistant persona
        assistant_agent.update_persona(history_for_llm.copy())
        # assistant
        reply = assistant_agent.respond(history_for_llm.copy(), temperature=temperature)
        conversation.append({"role": "assistant", "content": reply})
        history_for_llm.append({"role": "assistant", "content": reply})
    return conversation


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
) -> List[Dict[str, object]]:
    """Run a dataset for one specific (scenario, persona) condition."""
    results: List[Dict[str, object]] = []
    scenario = "shifting" if use_dynamic_scenario else "stable"
    logger.info(f"[simulate_dataset] scenario={scenario}, assistant_dynamic={use_dynamic_persona}, count={len(specs)}")
    for idx, spec in enumerate(tqdm(range(len(specs)), total=len(specs), desc=f"{scenario}|{'dyn' if use_dynamic_persona else 'static'}")):
        s = specs[idx]
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
            predictor=None if use_dynamic_persona else None,  # let Agent build llm/predictor when dynamic
        )
        conv = simulate_dialogue(
            user_agent=user_agent,
            assistant_agent=assistant_agent,
            num_turns=turns_per_dialogue,
            temperature=temperature,
        )
        results.append({
            "scenario": scenario,
            "dynamic": use_dynamic_persona,
            "assistant_P0": s.assistant_P0,
            "user_P0": s.user_P0,
            "conversation": conv,
            "user_persona": s.user_lines,
            "assistant_persona": s.assistant_lines,
            "first_message": s.first_message,
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
    for tag, sc_flag, dyn_flag in combos:
        outputs[tag] = simulate_dataset(
            specs=specs,
            use_dynamic_scenario=sc_flag,
            use_dynamic_persona=dyn_flag,
            turns_per_dialogue=turns_per_dialogue,
            temperature=temperature,
        )
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
        # 导出
        with open("simulated_persona_dialogues_4combos.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Saved to simulated_persona_dialogues_4combos.json")
    except Exception as e:
        print(f"Simulation failed: {e}")
