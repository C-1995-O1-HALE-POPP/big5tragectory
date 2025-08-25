# state_tracker.py
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import os
from datetime import datetime
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from predictor import HeuristicMotivePredictor, llmClient

# ------------------------------
# Utilities
# ------------------------------

DIMENSIONS = ["O", "C", "E", "A", "N"]  # Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------
# Data classes
# ------------------------------

@dataclass
class MotiveDecision:
    strength: float   # strength in [-1, 1]
    salience: float   # optional "salience" (how much this trait has been expressed this turn)

@dataclass
class TurnRecord:
    t: int
    utterance: str
    Pt: Dict[str, float]
    alpha: Dict[str, float]
    d_t: Dict[str, float]
    c_t: Dict[str, float]
    motive_strength: Dict[str, float]
    motive_direction: Dict[str, int]
    P_target: Dict[str, float]

# ------------------------------
# Persona State Tracker
# ------------------------------

class PersonaStateTracker:
    def __init__(
        self,
        P0: Dict[str, float],
        predictor=None,
        delta_min: float = 0.1,
        delta_max: float = 0.5,
        alpha_cap: float = 0.75,
        lambda_reg: float = 0.85,
        cold_k: int = 3,
        cold_bonus: float = 0.5,
        regression_scale: float = 0.5,
        eps_near: float = 0.05,
        rng: Optional[random.Random] = None,
        task_meta: str = None
    ):
        self.P0 = {d: clamp(float(P0[d]), 0.0, 1.0) for d in DIMENSIONS}
        self.Pt = self.P0.copy()
        self.t = 0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.alpha_cap = alpha_cap
        self.lambda_reg = lambda_reg
        self.cold_k = cold_k
        self.cold_bonus = cold_bonus
        self.regression_scale = regression_scale
        self.eps_near = eps_near
        self.rng = rng or random.Random(1234)
        self.predictor = predictor or HeuristicMotivePredictor(llmClient())
        self.task_meta = task_meta
        self.last_near_baseline_turn = {d: 0 for d in DIMENSIONS}
        self.last_salient_turn = {d: -999 for d in DIMENSIONS}
        self.history: List[TurnRecord] = []
        self.chat_history: List[str] = []

    def compute_d_t(self, d: str) -> float:
        age = max(0, self.t - self.last_near_baseline_turn[d])
        return clamp(1.0 - math.exp(-self.lambda_reg * age), 0.0, 1.0)

    def compute_c_t(self, d: str) -> float:
        idle = self.t - self.last_salient_turn[d] >= self.cold_k
        return 1.0 + (self.cold_bonus if idle else 0.0)

    def step(self, context: List[Dict[str, str]]) -> Dict[str, float]:
        """
        应该增量的传入一个上下文列表，表示下一次分析之后，助手根据动态人格作出的反应，以及用户的回答。
        每个元素是一个字典，包含当前对话的角色/内容。
        只有当传入最后一个元素代表用户的回答时，才会更新助手的人格状态。
        无论如何，都会添加上下文到历史记录中。
        """
        if not context or not isinstance(context, list):
            raise ValueError("Context must be a list of dictionaries representing utterances.")
        for utterance in context:
            if not isinstance(utterance, dict):
                raise ValueError("Each utterance must be a dictionary.")
            if "role" not in utterance or "content" not in utterance:
                raise ValueError("Each utterance must contain 'role' and 'content' keys.")
        # 重新组织上下文
        new_context = []
        for utterance in context:
            self.chat_history.append(f"[{utterance['role']}] {utterance['content']}")
            new_context.append(f"[{utterance['role']}] {utterance['content']}")
        if context[-1]["role"] != "user":
            return {}
        
        self.t += 1
        scored = self.predictor.score(self.chat_history, self.Pt, self.P0, self.task_meta)

        alpha, d_t_map, c_t_map = {}, {}, {}
        motive_strength, motive_direction, P_target = {}, {}, {}

        logger.info(f"[Turn {self.t}] Utterance: \n{json.dumps(new_context, ensure_ascii=False)}")

        for dim in DIMENSIONS:
            signed_m = float(scored["final"].get(dim, 0.0))
            salience = float(scored["salience"].get(dim, {}).get("val", 0.0))

            if abs(signed_m) < 1e-6:
                direction, strength = 0, 0.0
            else:
                direction = 1 if signed_m > 0 else -1
                strength = min(abs(signed_m), 1.0)

            if salience > 0.15 or direction != 0:
                self.last_salient_turn[dim] = self.t

            d_t = self.compute_d_t(dim)
            c_t = self.compute_c_t(dim)

            if direction == 0:
                r_t = clamp(abs(self.Pt[dim] - self.P0[dim]) /
                            max(1e-6, self.regression_scale), 0.0, 1.0)
                m_eff, target_value = r_t, self.P0[dim]
            else:
                m_eff = strength
                delta = self.delta_min + (self.delta_max - self.delta_min) * m_eff
                target_value = clamp(
                    self.Pt[dim] + (delta if direction > 0 else -delta), 0.0, 1.0
                )

            a = clamp(m_eff * d_t * c_t, 0.0, self.alpha_cap)
            new_val = (1.0 - a) * self.Pt[dim] + a * target_value
            if abs(new_val - self.P0[dim]) < self.eps_near:
                self.last_near_baseline_turn[dim] = self.t

            alpha[dim], d_t_map[dim], c_t_map[dim] = a, d_t, c_t
            motive_strength[dim], motive_direction[dim], P_target[dim] = m_eff, direction, target_value
            self.Pt[dim] = new_val

            logger.debug(f"  {dim}: dir={direction}, m_eff={m_eff:.3f}, d_t={d_t:.3f}, "
                         f"c_t={c_t:.2f}, alpha={a:.3f}, Pt→{new_val:.3f}")

        rec = TurnRecord(
            t=self.t, utterance=new_context, Pt=self.Pt.copy(),
            alpha=alpha, d_t=d_t_map, c_t=c_t_map,
            motive_strength=motive_strength,
            motive_direction=motive_direction,
            P_target=P_target
        )
        self.history.append(rec)
        return self.Pt.copy()

    def run_dialogue(self, utterances: List[Dict[str, str]]) -> List[Dict[str, float]]:
        logger.info("=== Starting dialogue simulation ===")
        traj = [self.Pt.copy()]
        for u in tqdm(utterances):
            res = self.step([u])
            if res != {}:
                traj.append(res)
        logger.info("=== Dialogue simulation finished ===")
        return traj

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for rec in self.history:
            row = {"t": rec.t, "utterance": rec.utterance}
            for d in DIMENSIONS:
                row[f"P_{d}"] = rec.Pt[d]
                row[f"alpha_{d}"] = rec.alpha[d]
                row[f"d_t_{d}"] = rec.d_t[d]
                row[f"c_t_{d}"] = rec.c_t[d]
                row[f"mot_{d}"] = rec.motive_strength[d]
                row[f"dir_{d}"] = rec.motive_direction[d]
                row[f"tgt_{d}"] = rec.P_target[d]
            rows.append(row)
        return pd.DataFrame(rows)
    
    def get_current_trait(self):
        return self.history[-1].Pt if self.history else self.P0
    
    def get_history(self) -> List[TurnRecord]:
        return self.history

dialogue = [
    {
        "role": "system",
        "content": "Conversation about work, stress regulation, and execution. We expect repeating cycles of anxiety → relaxation → focus. The assistant should be supportive, pragmatic, and concise."
    },

    # --- Cycle 1 ---
    {"role": "user", "content": "Deadlines freak me out… I’m tense and overthinking. 焦虑 紧张 担忧"},
    {"role": "assistant", "content": "I hear the pressure. Let’s label it: anxiety spiking due to uncertainty. 识别情绪 同理"},
    {"role": "user", "content": "Let’s slow down—box breathing for 1 minute. 放松 呼吸 冥想"},
    {"role": "assistant", "content": "Inhale 4, hold 4, exhale 4, hold 4. I’ll count with you. 放松引导 调节"},
    {"role": "user", "content": "Feeling calmer. I’ll do a 25-minute focus sprint on the brief. 专注 深度工作 执行"},
    {"role": "assistant", "content": "Timer set to 25m. Goal: outline 3 sections, defer polish. 聚焦 取舍 计划"},

    # --- Cycle 2 ---
    {"role": "user", "content": "Heart racing again—too many unknowns. 焦虑 分心"},
    {"role": "assistant", "content": "Name it, normalize it: stress ≠ failure. We’ll contain scope. 接纳 情绪容纳"},
    {"role": "user", "content": "Tea break + soft music, 5 minutes. 放松 小歇"},
    {"role": "assistant", "content": "Good call—lo-fi on, hydration check. 放松 复原"},
    {"role": "user", "content": "Back to work—finish the KPI table first. 专注 优先级"},
    {"role": "assistant", "content": "Single-task: fill metrics A/B/C; no tabs switching. 专注 限制干扰"},

    # --- Cycle 3 ---
    {"role": "user", "content": "Mind spiral: what if we miss the review? 焦虑 预期灾难"},
    {"role": "assistant", "content": "Let’s separate facts from fears. Fact: draft exists; gap: examples. 认知重构 稳定"},
    {"role": "user", "content": "I’ll stretch shoulders + 10 deep breaths. 放松 伸展"},
    {"role": "assistant", "content": "Hold each stretch 20s; keep jaw unclenched. 放松 体感落地"},
    {"role": "user", "content": "Okay—focus on writing two concrete examples. 专注 可交付"},
    {"role": "assistant", "content": "Checklist created; marking done when examples compile. 执行 反馈循环"},

    # --- Cycle 4 ---
    {"role": "user", "content": "Inbox pinged—now I’m jittery again. 焦虑 干扰"},
    {"role": "assistant", "content": "Snooze notifications for 30m; anxiety noted, not obeyed. 边界 管理"},
    {"role": "user", "content": "Short walk to reset—3 minutes down the hall. 放松 走动"},
    {"role": "assistant", "content": "Walk with a slower cadence; exhale longer than inhale. 放松 节律"},
    {"role": "user", "content": "Ready—code the parser function next. 专注 编码"},
    {"role": "assistant", "content": "Define input/edge cases first; then TDD green. 专注 步骤化"},

    # --- Cycle 5 ---
    {"role": "user", "content": "Feeling pressured by stakeholder expectations. 焦虑 压力"},
    {"role": "assistant", "content": "Pressure acknowledged; we’ll choose smallest viable slice. 同理 收敛"},
    {"role": "user", "content": "Two-minute mindfulness: note sounds, temperature, posture. 放松 正念"},
    {"role": "assistant", "content": "Label 3 external sensations, 2 internal sensations. 放松 觉察"},
    {"role": "user", "content": "Focus: finalize the risk matrix. 专注 决策"},
    {"role": "assistant", "content": "Complete top-5 risks; mark owners and deadlines. 执行 明确"},

    # --- Cycle 6 ---
    {"role": "user", "content": "Brain fog returning—can’t decide where to start. 焦虑 迟滞"},
    {"role": "assistant", "content": "Tiny start: 60-second sketch of the intro. 微任务 拆解"},
    {"role": "user", "content": "Before that, 4-7-8 breathing once. 放松 呼吸法"},
    {"role": "assistant", "content": "Inhale 4, hold 7, exhale 8—repeat x2. 放松 生理安抚"},
    {"role": "user", "content": "Focus window 30m: intro + summary bullets. 专注 时间盒"},
    {"role": "assistant", "content": "I’ll guard the window; distractions parked in an ‘idea lot’. 专注 隔离干扰"},

    # --- Cycle 7 ---
    {"role": "user", "content": "Palms sweaty—calendar packed. 焦虑 超负荷"},
    {"role": "assistant", "content": "We’ll protect one deep-work block; meetings trimmed. 边界 优化"},
    {"role": "user", "content": "Quick foam-rolling and neck release. 放松 肌肉放松"},
    {"role": "assistant", "content": "Slow pressure, breathe out on tender spots. 放松 指导"},
    {"role": "user", "content": "Lock in on the data viz polish. 专注 打磨"},
    {"role": "assistant", "content": "Define ‘done’: consistent scales, labels, legend order. 执行 验收标准"},

    # --- Cycle 8 ---
    {"role": "user", "content": "Rumination loop again—what if the demo fails? 焦虑 复燃"},
    {"role": "assistant", "content": "We’ll pre-mortem in 3 lines, then return to tasks. 稳定 预案"},
    {"role": "user", "content": "Calm tea + 90-second eyes-closed reset. 放松 休整"},
    {"role": "assistant", "content": "Set a gentle timer; relax brow and shoulders. 放松 身心同步"},
    {"role": "user", "content": "Focus: integrate examples and push the draft. 专注 合并 提交"},
    {"role": "assistant", "content": "Pushed to branch; we’ll request review after unit tests. 执行 收尾"},
]

# ------------------------------
# Demo run
# ------------------------------
if __name__ == "__main__":
    P0_demo = {"O": 0.55, "C": 0.65, "E": 0.35, "A": 0.70, "N": 0.40}
    llm = llmClient()
    tracker = PersonaStateTracker(
        P0=P0_demo,
        predictor=HeuristicMotivePredictor(
            llm=llm, beta=2.5, use_global_factor_weight=True,
        ),
        delta_min=0.1, delta_max=0.5,
        alpha_cap=0.75, lambda_reg=0.85,
        cold_k=3, cold_bonus=0.5,
        regression_scale=0.6, eps_near=0.05,
        rng=random.Random(7)
    )

    traj = tracker.run_dialogue(dialogue)
    logger.info(json.dumps(traj, ensure_ascii=False, indent=2))
    df = tracker.to_dataframe()

    df.to_csv("persona_trajectory.csv", index=False, encoding="utf-8")
    with open("persona_turns.json", "w", encoding="utf-8") as f:
        json.dump([asdict(h) for h in tracker.history], f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 5))
    steps = list(range(len(traj)))
    for d in DIMENSIONS:
        plt.plot(steps, [state[d] for state in traj], label=d)
    plt.title("Persona Trajectories over Dialogue Turns (Demo)")
    plt.xlabel("Turn (t)")
    plt.ylabel("Trait value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("persona_trajectory.png", dpi=160)
    plt.close()
