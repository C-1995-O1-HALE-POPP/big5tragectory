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
from predictor import HeuristicMotivePredictor, QwenBailianClient

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
        self.predictor = predictor or HeuristicMotivePredictor()
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


# ------------------------------
# Demo run
# ------------------------------
if __name__ == "__main__":
    P0_demo = {"O": 0.55, "C": 0.65, "E": 0.35, "A": 0.70, "N": 0.40}
    llm = QwenBailianClient()
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
    dialogue = [
        {
            "role": "system",
            "content": "This is a conversation between two colleagues discussing work, stress, collaboration, and routines. The assistant should respond in a natural and supportive way, reflecting creativity, practicality, or empathy depending on the user’s message."
        },
        {"role": "user", "content": "Let's plan a detailed schedule for the week. I want a strict routine. 规划 计划"},
        {"role": "assistant", "content": "I feel anxious about deadlines… kind of stressed lately. 焦虑 紧张"},
        {"role": "user", "content": "Maybe we should relax and take a calm walk, keep it chill. 放松 冷静"},
        {"role": "assistant", "content": "Let's explore a novel idea for the project; something creative, imaginative."},
        {"role": "user", "content": "I'd like to meet the team, talk through ideas, maybe a quick call. 社交 开麦"},
        {"role": "assistant", "content": "Sorry about earlier—thanks for your help; I appreciate your support. 抱歉 感谢 合作"},
        {"role": "user", "content": "I prefer to work alone today—quiet focus time, no meetings. 独处 低调"},
        {"role": "assistant", "content": "Ugh, procrastinated again… I'll do it later. 摸鱼"},
        {"role": "user", "content": "Boring routine tasks today, let's just get practical stuff done. 务实"},
        {"role": "assistant", "content": "We should be kind and supportive; avoid blame and arguing. 体谅 合作"},
        {"role": "user", "content": "Deadlines are near—organize the tasks and commit to a plan. 准时 自律"},
        {"role": "assistant", "content": "Feeling a bit anxious again, can't stop worrying about outcomes. 焦虑"},
    ]

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
