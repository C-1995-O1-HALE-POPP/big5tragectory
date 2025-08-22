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

    def compute_d_t(self, d: str) -> float:
        age = max(0, self.t - self.last_near_baseline_turn[d])
        return clamp(1.0 - math.exp(-self.lambda_reg * age), 0.0, 1.0)

    def compute_c_t(self, d: str) -> float:
        idle = self.t - self.last_salient_turn[d] >= self.cold_k
        return 1.0 + (self.cold_bonus if idle else 0.0)

    def step(self, utterance: str) -> Dict[str, float]:
        self.t += 1
        scored = self.predictor.score(utterance, self.Pt, self.P0, self.task_meta)

        alpha, d_t_map, c_t_map = {}, {}, {}
        motive_strength, motive_direction, P_target = {}, {}, {}

        logger.info(f"[Turn {self.t}] Utterance: {utterance}")

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
            t=self.t, utterance=utterance, Pt=self.Pt.copy(),
            alpha=alpha, d_t=d_t_map, c_t=c_t_map,
            motive_strength=motive_strength,
            motive_direction=motive_direction,
            P_target=P_target
        )
        self.history.append(rec)
        return self.Pt.copy()

    def run_dialogue(self, utterances: List[str]) -> List[Dict[str, float]]:
        logger.info("=== Starting dialogue simulation ===")
        traj = [self.Pt.copy()]
        for u in utterances:
            traj.append(self.step(u))
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
        "Let's plan a detailed schedule for the week. I want a strict routine. 规划 计划",
        "I feel anxious about deadlines… kind of stressed lately. 焦虑 紧张",
        "Maybe we should relax and take a calm walk, keep it chill. 放松 冷静",
        "Let's explore a novel idea for the project; something creative, imaginative.",
        "I'd like to meet the team, talk through ideas, maybe a quick call. 社交 开麦",
        "Sorry about earlier—thanks for your help; I appreciate your support. 抱歉 感谢 合作",
        "I prefer to work alone today—quiet focus time, no meetings. 独处 低调",
        "Ugh, procrastinated again… I'll do it later. 摸鱼",
        "Boring routine tasks today, let's just get practical stuff done. 务实",
        "We should be kind and supportive; avoid blame and arguing. 体谅 合作",
        "Deadlines are near—organize the tasks and commit to a plan. 准时 自律",
        "Feeling a bit anxious again, can't stop worrying about outcomes. 焦虑",
    ]

    traj = tracker.run_dialogue(dialogue)
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
