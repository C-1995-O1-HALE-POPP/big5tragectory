# state_tracker.py
# -*- coding: utf-8 -*-
"""
Section 4.1 Trigger + 4.3 Inference — Persona State Tracker (paper-aligned)
包含三处“更强回归到 P0”的实现：
A) 门控未触发时的被动回归（passive regression）
B) 有方向更新时的“皮筋回拉”目标混合（toward P0）
C) 每轮末尾的极小全局回归（micro drift toward P0）
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable
import random

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

# two-stage predictor（需保证其 score 输出 "motive.{dim}.m_norm" 与 "direction"）
from predictor import HeuristicMotivePredictor, llmClient

# ------------------------------
# Utilities
# ------------------------------

DIMENSIONS = ["O", "C", "E", "A", "N"]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------
# Data classes
# ------------------------------

@dataclass
class TurnRecord:
    t: int
    utterance: str                         # 当前触发计算的用户话语
    Pt: Dict[str, float]                   # 更新后的 P_t
    alpha: Dict[str, float]                # α_i
    decay: Dict[str, float]                # d_t = λ^(t-τ_i)
    m_signed: Dict[str, float]             # m_i 带符号（direction * m_norm ∈ [-1,1]）
    direction: Dict[str, int]              # d_i ∈ {-1,0,1}
    m_norm: Dict[str, float]               # ∈ [0,1]
    P_target: Dict[str, float]             # 更新时的目标状态

# ------------------------------
# Persona State Tracker (paper-aligned)
# ------------------------------

class PersonaStateTracker:
    def __init__(
        self,
        P0: Dict[str, float],
        predictor: Optional[HeuristicMotivePredictor] = None,

        # 论文里的 “±0.1 的方向性微调”（此处做成可调）
        target_step: float = 0.10,

        # 回归因子 λ（0<λ<1），以及 α 的上限（保障数值稳定）
        lambda_decay: float = 0.85,
        alpha_cap: float = 0.75,

        # ---- 4.1 Trigger → Gate 参数 ----
        gate_m_norm: float = 0.25,          # 判定显著性的最小 m_norm
        gate_min_dims: int = 1,             # 至少多少个维度触发（direction≠0 且 m_norm≥阈值）
        cooldown_k: int = 1,                 # 同一维度更新后至少间隔 k 轮才允许再次触发

        # ---- A) 门控未触发时的被动回归 ----
        passive_reg_alpha: float = 0.06,     # Gate=false 时的基础回归步长（0~0.12 建议）
        passive_reg_use_decay: bool = True,  # 是否乘以 d_t 衰减（更平滑）

        # ---- B) “皮筋回拉”：目标点朝 P0 混合 ----
        eta_base: float = 0.15,     # 回拉的基础强度（原来固定 0.15）（默认回拉强度）
        eta_scale: float = 0.50,    # 回拉与偏离距离的线性关系比例（原来固定 0.50）
        eta_cap: float = 0.75,      # 回拉强度的上限，防止完全锁死

        # ---- 护栏参数 ----
        guard_dist: float = 0.35,   # 偏离多大才触发护栏
        guard_alpha_cap: float = 0.25,  # 护栏下压步长的上限

        # ---- C) 每轮末尾的极小全局回归 ----
        global_drift: float = 0.02,          # 0~0.03：太大将掩盖正常更新

        # 其它
        rng: Optional[random.Random] = None,
        task_meta: Optional[Dict] = None,
        eps_update: float = 1e-9
    ):
        assert all(k in P0 for k in DIMENSIONS), "P0 must have O,C,E,A,N"
        self.P0 = {d: clamp(float(P0[d]), 0.0, 1.0) for d in DIMENSIONS}
        self.Pt = self.P0.copy()
        self.t = 0

        self.target_step = float(target_step)
        self.lambda_decay = float(lambda_decay)
        self.alpha_cap = float(alpha_cap)
        self.eps_update = float(eps_update)

        self.predictor = predictor or HeuristicMotivePredictor(llmClient())
        self.task_meta = task_meta
        self.rng = rng or random.Random(1234)

        # Gate 配置
        self.gate_m_norm = float(gate_m_norm)
        self.gate_min_dims = int(gate_min_dims)
        self.cooldown_k = int(cooldown_k)

        # A) 被动回归
        self.passive_reg_alpha = float(passive_reg_alpha)
        self.passive_reg_use_decay = bool(passive_reg_use_decay)

        # B) 皮筋回拉参数
        self.eta_base = float(eta_base)
        self.eta_scale = float(eta_scale)
        self.eta_cap = float(eta_cap)
        self.guard_dist = float(guard_dist)
        self.guard_alpha_cap = float(guard_alpha_cap)

        # C) 全局极小回归
        self.global_drift = float(global_drift)

        # τ_i：上一次该维度发生“有效更新”的对话轮次（初始化为 0）
        self.last_update_turn: Dict[str, int] = {d: 0 for d in DIMENSIONS}
        self.update_history = {d: [] for d in DIMENSIONS}

        # 历史
        self.history: List[TurnRecord] = []
        self.chat_history: List[str] = []

        logger.info(
            f"[Init] P0={self.P0} | target_step={self.target_step} "
            f"| lambda_decay={self.lambda_decay} | alpha_cap={self.alpha_cap} "
            f"| gate(m_norm≥{self.gate_m_norm}, min_dims={self.gate_min_dims}, cooldown={self.cooldown_k}) "
            f"| elastic(eta_base={self.eta_base}, eta_scale={self.eta_scale}, eta_cap={self.eta_cap}) "
            f"| guard(dist>{self.guard_dist}→α_cap={self.guard_alpha_cap})"
            f"| passive_reg_alpha={self.passive_reg_alpha}×decay={self.passive_reg_use_decay} "
            f"| global_drift={self.global_drift}"
        )

    # ---- Gate: 是否进入 Inference+Update ----
    def should_adjust(self, scored: dict) -> bool:
        motive = scored.get("motive", {}) or {}
        hits = []
        logs = []
        for d in DIMENSIONS:
            m = motive.get(d, {}) or {}
            m_norm = float(m.get("m_norm", 0.0))
            direction = int(m.get("direction", 0))
            direction = -1 if direction < 0 else (1 if direction > 0 else 0)
            since = self.t - self.last_update_turn[d]
            cool_ok = since >= self.cooldown_k
            good = (direction != 0) and (m_norm >= self.gate_m_norm) and cool_ok
            if good:
                hits.append(d)
            logs.append(f"{d}(dir={direction:+d}, m={m_norm:.2f}, since={since}, cool_ok={int(cool_ok)})")

        cond_dims = (len(hits) >= self.gate_min_dims)
        logger.info(f"[Gate] decision={cond_dims} (dim_hits={hits}, need≥{self.gate_min_dims}) | {', '.join(logs)}")
        return cond_dims

    # ---- 主入口：处理一批消息（通常按一条 user 消息触发一次） ----
    def step(self, context: List[Dict[str, str]]) -> Dict[str, float]:
        """
        context: list of {"role": "user"/"assistant"/"system", "content": "..."}
        只有当最后一条是 user 时，才进行一轮 persona 更新。
        """
        if not context or not isinstance(context, list):
            raise ValueError("Context must be a list of {'role','content'} dicts.")

        # 累积到完整上下文，以字符串形式喂给 predictor
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            self.chat_history.append(f"[{role.upper()}] {content}")

        if context[-1].get("role") != "user":
            return {}

        # 进入一次“用户回合”
        self.t += 1
        last_text = context[-1].get('content', '')
        logger.info(f"[Turn {self.t}] user: {last_text[:200]}")

        # 先跑一次两阶段评估
        scored = self.predictor.score(
            context_turns=self.chat_history,
            P_t=self.Pt,
            P0=self.P0,
            meta=self.task_meta,
            history=self.update_history  # 如果 predictor 用得到
        )

        # ---- A) Gate=false：执行被动回归至 P0 ----
        if not self.should_adjust(scored):
            alpha_passive, decay_now = {}, {}
            for d in DIMENSIONS:
                delta_turns = max(0, self.t - self.last_update_turn[d])
                d_t = self.lambda_decay ** delta_turns
                decay_now[d] = d_t
                a = self.passive_reg_alpha * (d_t if self.passive_reg_use_decay else 1.0)
                a = clamp(a, 0.0, 0.25)  # 被动回归防抖上限
                pt_now = self.Pt[d]
                pt_target = self.P0[d]
                pt_next = (1.0 - a) * pt_now + a * pt_target
                self.Pt[d] = clamp(pt_next, 0.0, 1.0)
                alpha_passive[d] = a
                logger.debug(f"[PassiveReg] {d}: a={a:.3f} P_t={pt_now:.3f}→{self.Pt[d]:.3f} →P0={pt_target:.3f}")

            # 轮末 C) 全局极小回归
            if self.global_drift > 0.0:
                for d in DIMENSIONS:
                    before = self.Pt[d]
                    self.Pt[d] = (1.0 - self.global_drift) * self.Pt[d] + self.global_drift * self.P0[d]
                    logger.debug(f"[Drift] {d}: {before:.3f}→{self.Pt[d]:.3f} (toward P0={self.P0[d]:.3f})")

            rec = TurnRecord(
                t=self.t,
                utterance=last_text,
                Pt=self.Pt.copy(),
                alpha=alpha_passive,
                decay=decay_now,
                m_signed={d: 0.0 for d in DIMENSIONS},
                direction={d: 0 for d in DIMENSIONS},
                m_norm={d: 0.0 for d in DIMENSIONS},
                P_target={d: self.P0[d] for d in DIMENSIONS},
            )
            self.history.append(rec)
            logger.info(f"[Turn {self.t}] Gate=false → passive regression applied. Pt={json.dumps(self.Pt, ensure_ascii=False)}")
            return self.Pt.copy()

        # ---- 进入 Update 阶段（按论文公式） ----
        alpha: Dict[str, float] = {}
        decay: Dict[str, float] = {}
        m_signed: Dict[str, float] = {}
        m_norm_map: Dict[str, float] = {}
        direction_map: Dict[str, int] = {}
        P_target: Dict[str, float] = {}

        updated_dims = []
        alpha_values = []

        for dim in DIMENSIONS:
            org = self.Pt[dim]
            mot = (scored.get("motive") or {}).get(dim, {})
            m_norm = float(mot.get("m_norm", 0.0))           # ∈ [0,1]
            direction = int(mot.get("direction", 0))         # -1/0/1
            direction = -1 if direction < 0 else (1 if direction > 0 else 0)

            # d_t = λ^(t - τ_i)
            delta_turns = max(0, self.t - self.last_update_turn[dim])
            d_t = self.lambda_decay ** delta_turns

            # α_i = clamp(m_norm * d_t, 0, α_cap)
            a = clamp(m_norm * d_t, 0.0, self.alpha_cap)

            # 计算 pt_target（方向性微调；direction=0 → 回落到 P0）
            pt_now = self.Pt[dim]
            if direction > 0:
                pt_target = clamp(pt_now + self.target_step, 0.0, 1.0)
            elif direction < 0:
                pt_target = clamp(pt_now - self.target_step, 0.0, 1.0)
            else:
                pt_target = self.P0[dim]

            # ---- B) “皮筋回拉”：目标点朝 P0 混合，偏离越远回拉越强 ----
            dist = abs(pt_now - self.P0[dim])
            eta = self.eta_base + self.eta_scale * dist         
            eta = clamp(eta, 0.0, self.eta_cap)
            pt_target = (1 - eta) * pt_target + eta * self.P0[dim]

            # 护栏（可选）：偏离过大时压制步长并强制朝 P0
            if dist > self.guard_dist:
                a = min(a, self.guard_alpha_cap)
                pt_target = self.P0[dim]

            # P_{t+1} = (1-α) P_t + α P_target
            pt_next = (1.0 - a) * pt_now + a * pt_target

            # 记录
            alpha[dim] = a
            decay[dim] = d_t
            m_signed[dim] = direction * m_norm
            direction_map[dim] = direction
            m_norm_map[dim] = m_norm
            P_target[dim] = pt_target

            # 应用
            changed = abs(pt_next - pt_now) > self.eps_update
            self.Pt[dim] = clamp(pt_next, 0.0, 1.0)
            if changed and a > 0.0:
                self.last_update_turn[dim] = self.t
                updated_dims.append(dim)
                alpha_values.append(a)
            self.update_history[dim].append(1 if self.Pt[dim] - org >= 0 else -1)

            logger.debug(
                f"[Upd] {dim}: dir={direction:+d}  m_norm={m_norm:.3f}  d_t={d_t:.3f}  "
                f"α={a:.3f}  P_t={pt_now:.3f}→{self.Pt[dim]:.3f}  P_target={pt_target:.3f}  dist={dist:.3f}"
            )

        # ---- C) 每轮末尾的极小全局回归（不改变 α 记录）----
        if self.global_drift > 0.0:
            for d in DIMENSIONS:
                before = self.Pt[d]
                self.Pt[d] = (1.0 - self.global_drift) * self.Pt[d] + self.global_drift * self.P0[d]
                logger.debug(f"[Drift] {d}: {before:.3f}→{self.Pt[d]:.3f} (toward P0={self.P0[d]:.3f})")

        # 回合汇总日志
        if alpha_values:
            alpha_mean = sum(alpha_values) / len(alpha_values)
            alpha_max = max(alpha_values)
        else:
            alpha_mean = 0.0
            alpha_max = 0.0
        logger.info(
            f"[Turn {self.t} Summary] updated_dims={updated_dims or '∅'} "
            f"| α_mean={alpha_mean:.3f} α_max={alpha_max:.3f} "
            f"| Pt={json.dumps(self.Pt, ensure_ascii=False)}"
        )

        # 保存记录
        rec = TurnRecord(
            t=self.t,
            utterance=last_text,
            Pt=self.Pt.copy(),
            alpha=alpha,
            decay=decay,
            m_signed=m_signed,
            direction=direction_map,
            m_norm=m_norm_map,
            P_target=P_target,
        )
        self.history.append(rec)

        return self.Pt.copy()

    # ---- 运行一个完整对话序列（逐条传入消息） ----
    def run_dialogue(self, utterances: List[Dict[str, str]]) -> List[Dict[str, float]]:
        logger.info("=== Starting dialogue (paper-aligned tracker: Gate + A/B/C regressions) ===")
        traj = [self.Pt.copy()]
        for u in tqdm(utterances):
            res = self.step([u])
            if res:
                traj.append(res)
        logger.info("=== Dialogue finished ===")
        return traj

    # ---- 导出为 DataFrame 便于分析/作图 ----
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for rec in self.history:
            row = {"t": rec.t, "utterance": rec.utterance}
            for d in DIMENSIONS:
                row[f"P_{d}"] = rec.Pt[d]
                row[f"alpha_{d}"] = rec.alpha[d]
                row[f"decay_{d}"] = rec.decay[d]
                row[f"m_signed_{d}"] = rec.m_signed[d]
                row[f"dir_{d}"] = rec.direction[d]
                row[f"m_norm_{d}"] = rec.m_norm[d]
                row[f"target_{d}"] = rec.P_target[d]
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- 便捷访问 ----
    def get_current_state(self) -> Dict[str, float]:
        return self.Pt.copy()

    def get_history(self) -> List[TurnRecord]:
        return list(self.history)

# ------------------------------
# Demo dialogue
# ------------------------------
dialogue = [
    {"role": "system", "content": "Conversation about work, stress regulation, and execution."},

    # --- Cycle 1 ---
    {"role": "user", "content": "Deadlines freak me out… I’m tense and overthinking. 焦虑 紧张 担忧"},
    {"role": "assistant", "content": "I hear the pressure. Let’s label it and plan a small step."},
    {"role": "user", "content": "Let’s slow down—box breathing for 1 minute. 放松 呼吸 冥想"},
    {"role": "assistant", "content": "Inhale 4, hold 4, exhale 4, hold 4. I’ll count with you."},
    {"role": "user", "content": "Feeling calmer. I’ll do a 25-minute focus sprint on the brief."},
    {"role": "assistant", "content": "Timer set to 25m. Outline 3 sections first."},

    # --- Cycle 2 ---
    {"role": "user", "content": "Heart racing again—too many unknowns. 焦虑 分心"},
    {"role": "assistant", "content": "Stress ≠ failure. We’ll contain scope."},
    {"role": "user", "content": "Tea break + soft music, 5 minutes. 放松 小歇"},
    {"role": "assistant", "content": "Lo-fi on, hydration check."},
    {"role": "user", "content": "Back to work—finish the KPI table first."},
    {"role": "assistant", "content": "Single-task: fill metrics A/B/C; no tabs switching."},

    # --- Cycle 3 ---
    {"role": "user", "content": "Mind spiral: what if we miss the review? 焦虑 预期灾难"},
    {"role": "assistant", "content": "Separate facts from fears—draft exists; add examples."},
    {"role": "user", "content": "Stretch shoulders + 10 deep breaths. 放松 伸展"},
    {"role": "assistant", "content": "Hold each stretch 20s; relax your jaw."},
    {"role": "user", "content": "Okay—focus on two concrete examples."},
    {"role": "assistant", "content": "Checklist created; mark as done when examples compile."},

    # --- Cycle 4 ---
    {"role": "user", "content": "Inbox pinged—now I’m jittery again. 焦虑 干扰"},
    {"role": "assistant", "content": "Snooze notifications for 30m; anxiety noted."},
    {"role": "user", "content": "Short walk to reset—3 minutes."},
    {"role": "assistant", "content": "Walk with a slower cadence; exhale longer."},
    {"role": "user", "content": "Ready—code the parser function next."},
    {"role": "assistant", "content": "Define edge cases, then TDD green."},
]

# ------------------------------
# Demo run
# ------------------------------
if __name__ == "__main__":
    # export LOGURU_LEVEL=DEBUG 可看到逐维细节
    P0_demo = {"O": 0.55, "C": 0.65, "E": 0.35, "A": 0.30, "N": 0.40}

    llm = llmClient()

    predictor = HeuristicMotivePredictor(
        llm=llm,
        beta=2.0,
        use_global_factor_weight=True,
        eps=0.15,           # ↓ 缩小死区，direction 更容易非零（你日志里很多维度本就很高）
    )

    tracker = PersonaStateTracker(
        P0=P0_demo,
        predictor=predictor,

        # 主动更新更敢
        target_step=0.3,   # ↑ 单步目标位移（从 0.08 提到 0.12）
        lambda_decay=0.30,  # ↑ 回归因子更慢衰，d_t 更大
        alpha_cap=1.0,     # ↑ 放宽每轮最大权重

        # Gate 放宽（你日志里 C/A/N 经常 m=0.75/0.88 但被冷却挡住）
        gate_m_norm=0.10,   # ↓ 触发阈值降低
        gate_min_dims=1,    # ↓ 只需一个维度满足即可
        cooldown_k=1,       # ↓ 相邻轮也可再次更新

        # 被动回归和漂移降到“背景噪声”级
        passive_reg_alpha=0.002,  # ↓ 避免把主动变化吃回去
        passive_reg_use_decay=True,
        global_drift=0.001,      # ↓ 极小化末尾回归

        # 皮筋回拉参数（新加）
        eta_base=0.01,     # 基础回拉强度
        eta_scale=0.10,    # 回拉与偏离距离的线性关系比例
        eta_cap=0.30,      # 回拉强度的上限

        # 护栏参数（新加）
        guard_dist=0.8,   # 偏离多大才触发护栏
        guard_alpha_cap=0.05,  # 护栏下压步长
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
    plt.title("Persona Trajectories over Dialogue Turns (Gate + A/B/C Regressions)")
    plt.xlabel("Turn (user-only)")
    plt.ylabel("Trait value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("persona_trajectory.png", dpi=160)
    plt.close()

