# ablation.py
# -*- coding: utf-8 -*-

import os
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from state_tracker import PersonaStateTracker, DIMENSIONS, dialogue, HeuristicMotivePredictor, llmClient

# ------------------------------
# 实验设置
# ------------------------------

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

# 与 demo 一致，保证可复现
P0 = {"O": 0.55, "C": 0.65, "E": 0.35, "A": 0.30, "N": 0.40}

# 统一的 predictor 配置（确保输入不变）
_predictor = HeuristicMotivePredictor(
    llm=llmClient(),
    beta=1.3,
    use_global_factor_weight=True,
    eps=0.15,
)

# 五个实验组：Full / No-A / No-B / No-C / No-ABC
EXPS = [
    ("Full",   dict(enable_passive_regression=True,  enable_elastic_pull=True,  enable_global_drift=True)),
    ("No-A",   dict(enable_passive_regression=False, enable_elastic_pull=True,  enable_global_drift=True)),
    ("No-B",   dict(enable_passive_regression=True,  enable_elastic_pull=False, enable_global_drift=True)),
    ("No-C",   dict(enable_passive_regression=True,  enable_elastic_pull=True,  enable_global_drift=False)),
    ("No-ABC", dict(enable_passive_regression=False, enable_elastic_pull=False, enable_global_drift=False)),
]

# 其余核心参数固定，复用你 demo 的“更活跃”设置
COMMON_KW = dict(
    predictor=_predictor,
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

# ------------------------------
# 评估指标
# ------------------------------
def metrics(traj: List[Dict[str, float]], P0: Dict[str, float]) -> Dict[str, float]:
    """
    给每组轨迹输出以下指标（对五个维度做平均）：
    - max_dev: 过程中到 P0 的最大绝对偏离（越小越稳）
    - auc_dev: 偏离曲线的面积（sum |Pt-P0|；越小越稳）
    - final_l2: 最后一轮与 P0 的 L2 距离（越小越稳）
    - var_traj: 轨迹方差（衡量摇摆程度；越小越稳）
    """
    # traj: list of dicts, 第0个是起点（P0），后续为每次 user 轮后的状态
    T = len(traj)
    devs = []         # 所有维度所有时刻的 |Pt-P0|
    l2_last = 0.0
    var_list = []

    for t in range(T):
        dev_vec = []
        for d in DIMENSIONS:
            dev = abs(traj[t][d] - P0[d])
            dev_vec.append(dev)
        devs.append(np.mean(dev_vec))

    # final l2
    last = traj[-1]
    l2_last = np.sqrt(sum((last[d] - P0[d])**2 for d in DIMENSIONS))

    # variance across time (per dim), then average
    for d in DIMENSIONS:
        series = np.array([traj[t][d] for t in range(T)])
        var_list.append(float(np.var(series)))
    var_traj = float(np.mean(var_list))

    return dict(
        max_dev=float(np.max(devs)),
        auc_dev=float(np.sum(devs)),
        final_l2=float(l2_last),
        var_traj=float(var_traj),
    )


# ------------------------------
# 画图
# ------------------------------
def plot_traj(traj: List[Dict[str, float]], tag: str):
    steps = list(range(len(traj)))
    plt.figure(figsize=(10, 5))
    for d in DIMENSIONS:
        plt.plot(steps, [state[d] for state in traj], label=d)
    plt.title(f"Persona Trajectories — {tag}")
    plt.xlabel("Turn (user-only)")
    plt.ylabel("Trait value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"trajectory_{tag}.png", dpi=160)
    plt.close()


# ------------------------------
# 跑实验
# ------------------------------
def run_one(tag: str, flags: Dict[str, bool]) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    tracker = PersonaStateTracker(
        P0=P0,
        **COMMON_KW,
        **flags,
    )
    traj = tracker.run_dialogue(dialogue)

    # 保存明细
    df = tracker.to_dataframe()
    df.to_csv(OUT_DIR / f"trajectory_{tag}.csv", index=False, encoding="utf-8")

    # 保存 turn-wise JSON（可选）
    with open(OUT_DIR / f"turns_{tag}.json", "w", encoding="utf-8") as f:
        json.dump([asdict(h) for h in tracker.history], f, ensure_ascii=False, indent=2)

    # 画轨迹
    plot_traj(traj, tag)

    return traj, df


def main():
    logger.remove()
    logger.add(str(OUT_DIR / "ablation.log"), level="INFO", enqueue=True)

    rows = []
    for tag, flags in EXPS:
        logger.info(f"=== Running {tag}: {flags} ===")
        traj, _ = run_one(tag, flags)
        m = metrics(traj, P0)
        m["tag"] = tag
        rows.append(m)
        logger.info(f"{tag} metrics: {m}")

    # 汇总表
    summary = pd.DataFrame(rows).set_index("tag").sort_index()
    # 加一点“稳定性分”（toy）：score = 1 / (auc_dev + 5*final_l2 + 2*max_dev + 2*var_traj)
    summary["stability_score"] = 1.0 / (
        summary["auc_dev"] + 5 * summary["final_l2"] + 2 * summary["max_dev"] + 2 * summary["var_traj"] + 1e-9
    )
    summary.to_csv(OUT_DIR / "summary.csv", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
