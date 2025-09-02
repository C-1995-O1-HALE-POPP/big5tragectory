# chat.py
# -*- coding: utf-8 -*-
"""
Gradio Chat UI integrating:
- HeuristicMotivePredictor + PersonaStateTracker
- Persona selection via prompt.AGENTS (id-based)
- Dynamic system prompt via generate_persona_system_prompt(persona_id, Pt)
- Live trajectory table + line plot (matplotlib Agg)

This version:
- Dynamic persona master toggle (on/off)
- Exposes 5 hyperparams: eta_base / eta_scale / eta_cap / guard_dist / guard_alpha_cap
- Preset dropdown in UI (Shape‑Shifter / Situational Harmonizer / Steady Identity / Custom)
  Presets only prefill controls; you can override before Initialize Session.

Tested with: gradio>=4.0, matplotlib>=3.7, pandas>=1.5
"""

# ---- MUST be set BEFORE importing matplotlib.pyplot ----
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

import io
import json
import uuid
import time
from typing import List, Tuple, Generator, Optional, Dict

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image

# ===== Your project modules =====
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker                       # tracker
from prompt import (                                                # personas & prompt builders
    SYSTEM_PROMPT,                   # base task line
    generate_persona_system_prompt,  # persona prompt builder (id + Pt)
    generate_persona_traits,         # persona -> P0 traits
    AGENTS,                          # list of personas with id/name/vec
)

# ==============================
# Loguru: JSONL logs + rotation
# ==============================
logger.add(
    "chat_history.jsonl",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True,
    backtrace=False,
    diagnose=False,
    level="DEBUG",
    format="{message}",  # pure JSON lines
)

def log_json(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ==============================
# LLM client (shared)
# ==============================
client = llmClient(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
    timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
)

# ==============================
# History <-> messages utility
# ==============================
History = List[Tuple[str, str]]

def history_to_messages(hist: History) -> list:
    """[(user, assistant), ...] -> [{'role','content'}, ...]"""
    msgs = []
    for u, a in hist or []:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs


# ==============================
# Streaming chat with logs
# ==============================
def stream_chat(
    message: str,
    history: History,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    session_id: Optional[str],
) -> Generator[str, None, None]:
    if not session_id:
        session_id = str(uuid.uuid4())

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    log_json(
        "round_start",
        session_id=session_id,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        system_prompt_len=len(system_prompt or ""),
        history_turns=len(history),
    )
    log_json("user_message", session_id=session_id, text=message)

    partial = ""
    try:
        client.change_model(model)
        client.change_temperature(temperature)

        t0 = time.time()
        for i, inc in enumerate(
            client.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
            ),
            start=1,
        ):
            partial += inc
            logger.debug(
                f"[stream_chat] chunk={i}, inc_len={len(inc)}, total_len={len(partial)}, elapsed={time.time()-t0:.2f}s"
            )
            yield partial

        log_json("assistant_message", session_id=session_id, text=partial)
        log_json("round_end", session_id=session_id, tokens=len(partial))
        logger.success(f"[stream_chat] finished, total_len={len(partial)}, elapsed={time.time()-t0:.2f}s")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log_json("error", session_id=session_id, where="stream_chat", detail=err)
        logger.error(f"[stream_chat] error: {err}")
        yield partial + f"\n\n[Error] {err}"


# ==============================
# Plot helper (Agg -> PNG bytes)
# ==============================
DIMENSIONS = ["O", "C", "E", "A", "N"]

def render_traj_img(traj: list[dict]) -> Image.Image | None:
    """Render OCEAN trajectory as a PIL.Image (Agg backend, thread-safe)."""
    if not traj:
        return None
    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.subplots()
    steps = list(range(len(traj)))
    for d in DIMENSIONS:
        ax.plot(steps, [state[d] for state in traj], label=d)
    ax.set_title("Persona Trajectories over Dialogue Turns")
    ax.set_xlabel("Turn (user-only)")
    ax.set_ylabel("Trait value [0..1]")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ==============================
# UI Text
# ==============================
DESCRIPTION = """
# Big5Trajectory Chat Assistant (Persona-ID driven)

- 选择一个 **Persona**（来自 `prompt.AGENTS`），系统将用该 persona 的 Big5 向量作为 **P0**。
- 动态系统提示由 `generate_persona_system_prompt(persona_id, Pt)` 构造，其中 `Pt` 来自 `PersonaStateTracker` 的当前状态。
- 点击 **Initialize Session** 应用当前 persona 与超参。
"""

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-3.5-turbo",
]

EXAMPLES = [
    [
        "Deadlines freak me out… I’m tense and overthinking.",
        SYSTEM_PROMPT,
        DEFAULT_MODELS[0],
        0.7,
        None,
        ""
    ],
    [
        "Tea break + soft music, 5 minutes.",
        SYSTEM_PROMPT,
        DEFAULT_MODELS[0],
        0.7,
        None,
        ""
    ],
]


# ==============================
# Gradio UI
# ==============================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(DESCRIPTION)

    # States
    history_state: gr.State = gr.State([])      # List[Tuple[str, str]]
    session_state: gr.State = gr.State("")
    predictor_state: gr.State = gr.State(None)  # HeuristicMotivePredictor
    tracker_state: gr.State = gr.State(None)    # PersonaStateTracker
    traj_state: gr.State = gr.State([])         # List[Dict[str,float]] trajectory
    current_persona_id: gr.State = gr.State("01")
    p0_state: gr.State = gr.State({"O":0.55,"C":0.65,"E":0.35,"A":0.30,"N":0.40})
    dynamic_enabled_state: gr.State = gr.State(True)

    # Session init helper
    def _init_session():
        sid = str(uuid.uuid4())
        log_json("session_init", session_id=sid)
        return sid, f"**Session ID:** `{sid}` · logs → `chat_history.jsonl`"

    # Persona choices
    _choices = [f'{p["id"]} - {p.get("name","")}' for p in AGENTS]
    _id_by_label = {f'{p["id"]} - {p.get("name","")}' : p["id"] for p in AGENTS}

    with gr.Column():
        with gr.Row():
            persona_drop = gr.Dropdown(
                label="Persona (from prompt.AGENTS)",
                choices=_choices,
                value=_choices[0],
                interactive=True,
                allow_custom_value=False,
                scale=2,
            )
            model_drop = gr.Dropdown(
                label="Model",
                choices=DEFAULT_MODELS,
                value=DEFAULT_MODELS[0],
                allow_custom_value=True,
                interactive=True,
                scale=1,
            )

        # Preset dropdown (UI prefill only)
        preset_drop = gr.Dropdown(
            label="Preset",
            choices=["Shape-Shifter", "Situational Harmonizer", "Steady Identity", "Custom"],
            value="Situational Harmonizer",
            interactive=True,
        )

        # Base task line（仍可编辑）
        system_box = gr.Textbox(
            label="Base Task Line (SYSTEM_PROMPT)",
            value=SYSTEM_PROMPT,
            placeholder="Task line used inside persona system prompt",
            lines=4,
        )

        # Sampling
        temperature_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature")
        max_tokens_box = gr.Number(label="max_tokens (≤0/void = unlimited)", value=None, precision=0)

        # 动态人格开关
        dynamic_checkbox = gr.Checkbox(value=True, label="Enable dynamic persona (use tracker updates)")

        # Persona P0 sliders
        with gr.Accordion("🧠 OCEAN baseline (P0)", open=True):
            with gr.Row():
                O_slider = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="O - Openness (P0)")
                C_slider = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="C - Conscientiousness (P0)")
                E_slider = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="E - Extraversion (P0)")
                A_slider = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="A - Agreeableness (P0)")
                N_slider = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="N - Neuroticism (P0)")
            dyn_prompt_preview = gr.Textbox(
                label="🧩 Dynamic Persona System Prompt (read-only preview)",
                value="",
                lines=10,
                interactive=False,
            )

        # Predictor hyperparams
        with gr.Accordion("🔧 Predictor (two-stage) hyperparameters", open=False):
            pred_beta = gr.Slider(0.8, 4.0, value=2.0, step=0.1, label="beta (logit sharpness)")
            pred_eps = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="eps (direction deadzone)")
            pred_use_global = gr.Checkbox(value=True, label="use_global_factor_weight")

        # Tracker hyperparams
        with gr.Accordion("🔧 Tracker (4.1 Trigger + 4.3 Inference) hyperparameters", open=True):
            target_step = gr.Slider(0.01, 0.50, value=0.30, step=0.01, label="target_step (± per update)")
            lambda_decay = gr.Slider(0.05, 0.95, value=0.30, step=0.01, label="lambda_decay (regression λ)")
            alpha_cap = gr.Slider(0.05, 1.50, value=1.0, step=0.05, label="alpha_cap (max α per dim)")

            gate_m_norm = gr.Slider(0.0, 0.9, value=0.10, step=0.01, label="gate m_norm threshold")
            gate_min_dims = gr.Slider(1, 5, value=1, step=1, label="gate min hit dims")
            cooldown_k = gr.Slider(0, 5, value=1, step=1, label="cooldown_k (min turns between updates)")

            passive_reg_alpha = gr.Slider(0.0, 0.2, value=0.002, step=0.001, label="passive_reg_alpha (Gate=false)")
            passive_reg_use_decay = gr.Checkbox(value=True, label="passive_reg_use_decay")
            global_drift = gr.Slider(0.0, 0.05, value=0.001, step=0.001, label="global_drift (per turn)")

        # Elastic pull-back & Guard rails
        with gr.Accordion("🪢 Elastic pull-back & Guard rails", open=False):
            with gr.Row():
                eta_base = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="eta_base (min pull)")
                eta_scale = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="eta_scale (pull vs. dist)")
                eta_cap = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="eta_cap (pull cap)")
            with gr.Row():
                guard_dist = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="guard_dist (trigger)")
                guard_alpha_cap = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="guard_alpha_cap (α cap)")

        init_btn = gr.Button("🚀 Initialize Session (apply persona & hyperparameters)", variant="primary")
        with gr.Accordion("Trajectory", open=False):
            session_md = gr.Markdown()
            with gr.Row():
                traj_df = gr.Dataframe(label="Trajectory (per user turn)", interactive=False, wrap=True)
                traj_img = gr.Image(label="Trajectory Plot", interactive=False)

    # Chat area
    chatbot = gr.Chatbot(height=520, type="messages", show_copy_button=True)
    with gr.Row():
        msg_box = gr.Textbox(placeholder="Type your message, press Enter or click Send...", lines=2, scale=8)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear chat", scale=1)

    # --- Session id & preview on load ---
    def _persona_label_to_id(label: str) -> str:
        return _id_by_label.get(label, "01")

    def _make_preview(persona_label: str, base_task: str, O: float, C: float, E: float, A: float, N: float):
        pid = _persona_label_to_id(persona_label)
        Pt = {"O":O, "C":C, "E":E, "A":A, "N":N}
        try:
            preview = generate_persona_system_prompt(
                persona_id=pid,
                Pt=Pt,
                include_base_task_line=True,
                include_big5_details=True,
            )
            if SYSTEM_PROMPT and base_task and base_task != SYSTEM_PROMPT:
                preview = preview.replace(SYSTEM_PROMPT, base_task)
            return preview
        except Exception as e:
            return f"[Dynamic prompt error] {type(e).__name__}: {e}"

    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])
    demo.load(
        _make_preview,
        inputs=[persona_drop, system_box, O_slider, C_slider, E_slider, A_slider, N_slider],
        outputs=[dyn_prompt_preview],
    )

    # 当选择 persona 时，自动把滑块同步为该 persona 的 P0
    def _sync_sliders_with_persona(persona_label: str):
        pid = _persona_label_to_id(persona_label)
        trait = generate_persona_traits(pid)  # dict like {"O":0.6,...}
        return (pid,
                gr.update(value=trait["O"]),
                gr.update(value=trait["C"]),
                gr.update(value=trait["E"]),
                gr.update(value=trait["A"]),
                gr.update(value=trait["N"]),
                _make_preview(persona_label, system_box.value, trait["O"], trait["C"], trait["E"], trait["A"], trait["N"]),
                trait,
                )

    persona_drop.change(
        _sync_sliders_with_persona,
        inputs=[persona_drop],
        outputs=[current_persona_id, O_slider, C_slider, E_slider, A_slider, N_slider, dyn_prompt_preview, p0_state],
    )

    # 任何滑块或 base_task 变化时，刷新预览 & 同步 p0_state
    def _update_preview_and_p0(persona_label, base_task, O, C, E, A, N):
        preview = _make_preview(persona_label, base_task, O, C, E, A, N)
        return preview, {"O":O, "C":C, "E":E, "A":A, "N":N}

    for comp in [system_box, O_slider, C_slider, E_slider, A_slider, N_slider]:
        comp.change(
            _update_preview_and_p0,
            inputs=[persona_drop, system_box, O_slider, C_slider, E_slider, A_slider, N_slider],
            outputs=[dyn_prompt_preview, p0_state],
        )

    # 同步动态开关到状态
    dynamic_checkbox.change(lambda v: v, inputs=[dynamic_checkbox], outputs=[dynamic_enabled_state])

    # ---- Preset logic: choose -> update sliders ----
    def _apply_preset(name: str):
        # default (Situational Harmonizer)
        preset = {
            "dynamic_on": True,
            "beta": 2.0, "eps": 0.15, "use_global": True,
            "target_step": 0.30, "lambda_decay": 0.30, "alpha_cap": 1.0,
            "gate_m_norm": 0.10, "gate_min_dims": 1, "cooldown_k": 1,
            "passive_reg_alpha": 0.002, "passive_reg_use_decay": True, "global_drift": 0.001,
            "eta_base": 0.15, "eta_scale": 0.50, "eta_cap": 0.75,
            "guard_dist": 0.35, "guard_alpha_cap": 0.25,
        }
        if name == "Shape-Shifter":
            preset.update({
                "dynamic_on": True,
                "eta_base": 0.01, "eta_scale": 0.10, "eta_cap": 0.30,
                "guard_dist": 0.80, "guard_alpha_cap": 0.05,
            })
        elif name == "Situational Harmonizer":
            pass
        elif name == "Steady Identity":
            preset.update({"dynamic_on": False})
        # Custom 保留当前值：为了简化，这里仍返回默认中庸 preset，你也可以实现读取现有控件值的版本

        return (
            gr.update(value=preset["dynamic_on"]),
            gr.update(value=preset["beta"]),
            gr.update(value=preset["eps"]),
            gr.update(value=preset["use_global"]),
            gr.update(value=preset["target_step"]),
            gr.update(value=preset["lambda_decay"]),
            gr.update(value=preset["alpha_cap"]),
            gr.update(value=preset["gate_m_norm"]),
            gr.update(value=preset["gate_min_dims"]),
            gr.update(value=preset["cooldown_k"]),
            gr.update(value=preset["passive_reg_alpha"]),
            gr.update(value=preset["passive_reg_use_decay"]),
            gr.update(value=preset["global_drift"]),
            gr.update(value=preset["eta_base"]),
            gr.update(value=preset["eta_scale"]),
            gr.update(value=preset["eta_cap"]),
            gr.update(value=preset["guard_dist"]),
            gr.update(value=preset["guard_alpha_cap"]),
        )

    preset_drop.change(
        _apply_preset,
        inputs=[preset_drop],
        outputs=[
            dynamic_checkbox,
            pred_beta, pred_eps, pred_use_global,
            target_step, lambda_decay, alpha_cap,
            gate_m_norm, gate_min_dims, cooldown_k,
            passive_reg_alpha, passive_reg_use_decay, global_drift,
            eta_base, eta_scale, eta_cap, guard_dist, guard_alpha_cap,
        ],
    )

    # ====== Initialize predictor + tracker with hyperparams & persona P0 ======
    def init_session_and_models(
        session_id: str,
        persona_label: str,
        # predictor
        beta: float, eps: float, use_global: bool,
        # tracker
        target_step_v: float, lambda_decay_v: float, alpha_cap_v: float,
        gate_m: float, gate_dims: int, cooldown: int,
        passive_alpha: float, passive_use_decay: bool, drift: float,
        # elastic & guard
        eta_base_v: float, eta_scale_v: float, eta_cap_v: float,
        guard_dist_v: float, guard_alpha_cap_v: float,
        # P0 (may be modified by user; default synced to persona)
        O: float, C: float, E: float, A: float, N: float,
        base_task: str,
        dynamic_on: bool,
    ):
        pid = _persona_label_to_id(persona_label)
        P0 = {"O":O, "C":C, "E":E, "A":A, "N":N}

        log_json("init_hparams",
                 session_id=session_id,
                 persona_id=pid,
                 predictor={"beta":beta,"eps":eps,"use_global":use_global},
                 tracker={
                    "target_step":target_step_v,"lambda_decay":lambda_decay_v,"alpha_cap":alpha_cap_v,
                    "gate_m":gate_m,"gate_dims":gate_dims,"cooldown":cooldown,
                    "passive_alpha":passive_alpha,"passive_use_decay":passive_use_decay,"drift":drift,
                    "eta_base":eta_base_v,"eta_scale":eta_scale_v,"eta_cap":eta_cap_v,
                    "guard_dist":guard_dist_v,"guard_alpha_cap":guard_alpha_cap_v,
                 },
                 P0=P0,
                 base_task=base_task,
                 dynamic_on=dynamic_on)

        predictor = HeuristicMotivePredictor(
            llm=client, beta=float(beta), use_global_factor_weight=bool(use_global), eps=float(eps)
        )
        tracker = PersonaStateTracker(
            P0=P0,
            predictor=predictor,
            target_step=float(target_step_v),
            lambda_decay=float(lambda_decay_v),
            alpha_cap=float(alpha_cap_v),
            gate_m_norm=float(gate_m),
            gate_min_dims=int(gate_dims),
            cooldown_k=int(cooldown),
            passive_reg_alpha=float(passive_alpha),
            passive_reg_use_decay=bool(passive_use_decay),
            global_drift=float(drift),
            # elastic & guard
            eta_base=float(eta_base_v),
            eta_scale=float(eta_scale_v),
            eta_cap=float(eta_cap_v),
            guard_dist=float(guard_dist_v),
            guard_alpha_cap=float(guard_alpha_cap_v),
        )

        # Fresh states
        traj = [P0]  # 初始化展示 P0（无论是否动态，第一点都是 P0）
        df = pd.DataFrame(traj)
        img = render_traj_img(traj)

        # 初始化时也刷新预览
        preview = _make_preview(persona_label, base_task, O, C, E, A, N)
        return (
            pid, predictor, tracker, traj, df, gr.update(value=img),
            [], gr.update(value=preview),
            P0,
            dynamic_on,
        )

    init_btn.click(
        init_session_and_models,
        inputs=[
            session_state, persona_drop,
            pred_beta, pred_eps, pred_use_global,
            target_step, lambda_decay, alpha_cap,
            gate_m_norm, gate_min_dims, cooldown_k,
            passive_reg_alpha, passive_reg_use_decay, global_drift,
            # elastic & guard
            eta_base, eta_scale, eta_cap, guard_dist, guard_alpha_cap,
            O_slider, C_slider, E_slider, A_slider, N_slider,
            system_box,
            dynamic_checkbox,
        ],
        outputs=[
            current_persona_id, predictor_state, tracker_state, traj_state, traj_df, traj_img,
            history_state, dyn_prompt_preview,
            p0_state,
            dynamic_enabled_state,
        ],
    )

    # ====== Submit flow ======
    def user_submit(user_msg: str, history: History):
        if user_msg is None:
            user_msg = ""
        history = (history or []) + [(user_msg, "")]
        messages = history_to_messages(history)
        return gr.update(value=""), history, messages

    def bot_respond(
        history: History,
        persona_label: str,
        base_task: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: str,
        predictor_obj: HeuristicMotivePredictor,
        tracker_obj: PersonaStateTracker,
        traj: list,
        p0: Dict[str, float],
        dynamic_on: bool,
    ):
        """
        回合流程：
        - 若 dynamic_on=True：
            1) tracker.step(...) 用本轮 user 触发更新 Pt
            2) 用新的 Pt 构造系统提示
        - 若 dynamic_on=False：
            1) 跳过 tracker.step
            2) 直接使用 P0 作为 Pt 构造系统提示
        3) LLM 流式回复
        4) 轨迹记录：dynamic_on=False 时轨迹将保持在 P0（用于可视一致性）
        """
        if not history:
            yield [], history, traj, None, None
            return
        if tracker_obj is None:
            msg = "Please click **Initialize Session** first."
            prior = history[:-1]
            cur = prior + [(history[-1][0], msg)]
            yield history_to_messages(cur), cur, traj, None, None
            return

        user_msg, _ = history[-1]
        prior = history[:-1]

        # 让 tracker 读上下文，但在 dynamic_off 时不更新内部状态
        if dynamic_on:
            tracker_context = []
            for u, a in prior:
                if u: tracker_context.append({"role":"user", "content":u})
                if a: tracker_context.append({"role":"assistant", "content":a})
            tracker_context.append({"role":"user", "content":user_msg})
            _ = tracker_obj.step(tracker_context)  # 更新 Pt
            cur_pt = tracker_obj.get_current_state()
        else:
            # 固定为 P0；尽量从 tracker_obj.P0 获取，否则回退到 p0_state
            cur_pt = getattr(tracker_obj, "P0", p0)

        # persona id
        pid = _persona_label_to_id(persona_label)

        # 生成 persona 系统提示
        sys_dyn = generate_persona_system_prompt(
            persona_id=pid,
            Pt=cur_pt,
            include_base_task_line=True,
            include_big5_details=True,
        )
        if SYSTEM_PROMPT and base_task and base_task != SYSTEM_PROMPT:
            sys_dyn = sys_dyn.replace(SYSTEM_PROMPT, base_task)

        # 流式回复
        partial = ""
        for chunk in stream_chat(
            message=user_msg,
            history=prior,
            system_prompt=sys_dyn,
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens else None,
            session_id=session_id,
        ):
            partial = chunk
            cur = prior + [(user_msg, partial)]
            yield history_to_messages(cur), cur, traj, gr.update(), gr.update()

        # 回合结束：记录轨迹
        if dynamic_on:
            traj = traj + [cur_pt]
        else:
            traj = traj + [p0]

        df = pd.DataFrame(traj)
        final_hist = prior + [(user_msg, partial)]
        img = render_traj_img(traj)
        yield history_to_messages(final_hist), final_hist, traj, df, gr.update(value=img)

    # Bind events
    msg_box.submit(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[
            history_state, persona_drop, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state,
            p0_state,
            dynamic_enabled_state,
        ],
        outputs=[chatbot, history_state, traj_state, traj_df, traj_img],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[
            history_state, persona_drop, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state,
            p0_state,
            dynamic_enabled_state,
        ],
        outputs=[chatbot, history_state, traj_state, traj_df, traj_img],
    )

    # Clear
    def clear_chat():
        return [], [], []

    clear_btn.click(
        clear_chat, inputs=None, outputs=[history_state, chatbot, msg_box],
    )

    with gr.Accordion("⚙️ Environment", open=False):
        gr.Markdown(
            f"""
- `OPENAI_BASE_URL`: `{os.getenv("OPENAI_BASE_URL", "") or "(not set)"}`
- `OPENAI_API_KEY`: `{"set" if os.getenv("OPENAI_API_KEY") else "not set"}`
- Logs: `chat_history.jsonl` (JSON Lines; rotation, 7-day retention, compression)
"""
        )

if __name__ == "__main__":
    demo.queue(max_size=64).launch(server_name="0.0.0.0", server_port=27861, share=False)
