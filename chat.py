# chat.py
# -*- coding: utf-8 -*-
"""
Gradio Chat UI that integrates:
- Two-stage HeuristicMotivePredictor + PersonaStateTracker (paper-aligned)
- Pre-session hyperparameters (predictor & tracker)
- Dynamic system prompt (bucketed by OCEAN)
- Live trajectory table + line plot (safe matplotlib Agg backend)

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
from typing import List, Tuple, Generator, Optional

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from PIL import Image
import io
import matplotlib.pyplot as plt


# Your modules
from prompt import big5_system_prompts_en, SYSTEM_PROMPT
from predictor import HeuristicMotivePredictor, llmClient
from state_tracker import PersonaStateTracker  # ensure class name matches

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
# Dynamic system prompt assembly
# ==============================
from math import isfinite

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
    """Compose base prompt + five trait bucketed prompts."""
    parts = []
    if enable_base and base_text.strip():
        parts.append(base_text.strip())

    for trait in ["O", "C", "E", "A", "N"]:
        v = vals.get(trait, None)
        if v is None or not isfinite(v):
            continue
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{trait} must be in [0.0, 1.0], got {v}")

        bucket = round(v, 1)
        if bucket not in table[trait]:
            bucket = _nearest_key(table[trait], bucket)
        parts.append(table[trait][bucket])

    return " ".join(parts).strip()


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
    # 一定要 .copy()，否则关闭 BytesIO 后图像句柄会失效
    return Image.open(buf).copy()


# ==============================
# Defaults & examples
# ==============================
DESCRIPTION = """
# Big5Trajectory Chat Assistant

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

    # Session init
    def _init_session():
        sid = str(uuid.uuid4())
        log_json("session_init", session_id=sid)
        return sid, f"**Session ID:** `{sid}` · logs → `chat_history.jsonl`"

    with gr.Column():
        # Base system prompt
        system_box = gr.Textbox(
            label="System Prompt (base)",
            value=SYSTEM_PROMPT,
            placeholder="Optional base persona / role",
            lines=6,
        )

        # Model / sampling
        model_drop = gr.Dropdown(
            label="Model",
            choices=DEFAULT_MODELS,
            value=DEFAULT_MODELS[0],
            allow_custom_value=True,
            interactive=True,
        )
        temperature_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="temperature")
        max_tokens_box = gr.Number(label="max_tokens (≤0/void = unlimited)", value=None, precision=0)

        # Personality sliders (also used as P0 baseline)
        with gr.Accordion("🧠 OCEAN baseline (P0) & Dynamic System Prompt", open=True):
            enable_base_ck = gr.Checkbox(value=True, label="Enable base System Prompt (above)")
            with gr.Row():
                O_slider = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="O - Openness (P0)")
                C_slider = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="C - Conscientiousness (P0)")
                E_slider = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="E - Extraversion (P0)")
                A_slider = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="A - Agreeableness (P0)")
                N_slider = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="N - Neuroticism (P0)")
            dyn_prompt_preview = gr.Textbox(
                label="🧩 Dynamic System Prompt (read-only preview)",
                value="",
                lines=6,
                interactive=False,
            )

        # Predictor hyperparams
        with gr.Accordion("🔧 Predictor (two-stage) hyperparameters", open=False):
            pred_beta = gr.Slider(0.8, 4.0, value=1.3, step=0.1, label="beta (logit sharpness)")
            pred_eps = gr.Slider(0.0, 0.5, value=0.25, step=0.01, label="eps (direction deadzone)")
            pred_use_global = gr.Checkbox(value=True, label="use_global_factor_weight")

        # Tracker hyperparams
        with gr.Accordion("🔧 Tracker (4.1 Trigger + 4.3 Inference) hyperparameters", open=True):
            target_step = gr.Slider(0.01, 0.30, value=0.08, step=0.01, label="target_step (± per update)")
            lambda_decay = gr.Slider(0.2, 0.95, value=0.55, step=0.01, label="lambda_decay (regression λ)")
            alpha_cap = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="alpha_cap (max α per dim)")

            gate_m_norm = gr.Slider(0.0, 0.9, value=0.30, step=0.05, label="gate m_norm threshold")
            gate_min_dims = gr.Slider(1, 5, value=2, step=1, label="gate min hit dims")
            cooldown_k = gr.Slider(0, 5, value=2, step=1, label="cooldown_k (min turns between updates)")

            passive_reg_alpha = gr.Slider(0.0, 0.2, value=0.06, step=0.01, label="passive_reg_alpha (Gate=false)")
            passive_reg_use_decay = gr.Checkbox(value=True, label="passive_reg_use_decay")
            global_drift = gr.Slider(0.0, 0.05, value=0.02, step=0.005, label="global_drift (per turn)")

        init_btn = gr.Button("🚀 Initialize Session (apply hyperparameters)", variant="primary")
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

    # Helpers
    def _update_dyn_prompt(base_text, enable_base, O, C, E, A, N):
        vals = {"O": O, "C": C, "E": E, "A": A, "N": N}
        try:
            return generate_dynamic_system_prompt(
                base_text=base_text,
                enable_base=bool(enable_base),
                vals=vals,
                table=big5_system_prompts_en,
            )
        except Exception as e:
            return f"[Dynamic prompt error] {type(e).__name__}: {e}"

    # live preview on any change
    for comp in [system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider]:
        comp.change(
            _update_dyn_prompt,
            inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
            outputs=[dyn_prompt_preview],
        )

    # Also compute preview on load + initialize session id
    demo.load(_update_dyn_prompt,
              inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
              outputs=[dyn_prompt_preview])
    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])

    # ====== Initialize predictor + tracker with hyperparams ======
    def init_session_and_models(
        session_id: str,
        # predictor
        beta: float, eps: float, use_global: bool,
        # tracker
        target_step_v: float, lambda_decay_v: float, alpha_cap_v: float,
        gate_m: float, gate_dims: int, cooldown: int,
        passive_alpha: float, passive_use_decay: bool, drift: float,
        # P0
        O: float, C: float, E: float, A: float, N: float,
    ):
        log_json("init_hparams", session_id=session_id, predictor={"beta":beta,"eps":eps,"use_global":use_global},
                 tracker={"target_step":target_step_v,"lambda_decay":lambda_decay_v,"alpha_cap":alpha_cap_v,
                          "gate_m":gate_m,"gate_dims":gate_dims,"cooldown":cooldown,
                          "passive_alpha":passive_alpha,"passive_use_decay":passive_use_decay,"drift":drift},
                 P0={"O":O,"C":C,"E":E,"A":A,"N":N})

        predictor = HeuristicMotivePredictor(
            llm=client, beta=float(beta), use_global_factor_weight=bool(use_global), eps=float(eps)
        )
        tracker = PersonaStateTracker(
            P0={"O":O,"C":C,"E":E,"A":A,"N":N},
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
        )

        # Fresh states
        traj = [tracker.get_current_state()]
        df = pd.DataFrame(traj)
        img = render_traj_img(traj)
        return (predictor, tracker, traj, df, gr.update(value=img), [])

    init_btn.click(
        init_session_and_models,
        inputs=[
            session_state,
            pred_beta, pred_eps, pred_use_global,
            target_step, lambda_decay, alpha_cap,
            gate_m_norm, gate_min_dims, cooldown_k,
            passive_reg_alpha, passive_reg_use_decay, global_drift,
            O_slider, C_slider, E_slider, A_slider, N_slider,
        ],
        outputs=[predictor_state, tracker_state, traj_state, traj_df, traj_img, history_state],
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
        system_prompt_base: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: str,
        predictor_obj: HeuristicMotivePredictor,
        tracker_obj: PersonaStateTracker,
        traj: list,
        enable_base: bool,
    ):
        """
        关键点：
        1) 先调用 tracker.step(...) 更新 Pt
        2) 使用新的 Pt 生成动态 system prompt
        3) 调用 LLM（stream_chat）
        4) 更新轨迹表与图
        """
        if not history:
            yield [], history, traj, None, None
            return
        if tracker_obj is None:
            # 未初始化时给出友好提示
            msg = "Please click **Initialize Session** to apply hyperparameters first."
            prior = history[:-1]
            cur = prior + [(history[-1][0], msg)]
            yield history_to_messages(cur), cur, traj, None, None
            return

        # 取最后一条 user 消息、以及之前上下文
        user_msg, _ = history[-1]
        prior = history[:-1]

        # === 1) 先跑 tracker（仅喂 user 一条作为触发轮；也可喂完整历史，取决于你的 predictor 设计）===
        # 这里把 prior + 当前 user 组装为 [ {"role":...}, ... ] 给 tracker 使用
        tracker_context = []
        for u, a in prior:
            if u: tracker_context.append({"role":"user", "content":u})
            if a: tracker_context.append({"role":"assistant", "content":a})
        tracker_context.append({"role":"user", "content":user_msg})

        _ = tracker_obj.step(tracker_context)  # 更新 Pt
        cur_pt = tracker_obj.get_current_state()

        # === 2) 生成动态 system prompt ===
        sys_dyn = generate_dynamic_system_prompt(
            base_text=system_prompt_base, enable_base=bool(enable_base),
            vals=cur_pt, table=big5_system_prompts_en
        )

        # === 3) 调用 LLM（stream）===
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
            # 轨迹在本轮已更新，不用每个增量都重绘图（避免卡顿），但保留接口
            yield history_to_messages(cur), cur, traj, gr.update(), gr.update()

        # === 4) 回合结束：更新轨迹（tracker 已经更新过一次）===
        traj = traj + [cur_pt]
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
            history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state, enable_base_ck
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
            history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state,
            predictor_state, tracker_state, traj_state, enable_base_ck
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
    # 单并发可降低绘图频率下的竞争；如需更高并发可上调
    demo.queue(max_size=64).launch(server_name="0.0.0.0", server_port=27861, share=False)
