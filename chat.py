# app.py
import os
import json
import uuid
from typing import List, Tuple, Generator, Optional
import time

import gradio as gr
from loguru import logger

from prompt import big5_system_prompts_en, SYSTEM_PROMPT
from predictor import llmClient

# ==============================
# Loguru: JSONL logs + rotation
# ==============================
logger.remove()
logger.add(
    "chat_history.jsonl",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True,
    backtrace=False,
    diagnose=False,
    level="INFO",
    format="{message}",  # write pure JSON lines
)

def log_json(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ==============================
# LLM client
# ==============================
client = llmClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=1,
    timeout=60,
)


# ==============================
# History <-> messages utility
# ==============================
History = List[Tuple[str, str]]

def history_to_messages(hist: History) -> list:
    """[(user, assistant), ...] -> [{'role','content'}, ...]"""
    msgs = []
    for u, a in hist or []:
        if u is not None and u != "":
            msgs.append({"role": "user", "content": u})
        if a is not None and a != "":
            msgs.append({"role": "assistant", "content": a})
    return msgs


from math import isfinite

def _nearest_key(d: dict[float, str], v: float) -> float:
    """Pick the closest bucket key (0.0~1.0) from dict d."""
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
# Core chat (streaming) + logs
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

    # Build messages
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
        system_prompt=system_prompt,
        history_turns=len(history),
    )
    log_json("user_message", session_id=session_id, text=message)

    partial = ""
    try:
        # Sync UI selections to llmClient
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
            # Progress logs
            logger.info(
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
# Config & examples
# ==============================
DESCRIPTION = """
# Big5Trajectory Chat Assistant (Gradio + OpenAI)
"""

EXAMPLE_SYSTEM = SYSTEM_PROMPT

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-3.5-turbo",
    "gpt-5-nano-2025-08-07",
    "gpt-5-chat-latest",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
]

EXAMPLES = [
    [
        "Write a Python bubble sort in ≤8 lines.",
        EXAMPLE_SYSTEM,
        DEFAULT_MODELS[0],
        0.7,
        None,
        ""
    ],
    [
        "Translate: ‘并行计算的关键在于任务划分与数据局部性。’ into English.",
        EXAMPLE_SYSTEM,
        DEFAULT_MODELS[0],
        0.7,
        None,
        ""
    ],
]


# ==============================
# Gradio UI (with Send button)
# ==============================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(DESCRIPTION)

    # State: history & session ID
    history_state: gr.State = gr.State([])   # List[Tuple[str, str]]
    session_state: gr.State = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            system_box = gr.Textbox(
                label="System Prompt (base)",
                value=EXAMPLE_SYSTEM,
                placeholder="Optional; define assistant role and boundaries.",
                lines=6,
            )
        with gr.Column(scale=2):
            model_drop = gr.Dropdown(
                label="Model",
                choices=DEFAULT_MODELS,
                value=DEFAULT_MODELS[0],
                allow_custom_value=True,
                interactive=True,
            )
            temperature_slider = gr.Slider(
                label="temperature (diversity)",
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
            )
            max_tokens_box = gr.Number(
                label="max_tokens (leave blank/≤0 = unlimited)",
                value=None,
                precision=0,
            )

    # Personality panel
    with gr.Accordion("🧠 Personality (OCEAN)", open=True):
        with gr.Row():
            enable_base_ck = gr.Checkbox(value=True, label="Enable base System Prompt (above)")
        with gr.Row():
            O_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="O - Openness")
            C_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="C - Conscientiousness")
            E_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="E - Extraversion")
            A_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="A - Agreeableness")
            N_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="N - Neuroticism")
        dyn_prompt_preview = gr.Textbox(
            label="🧩 Dynamic System Prompt (read-only preview)",
            value="",
            lines=6,
            interactive=False,
        )

    session_md = gr.Markdown("")

    # Chat + input + buttons
    chatbot = gr.Chatbot(height=520, type="messages", show_copy_button=True)
    with gr.Row():
        msg_box = gr.Textbox(placeholder="Type your message, press Enter or click Send...", lines=2, scale=8)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear chat", scale=1)

    # Init session_id
    def _init_session():
        sid = str(uuid.uuid4())
        md = f"**Session ID:** `{sid}` (all messages are logged to `chat_history.jsonl` for retrieval)"
        log_json("session_init", session_id=sid)
        return sid, md

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

    # Update preview when any control changes
    for comp in [system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider]:
        comp.change(
            _update_dyn_prompt,
            inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
            outputs=[dyn_prompt_preview],
        )

    # Also compute preview on load
    demo.load(
        _update_dyn_prompt,
        inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
        outputs=[dyn_prompt_preview],
    )
    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])

    # --- Submit flow: two steps ---

    def user_submit(user_msg: str, history: History):
        if user_msg is None:
            user_msg = ""
        history = (history or []) + [(user_msg, "")]
        messages = history_to_messages(history)
        return gr.update(value=""), history, messages

    def bot_respond(
        history: History,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: str,
    ):
        if not history:
            yield [], history
            return

        user_msg, _ = history[-1]
        prior = history[:-1]

        partial = ""
        for chunk in stream_chat(
            message=user_msg,
            history=prior,
            system_prompt=system_prompt,
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens) if max_tokens else None,
            session_id=session_id,
        ):
            partial = chunk
            cur = prior + [(user_msg, partial)]
            yield history_to_messages(cur), cur

        final_hist = prior + [(user_msg, partial)]
        yield history_to_messages(final_hist), final_hist

    # Bind events: enter / send button
    msg_box.submit(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot, history_state],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot, history_state],
    )

    # Clear
    def clear_chat():
        return [], [], []

    clear_btn.click(
        clear_chat,
        inputs=None,
        outputs=[history_state, chatbot, msg_box],
    )

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=[msg_box, system_box, model_drop, temperature_slider, max_tokens_box],
        examples_per_page=8,
        label="Examples",
        cache_examples=False,
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
