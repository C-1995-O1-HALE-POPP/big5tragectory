# app.py
import os
import json
import uuid
from typing import List, Tuple, Generator, Optional

import gradio as gr
from loguru import logger
from openai import OpenAI

# ==============================
# Loguru：JSON 行日志 + 轮转
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
    format="{message}",  # 直接写 JSON 行
)

def log_json(event: str, **kwargs):
    logger.info(json.dumps({"event": event, **kwargs}, ensure_ascii=False))


# ==============================
# OpenAI 客户端
# ==============================
def build_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 OPENAI_API_KEY")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

client = build_client()


# ==============================
# 工具：历史 <-> messages 转换
# ==============================
History = List[Tuple[str, str]]

def history_to_messages(hist: History) -> list:
    """将 [(user, assistant), ...] 转为 [{'role','content'}, ...]"""
    msgs = []
    for u, a in hist or []:
        if u is not None and u != "":
            msgs.append({"role": "user", "content": u})
        if a is not None and a != "":
            msgs.append({"role": "assistant", "content": a})
    return msgs


# ==============================
# 核心对话（流式）+ 日志
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

    # 组装 messages
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # 日志：本轮开始 & 用户消息
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

    # OpenAI Chat Completions（流式）
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        stream=True,
    )

    partial = ""
    try:
        for chunk in stream:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta and delta.content:
                partial += delta.content
                yield partial
        log_json("assistant_message", session_id=session_id, text=partial)
        log_json("round_end", session_id=session_id, tokens=len(partial))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log_json("error", session_id=session_id, where="stream_chat", detail=err)
        yield partial + f"\n\n[Error] {err}"


# ==============================
# 配置 & 示例
# ==============================
DESCRIPTION = """
# 🔧 Gradio × OpenAI Chat Completions × Loguru
- 流式输出、上下文记忆、可自定义系统提示词与模型  
- `chat_history.jsonl` 持久化保存对话（JSON Lines），自动轮转/保留/压缩  
- 每个浏览器会话拥有独立 `session_id`
"""

EXAMPLE_SYSTEM = "你是一个有帮助、严谨且简洁的中文助理。"
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"]

EXAMPLES = [
    [
        "用 8 行以内代码写一个 Python 冒泡排序示例。",  # message
        EXAMPLE_SYSTEM,                                  # system_prompt
        DEFAULT_MODELS[0],                               # model
        0.7,                                             # temperature
        None,                                            # max_tokens
        ""                                               # session_id 占位（init 时填充）
    ],
    [
        "把这段话翻译成英文：'并行计算的关键在于任务划分与数据局部性。'",
        EXAMPLE_SYSTEM,
        DEFAULT_MODELS[0],
        0.7,
        None,
        ""
    ],
]


# ==============================
# 自定义 Blocks UI（Gradio 5 兼容，带发送按钮）
# ==============================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(DESCRIPTION)

    # State：历史 & 会话 ID
    history_state: gr.State = gr.State([])       # List[Tuple[str, str]]
    session_state: gr.State = gr.State("")

    with gr.Row():
        with gr.Column(scale=3):
            system_box = gr.Textbox(
                label="System Prompt（系统提示词）",
                value=EXAMPLE_SYSTEM,
                placeholder="可为空；用于限定助手角色与边界",
                lines=6,
            )
        with gr.Column(scale=2):
            model_drop = gr.Dropdown(
                label="OpenAI 模型",
                choices=DEFAULT_MODELS,
                value=DEFAULT_MODELS[0],
                allow_custom_value=True,
                interactive=True,
            )
            temperature_slider = gr.Slider(
                label="temperature（多样性）",
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
            )
            max_tokens_box = gr.Number(
                label="max_tokens（回复上限，留空/≤0 表示不限制）",
                value=None,
                precision=0,
            )

    session_md = gr.Markdown("")

    # Chat 显示 + 输入框 + 按钮
    chatbot = gr.Chatbot(height=520, type="messages", show_copy_button=True)
    with gr.Row():
        msg_box = gr.Textbox(placeholder="输入你的问题，按 Enter 或点 发送...", lines=2, scale=8)
        send_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空对话", scale=1)

    # 初始化 session_id
    def _init_session():
        sid = str(uuid.uuid4())
        md = f"**Session ID:** `{sid}`（此会话的所有消息会写入 `chat_history.jsonl`，便于检索）"
        log_json("session_init", session_id=sid)
        return sid, md

    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])

    # --- 提交流程：分两步 ---
    # 1) 用户提交：更新历史，并把 messages 映射给 chatbot
    def user_submit(user_msg: str, history: History):
        if user_msg is None:
            user_msg = ""
        history = (history or []) + [(user_msg, "")]  # 先占位 assistant
        messages = history_to_messages(history)
        return gr.update(value=""), history, messages  # 清空输入框 & 刷新 Chatbot

    # 2) 模型响应（流式）：不断产出新的 messages
    def bot_respond(
        history: History,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: str,
    ):
        if not history:
            yield []  # nothing to show
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
            yield history_to_messages(cur)

        # 最终一次（确保完成态）
        cur = prior + [(user_msg, partial)]
        yield history_to_messages(cur)

    # 事件绑定：输入框回车 / 发送按钮
    msg_box.submit(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot],
    )

    # 清空按钮
    def clear_chat():
        return [], [], []  # history_state, chatbot(messages), msg_box

    clear_btn.click(
        clear_chat,
        inputs=None,
        outputs=[history_state, chatbot, msg_box],
    )

    # 示例：把 examples 正确注入到 inputs
    gr.Examples(
        examples=EXAMPLES,
        inputs=[msg_box, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        examples_per_page=8,
        label="示例",
    )

    with gr.Accordion("⚙️ 环境信息", open=False):
        gr.Markdown(
            f"""
- `OPENAI_BASE_URL`: `{os.getenv("OPENAI_BASE_URL", "") or "(未设置)"}`
- `OPENAI_API_KEY`: `{"已设置" if os.getenv("OPENAI_API_KEY") else "未设置"}`  
- 日志文件：`chat_history.jsonl`（JSON Lines；自动轮转、保留 7 天、压缩）
"""
        )

if __name__ == "__main__":
    demo.queue(max_size=64).launch(server_name="0.0.0.0", server_port=27861, share=False)
