# app.py
import os
import json
import uuid
from typing import List, Tuple, Generator, Optional

import gradio as gr
from loguru import logger
from openai import OpenAI

from prompt import big5_system_prompts_en, SYSTEM_PROMPT
from predictor import llmClient
import time
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

client = llmClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=1,
    timeout=60,
    )


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


from math import isfinite

def _nearest_key(d: dict[float, str], v: float) -> float:
    """从字典 d 的键中选距离 v 最近的一个（键为 0.0~1.0 的离散点）"""
    keys = list(d.keys())
    # 兜底：如果字典为空（不太可能），直接返回 v 四舍五入到 1 位小数
    if not keys:
        return round(v, 1)
    # 正常从已有键里挑最近
    return min(keys, key=lambda k: abs(k - v))

def generate_dynamic_system_prompt(
    base_text: str,
    enable_base: bool,
    vals: dict[str, float],
    table: dict[str, dict[float, str]],
) -> str:
    """组合基础提示词 + 五维人格分档提示词"""
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
        # 若该档不存在，用最近的键兜底
        if bucket not in table[trait]:
            bucket = _nearest_key(table[trait], bucket)
        parts.append(table[trait][bucket])

    return " ".join(parts).strip()

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

    # 拼 messages
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
        # 同步 UI 选择到 llmClient
        client.change_model(model)
        client.change_temperature(temperature)

        t0 = time.time()
        for i, inc in enumerate(client.chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens if (max_tokens and max_tokens > 0) else None,
        ), start=1):
            partial += inc
            # 这里打印进度日志：第几块，增量长度，总长度，耗时
            logger.info(f"[stream_chat] chunk={i}, inc_len={len(inc)}, total_len={len(partial)}, elapsed={time.time()-t0:.2f}s")
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
# 配置 & 示例
# ==============================
DESCRIPTION = """
# Big5Tragectory 聊天助手（Gradio + OpenAI）
"""

EXAMPLE_SYSTEM = SYSTEM_PROMPT

DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo","gpt-5-nano-2025-08-07","gpt-5-chat-latest","o3", "o3-mini", "o1", "o1-mini"]

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
                label="System Prompt（基础提示词）",
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

    # ← 在这下面加“人格面板”
    with gr.Accordion("🧠 Personality (OCEAN)", open=True):
        with gr.Row():
            enable_base_ck = gr.Checkbox(value=True, label="启用基础提示词（上面的 System Prompt）")
        with gr.Row():
            O_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="O - Openness")
            C_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="C - Conscientiousness")
            E_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="E - Extraversion")
            A_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="A - Agreeableness")
            N_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="N - Neuroticism")
        dyn_prompt_preview = gr.Textbox(
            label="🧩 动态系统提示词（只读预览）",
            value="",
            lines=6,
            interactive=False,
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
            return f"[动态提示词生成错误] {type(e).__name__}: {e}"

    # 绑定变化：任一控件变化就刷新预览
    for comp in [system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider]:
        comp.change(
            _update_dyn_prompt,
            inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
            outputs=[dyn_prompt_preview],
        )

    # 初次加载时也计算一次
    demo.load(
        _update_dyn_prompt,
        inputs=[system_box, enable_base_ck, O_slider, C_slider, E_slider, A_slider, N_slider],
        outputs=[dyn_prompt_preview],
    )
    demo.load(_init_session, inputs=None, outputs=[session_state, session_md])
    # --- 提交流程：分两步 ---

    def user_submit(user_msg: str, history: History):
        if user_msg is None:
            user_msg = ""
        # 先占位 assistant
        history = (history or []) + [(user_msg, "")]
        messages = history_to_messages(history)
        # 返回：清空输入框、更新 history_state（占位版）、让 Chatbot 先显示到“我说完了”
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
            # 双输出：chatbot, history_state
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
            # 一边流式渲染 Chatbot，一边把“当前 partial”写回 history_state
            yield history_to_messages(cur), cur

        # 最终一次（确保完成态），把最终回复持久写回 history_state
        final_hist = prior + [(user_msg, partial)]
        yield history_to_messages(final_hist), final_hist

    # 事件绑定：输入框回车 / 发送按钮
    msg_box.submit(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot, history_state],   # ← 这里改成两个输出
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, history_state],
        outputs=[msg_box, history_state, chatbot],
    ).then(
        bot_respond,
        inputs=[history_state, system_box, model_drop, temperature_slider, max_tokens_box, session_state],
        outputs=[chatbot, history_state],   # ← 同上
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
        inputs=[msg_box, system_box, model_drop, temperature_slider, max_tokens_box],
        examples_per_page=8,
        label="示例",
        cache_examples=False,
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
