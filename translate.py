# -*- coding: utf-8 -*-
"""
reduce_persona_seeds_gpt_zh.py (concurrent)
==========================================

- GPT 翻译（OpenAI 兼容），用线程池并发翻译。
- 仅保留 25 个“独立种子”，四种组合共 100 条。
- 覆盖 first_message 与 conversation[*].content → 只保留中文。
- loguru 输出必要进度与统计。
- 并发参数：--workers 并发线程数；--qps 每秒最大请求数（近似）。

用法：
    python reduce_persona_seeds_gpt_zh.py \
        --input /mnt/data/simulated_persona_dialogues_4combos.json \
        --output /mnt/data/simulated_persona_dialogues_4combos_25seed_zh.json \
        --num_seeds 25 \
        --combo-keys stable_static stable_dynamic switch_static switch_dynamic \
        --model gpt-4o-mini \
        --workers 8 --qps 6
"""

import argparse
import json
import os
import time
import hashlib
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional

from loguru import logger


# =============== OpenAI 兼容客户端 + 并发限速器 ===============
class _RateLimiter:
    """
    简单滑动窗口限速器：保证近 1 秒内请求数不超过 qps。
    线程安全，用于多线程前的 _throttle()。
    """
    def __init__(self, qps: int):
        self.qps = max(1, int(qps))
        self._dq = deque()  # 存储最近请求时间戳
        self._lock = threading.Lock()

    def throttle(self):
        while True:
            with self._lock:
                now = time.time()
                # 移除 1 秒前的时间戳
                while self._dq and now - self._dq[0] >= 1.0:
                    self._dq.popleft()
                if len(self._dq) < self.qps:
                    self._dq.append(now)
                    return
                # 需要等待
                wait = 1.0 - (now - self._dq[0])
            if wait > 0:
                time.sleep(wait)


class GPTTranslator:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, timeout: int = 60,
                 workers: int = 8, qps: int = 6):
        from openai import OpenAI  # 需要安装 openai>=1.0
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 OPENAI_API_KEY。")

        base_url = os.getenv("OPENAI_BASE_URL", None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.timeout = timeout

        self.system_prompt = (
            "你是专业的翻译引擎。将输入文本翻译为自然流畅、准确的简体中文。"
            "保留角色人称与语气，不添加解释或总结，不扩写，不省略。"
            "如果句首有 'Persona A:'、'User:' 等前缀，保留前缀并仅翻译后面的内容。"
        )

        self.workers = max(1, int(workers))
        self.limiter = _RateLimiter(qps=max(1, int(qps)))

    def _translate_call(self, text: str) -> str:
        # 限速器先通过，再发请求
        self.limiter.throttle()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def translate_one(self, text: str, max_retries: int = 3, backoff: float = 0.7) -> str:
        """单条文本翻译（串行调用），保留重试逻辑。"""
        text = text or ""
        if not text.strip():
            return text
        for attempt in range(1, max_retries + 1):
            try:
                zh = self._translate_call(text)
                if zh:
                    return zh
                raise RuntimeError("空响应内容")
            except Exception as e:
                wait = backoff * attempt
                logger.warning(f"[翻译重试] 第 {attempt}/{max_retries} 次失败：{e}; {wait:.1f}s 后重试")
                time.sleep(wait)
        logger.error("[翻译失败] 已用尽重试，保留原文")
        return text

    def translate_many(self, texts: List[str], max_retries: int = 3, backoff: float = 0.7) -> List[str]:
        """
        并发翻译一批文本。
        - 维持输入顺序：按 index 回填。
        - 每个任务内部仍带重试，并受全局限速器控制。
        """
        results = [None] * len(texts)  # type: ignore
        # 预先记录非空索引，空串直接原样返回
        tasks = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        def _worker(idx: int, txt: str) -> Tuple[int, str]:
            # 单任务：调用 translate_one
            zh = self.translate_one(txt, max_retries=max_retries, backoff=backoff)
            return idx, zh

        if not tasks:
            return texts

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut_map = {ex.submit(_worker, i, t): i for i, t in tasks}
            done_ct = 0
            total_ct = len(tasks)
            for fut in as_completed(fut_map):
                i = fut_map[fut]
                try:
                    idx, zh = fut.result()
                    results[idx] = zh
                except Exception as e:
                    logger.error(f"[并发任务异常] idx={i}: {e}")
                    # 出现异常则保底原文
                    results[i] = texts[i]
                done_ct += 1
                if done_ct % 10 == 0 or done_ct == total_ct:
                    logger.info(f"[并发进度] {done_ct}/{total_ct}")

        # 填充空白/None
        out = []
        for i, t in enumerate(texts):
            out.append(results[i] if results[i] is not None else (t or ""))
        return out


# =============== 工具函数 ===============
def normalize_seed(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())

def detect_combo_layout(obj: Any) -> Tuple[bool, Dict[str, List[dict]]]:
    """
    返回 (is_mapping, combos_dict)
    - dict: 认为每个 key 是一个组合
    - list: 则合成 single 组合
    """
    if isinstance(obj, dict):
        combos = {k: v for k, v in obj.items() if isinstance(v, list)}
        if combos:
            return True, combos
    if isinstance(obj, list):
        return False, {"single": obj}
    raise ValueError("无法识别的 JSON 顶层结构，应为 dict 或 list。")

def build_seed_index(combos: Dict[str, List[dict]]) -> Dict[str, Dict[str, dict]]:
    """
    seed -> { combo_key -> item }
    以 item['first_message'] 作为种子键（规范化）；缺失时用 dialogue_id 兜底。
    """
    seed_map: Dict[str, Dict[str, dict]] = {}
    total = 0
    for ck, items in combos.items():
        for it in items:
            total += 1
            seed = normalize_seed(it.get("first_message", "")) or normalize_seed(it.get("dialogue_id", ""))
            if not seed:
                logger.warning(f"[跳过] 找不到 first_message / dialogue_id，组合={ck}")
                continue
            seed_map.setdefault(seed, {})
            seed_map[seed][ck] = it
    logger.info(f"[索引] 共收集条目 {total}，形成种子 {len(seed_map)} 个")
    return seed_map

def pick_common_seeds(seed_map: Dict[str, Dict[str, dict]], combo_keys: List[str], num: int) -> List[str]:
    """选择在所有 combo_keys 中都出现的种子；按 SHA1 稳定排序后取前 num 个。"""
    commons = [s for s, per in seed_map.items() if all(ck in per for ck in combo_keys)]
    logger.info(f"[交集] 在组合 {combo_keys} 上的公共种子数 = {len(commons)}")
    if len(commons) < num:
        raise ValueError(f"公共种子不足：仅 {len(commons)} 个，要求 {num} 个。")
    commons.sort(key=lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    chosen = commons[:num]
    logger.info(f"[选择] 选取前 {num} 个公共种子（SHA1 排序稳定抽样）")
    return chosen

def translate_item_concurrent(translator: GPTTranslator, item: dict) -> None:
    """
    原地覆盖 item：
      - first_message
      - conversation[*].content
    采用批量并发翻译，回填保持顺序。
    """
    texts: List[str] = []
    index_map: List[Tuple[str, int]] = []  # ("fm", -1) 或 ("conv", idx)

    # first_message
    fm = item.get("first_message", "")
    if isinstance(fm, str):
        index_map.append(("fm", -1))
        texts.append(fm)
    else:
        index_map.append(("fm", -1))
        texts.append("")

    # conversation
    conv = item.get("conversation", [])
    if isinstance(conv, list):
        for idx, m in enumerate(conv):
            if isinstance(m, dict) and isinstance(m.get("content", ""), str):
                index_map.append(("conv", idx))
                texts.append(m["content"])
            else:
                index_map.append(("conv", idx))
                texts.append("")

    # 批量并发翻译
    zh_texts = translator.translate_many(texts)

    # 回填
    cursor = 0
    for kind, idx in index_map:
        zh = zh_texts[cursor]
        cursor += 1
        if kind == "fm":
            item["first_message"] = zh
        else:
            # conv
            if isinstance(conv, list) and 0 <= idx < len(conv) and isinstance(conv[idx], dict):
                conv[idx]["content"] = zh
    # 其它元数据保持不动


def process(
    data: Any,
    combo_keys: Optional[List[str]],
    num_seeds: int,
    translator: GPTTranslator
) -> Dict[str, List[dict]]:
    is_mapping, combos = detect_combo_layout(data)
    if not combo_keys:
        combo_keys = list(combos.keys())

    logger.info(f"[组合] 目标组合键：{combo_keys}")
    seed_map = build_seed_index(combos)
    chosen_seeds = pick_common_seeds(seed_map, combo_keys, num_seeds)

    out: Dict[str, List[dict]] = {}
    for ck in combo_keys:
        items = []
        logger.info(f"[处理组合] {ck} ...")
        for i, seed in enumerate(chosen_seeds, 1):
            item = seed_map[seed][ck]
            obj = json.loads(json.dumps(item, ensure_ascii=False))  # 深拷贝
            logger.info(f"  - [{i:02d}/{len(chosen_seeds)}] 并发翻译种子：{seed[:50]}{'...' if len(seed)>50 else ''}")
            translate_item_concurrent(translator, obj)
            items.append(obj)
        out[ck] = items

    return out


def main():
    # ===== 日志设置：控制台 + 文件 =====
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add("reduce_25seed_zh.log", rotation="5 MB", retention="7 days",
               encoding="utf-8", enqueue=True, level="INFO")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 JSON 路径")
    ap.add_argument("--output", required=True, help="输出 JSON 路径")
    ap.add_argument("--num_seeds", type=int, default=25, help="要保留的种子数量（默认 25）")
    ap.add_argument("--combo-keys", nargs="*", default=None,
                    help="指定需要保留的组合键；不指定则自动读取全部键")
    ap.add_argument("--model", default=None, help="OpenAI 兼容接口使用的模型名；默认取 OPENAI_MODEL 或 gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    ap.add_argument("--qps", type=int, default=6, help="每秒最大请求数（默认 6）")
    args = ap.parse_args()

    # ===== 构建翻译器（仅 GPT + 并发限速）=====
    translator = GPTTranslator(model=args.model, workers=args.workers, qps=args.qps)

    # ===== 读取 =====
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ===== 处理 =====
    t0 = time.time()
    out_mapping = process(
        data=data,
        combo_keys=args.combo_keys,
        num_seeds=args.num_seeds,
        translator=translator
    )
    dt = time.time() - t0

    # ===== 写出（保持映射结构；追加元信息）=====
    meta = {
        "_meta": {
            "num_seeds": args.num_seeds,
            "combo_keys": list(out_mapping.keys()),
            "translator": "OpenAI Chat Completions (concurrent)",
            "model": translator.model,
            "elapsed_sec": round(dt, 2),
            "workers": args.workers,
            "qps": args.qps,
        }
    }
    out_obj = dict(out_mapping)
    out_obj.update(meta)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # ===== 摘要 =====
    total = sum(len(v) for v in out_mapping.values())
    logger.info(f"[完成] 输出：{args.output}；组合={len(out_mapping)}；总条目={total}；耗时={dt:.2f}s；workers={args.workers}；qps={args.qps}")

if __name__ == "__main__":
    main()
