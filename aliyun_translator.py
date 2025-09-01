# -*- coding: utf-8 -*-
"""
reduce_persona_seeds_aliyun_zh.py (only Aliyun, concurrent)
==========================================================

- 并发翻译到中文：只用阿里云机器翻译（alimt）。
- 仅保留 N 个“独立种子”，四种组合共 4N 条。
- 覆盖 first_message 与 conversation[*].content → 最终只保留中文。
- 线程池并发 + 简易 QPS 限速；loguru 输出进度。

依赖：
  pip install alibabacloud_alimt20181012 alibabacloud_tea_openapi loguru

所需环境变量：
  ALIBABA_CLOUD_ACCESS_KEY_ID
  ALIBABA_CLOUD_ACCESS_KEY_SECRET
  (可选) ALIBABA_CLOUD_REGION_ID, 默认 "cn-hangzhou"
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


# =============== 简易滑动窗口限速 ===============
class _RateLimiter:
    def __init__(self, qps: int):
        self.qps = max(1, int(qps))
        self._dq = deque()
        self._lock = threading.Lock()

    def throttle(self):
        while True:
            with self._lock:
                now = time.time()
                while self._dq and now - self._dq[0] >= 1.0:
                    self._dq.popleft()
                if len(self._dq) < self.qps:
                    self._dq.append(now)
                    return
                wait = 1.0 - (now - self._dq[0])
            if wait > 0:
                time.sleep(wait)


# =============== 统一翻译接口（基类） ===============
class BaseTranslator:
    def __init__(self, workers: int = 8, qps: int = 20):
        self.workers = max(1, int(workers))
        self.limiter = _RateLimiter(qps=max(1, int(qps)))

    def _call(self, text: str) -> str:
        raise NotImplementedError

    def translate_one(self, text: str, max_retries: int = 3, backoff: float = 0.7) -> str:
        text = text or ""
        if not text.strip():
            return text
        for attempt in range(1, max_retries + 1):
            try:
                self.limiter.throttle()
                zh = self._call(text)
                if zh:
                    return zh.strip()
                raise RuntimeError("empty result")
            except Exception as e:
                wait = backoff * attempt
                logger.warning(f"[翻译重试] 第 {attempt}/{max_retries} 次失败：{e}; {wait:.1f}s 后重试")
                time.sleep(wait)
        logger.error("[翻译失败] 已用尽重试，保留原文")
        return text

    def translate_many(self, texts: List[str], max_retries: int = 3, backoff: float = 0.7) -> List[str]:
        results: List[Optional[str]] = [None] * len(texts)
        tasks = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        def _worker(idx: int, txt: str) -> Tuple[int, str]:
            zh = self.translate_one(txt, max_retries=max_retries, backoff=backoff)
            return idx, zh

        if not tasks:
            return texts

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut_map = {ex.submit(_worker, i, t): i for i, t in tasks}
            done_ct, total_ct = 0, len(tasks)
            for fut in as_completed(fut_map):
                i = fut_map[fut]
                try:
                    idx, zh = fut.result()
                    results[idx] = zh
                except Exception as e:
                    logger.error(f"[并发任务异常] idx={i}: {e}")
                    results[i] = texts[i]
                done_ct += 1
                if done_ct % 10 == 0 or done_ct == total_ct:
                    logger.info(f"[并发进度] {done_ct}/{total_ct}")

        return [results[i] if results[i] is not None else (texts[i] or "") for i in range(len(texts))]


# =============== 阿里云机器翻译后端（唯一后端） ===============
class AliyunTranslator(BaseTranslator):
    def __init__(self, **kw):
        super().__init__(workers=kw.get("workers", 8), qps=kw.get("qps", 20))
        # SDK
        from alibabacloud_tea_openapi import models as open_api_models
        from alibabacloud_alimt20181012.client import Client as AlimtClient
        from alibabacloud_alimt20181012 import models as alimt_models

        access_key_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
        access_key_secret = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        region_id = os.getenv("ALIBABA_CLOUD_REGION_ID", "cn-hangzhou")
        if not access_key_id or not access_key_secret:
            raise RuntimeError("缺少 ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量。")

        self._open_api_models = open_api_models
        self._alimt_models = alimt_models

        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=region_id,
            endpoint="mt.aliyuncs.com",
        )
        self._client = AlimtClient(config)

        # 固定参数
        self._format_type = "text"  # 纯文本
        self._scene = "general"     # 通用领域
        self._src_lang = "auto"
        self._tgt_lang = "zh"       # 简体中文；也可用 "zh-CN"

    def _call(self, text: str) -> str:
        req = self._alimt_models.TranslateGeneralRequest(
            source_language=self._src_lang,
            target_language=self._tgt_lang,
            source_text=text,
            format_type=self._format_type,
            scene=self._scene,
        )
        resp = self._client.translate_general(req)
        data = getattr(getattr(resp, "body", None), "data", None)
        zh = getattr(data, "translated", None) if data else None
        return zh or text


# =============== 数据处理工具函数 ===============
def normalize_seed(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())

def detect_combo_layout(obj: Any) -> Tuple[bool, Dict[str, List[dict]]]:
    if isinstance(obj, dict):
        combos = {k: v for k, v in obj.items() if isinstance(v, list)}
        if combos:
            return True, combos
    if isinstance(obj, list):
        return False, {"single": obj}
    raise ValueError("无法识别的 JSON 顶层结构，应为 dict 或 list。")

def build_seed_index(combos: Dict[str, List[dict]]) -> Dict[str, Dict[str, dict]]:
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
    commons = [s for s, per in seed_map.items() if all(ck in per for ck in combo_keys)]
    logger.info(f"[交集] 在组合 {combo_keys} 上的公共种子数 = {len(commons)}")
    if len(commons) < num:
        raise ValueError(f"公共种子不足：仅 {len(commons)} 个，要求 {num} 个。")
    commons.sort(key=lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    chosen = commons[:num]
    logger.info(f"[选择] 选取前 {num} 个公共种子（SHA1 排序稳定抽样）")
    return chosen

def translate_item_concurrent(translator: BaseTranslator, item: dict) -> None:
    texts: List[str] = []
    index_map: List[Tuple[str, int]] = []

    fm = item.get("first_message", "")
    if isinstance(fm, str):
        index_map.append(("fm", -1))
        texts.append(fm)
    else:
        index_map.append(("fm", -1))
        texts.append("")

    conv = item.get("conversation", [])
    if isinstance(conv, list):
        for idx, m in enumerate(conv):
            if isinstance(m, dict) and isinstance(m.get("content", ""), str):
                index_map.append(("conv", idx))
                texts.append(m["content"])
            else:
                index_map.append(("conv", idx))
                texts.append("")

    zh_texts = translator.translate_many(texts)

    cursor = 0
    for kind, idx in index_map:
        zh = zh_texts[cursor]
        cursor += 1
        if kind == "fm":
            item["first_message"] = zh
        else:
            if isinstance(conv, list) and 0 <= idx < len(conv) and isinstance(conv[idx], dict):
                conv[idx]["content"] = zh


def process(
    data: Any,
    combo_keys: Optional[List[str]],
    num_seeds: int,
    translator: BaseTranslator
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


# =============== CLI ===============
def main():
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add("reduce_25seed_aliyun.log", rotation="5 MB", retention="7 days",
               encoding="utf-8", enqueue=True, level="INFO")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 JSON 路径")
    ap.add_argument("--output", required=True, help="输出 JSON 路径")
    ap.add_argument("--num_seeds", type=int, default=25, help="要保留的种子数量（默认 25）")
    ap.add_argument("--combo-keys", nargs="*", default=None,
                    help="指定需要保留的组合键；不指定则自动读取全部键")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    ap.add_argument("--qps", type=int, default=20, help="每秒最大请求数（默认 20）")
    args = ap.parse_args()

    translator = AliyunTranslator(workers=args.workers, qps=args.qps)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    t0 = time.time()
    out_mapping = process(
        data=data,
        combo_keys=args.combo_keys,
        num_seeds=args.num_seeds,
        translator=translator
    )
    dt = time.time() - t0

    meta = {
        "_meta": {
            "num_seeds": args.num_seeds,
            "combo_keys": list(out_mapping.keys()),
            "translator_provider": "aliyun",
            "elapsed_sec": round(dt, 2),
            "workers": args.workers,
            "qps": args.qps,
        }
    }
    out_obj = dict(out_mapping)
    out_obj.update(meta)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in out_mapping.values())
    logger.info(f"[完成] 输出：{args.output}；组合={len(out_mapping)}；总条目={total}；耗时={dt:.2f}s；workers={args.workers}；qps={args.qps}")

if __name__ == "__main__":
    main()
