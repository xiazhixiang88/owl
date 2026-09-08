#!/usr/bin/env python3
"""Run the GPU-preprocess GigaPath-Flash pipeline with an in-place terminal dashboard.

The extraction implementation lives in
extract_gigapath_flash_features_gpu_preprocess.py. This runner only changes the
terminal presentation: overall status + one fixed line per GPU + one event line.
No tqdm dependency is required.
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import time
from typing import Any

import torch

import extract_gigapath_flash_features as base
import extract_gigapath_flash_features_gpu_preprocess as gp


class TerminalDashboard:
    def __init__(self, gpu_ids: list[int], total_tasks: int, total_task_tiles: int) -> None:
        self.gpu_ids = list(gpu_ids)
        self.total_tasks = total_tasks
        self.total_task_tiles = total_task_tiles
        self.finished = 0
        self.failed = 0
        self.completed_tiles = 0
        self.start = time.perf_counter()
        self.gpu_lines = {gpu: f"GPU {gpu}: loading model..." for gpu in self.gpu_ids}
        self.event = "event: -"
        self.enabled = sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
        self._initialized = False
        self._closed = False
        self._nlines = 2 + len(self.gpu_ids)

    def _fit(self, text: str) -> str:
        width = shutil.get_terminal_size(fallback=(180, 40)).columns
        if width <= 8:
            return text
        if len(text) >= width:
            return text[: max(1, width - 2)] + "~"
        return text

    def _lines(self) -> list[str]:
        elapsed = time.perf_counter() - self.start
        aggregate = self.completed_tiles / elapsed if elapsed > 0 else 0.0
        overall = (
            f"OVERALL  WSI {self.finished}/{self.total_tasks}"
            f" | failed={self.failed}"
            f" | completed tiles={self.completed_tiles:,}/{self.total_task_tiles:,}"
            f" | elapsed={elapsed:.1f}s"
            f" | completed-throughput={aggregate:.1f} tile/s"
        )
        lines = [overall]
        lines.extend(self.gpu_lines[gpu] for gpu in self.gpu_ids)
        lines.append(self.event)
        return [self._fit(line) for line in lines]

    def render(self) -> None:
        if self._closed:
            return
        if not self.enabled:
            return

        lines = self._lines()
        if not self._initialized:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.write("\n" * self._nlines)
            self._initialized = True

        sys.stdout.write(f"\x1b[{self._nlines}A")
        for line in lines:
            sys.stdout.write("\r\x1b[2K" + line + "\n")
        sys.stdout.flush()

    def plain(self, text: str) -> None:
        if not self.enabled:
            print(text, flush=True)

    def set_gpu(self, gpu: int, text: str) -> None:
        self.gpu_lines[gpu] = text
        self.render()

    def set_event(self, text: str) -> None:
        self.event = "event: " + text
        self.render()

    def progress(self, msg: dict[str, Any]) -> None:
        gpu = int(msg["gpu"])
        processed = int(msg["processed"])
        produced = int(msg["produced"])
        total = int(msg["total"])
        pct = 100.0 * processed / total if total else 100.0
        self.gpu_lines[gpu] = (
            f"GPU {gpu}  {msg['case_id']}"
            f" | {processed:,}/{total:,} ({pct:5.1f}%)"
            f" | {float(msg['rate']):.1f} tile/s"
            f" | CPU={produced:,}/{total:,}"
            f" | queue={int(msg['queue'])}/{int(msg['queue_max'])}"
        )
        self.render()

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled:
            self.render()
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        self._closed = True


def main() -> None:
    args = gp.parse_args()
    gpu_ids = base.parse_gpu_ids(args.gpus)
    gp.validate_args(args, gpu_ids)

    checkpoint = base.resolve_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, missing_wsi, existing_output = base.discover_tasks(
        args.data_dir,
        args.coords_dir,
        args.output_dir,
        args.overwrite,
        args.max_cases,
    )

    raw_batch_mib = args.batch_size * 3 * gp.RAW_SIZE * gp.RAW_SIZE / (1024 ** 2)
    print("GigaPath-Flash GPU-preprocess producer/consumer extraction", flush=True)
    print(f"  GPUs:                  {gpu_ids}", flush=True)
    print(f"  concurrent WSIs:       {len(gpu_ids)}", flush=True)
    print(f"  parser threads / WSI:  {args.parser_threads}", flush=True)
    print(f"  total parser threads:  {args.parser_threads * len(gpu_ids)}", flush=True)
    print(f"  batch size / GPU:      {args.batch_size}", flush=True)
    print(f"  raw batch size:        {raw_batch_mib:.0f} MiB pinned", flush=True)
    print(f"  queue depth / GPU:     {args.queue_batches} raw batches", flush=True)
    print(f"  preprocess:            GPU bicubic antialias + center crop", flush=True)
    print(f"  progress display:      in-place dashboard ({args.progress_interval:g}s updates)", flush=True)
    print(f"  alignment samples:     {args.validate_preprocess} / GPU", flush=True)
    print(f"  alignment min cosine:  {args.alignment_min_cosine}", flush=True)
    print(f"  pending WSIs:          {len(tasks)}", flush=True)
    print(f"  existing outputs skip: {existing_output}", flush=True)
    print(f"  missing WSI skip:      {missing_wsi}", flush=True)
    print(f"  checkpoint:            {checkpoint}", flush=True)

    if not tasks:
        print("Nothing to process.", flush=True)
        return

    cfg = gp.WorkerConfig(
        parser_threads=args.parser_threads,
        batch_size=args.batch_size,
        queue_batches=args.queue_batches,
        amp_dtype=args.amp_dtype,
        feature_dtype=args.feature_dtype,
        compile_model=args.compile_model,
        progress_interval=args.progress_interval,
        validate_preprocess=args.validate_preprocess,
        alignment_min_cosine=args.alignment_min_cosine,
    )

    ctx = torch.multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    workers = [
        ctx.Process(
            target=gp.gpu_worker,
            args=(gpu_id, task_q, result_q, str(checkpoint), cfg),
            name=f"gigapath-gpu-preprocess-{gpu_id}",
        )
        for gpu_id in gpu_ids
    ]
    for proc in workers:
        proc.start()

    for task in tasks:
        task_q.put(task)
    for _ in workers:
        task_q.put(None)

    total_tasks = len(tasks)
    total_task_tiles = sum(task.num_tiles for task in tasks)
    dashboard = TerminalDashboard(gpu_ids, total_tasks, total_task_tiles)
    worker_errors = 0
    error_details: list[str] = []

    dashboard.render()

    try:
        while dashboard.finished < total_tasks:
            try:
                msg = result_q.get(timeout=1.0)
            except queue.Empty:
                dashboard.render()
                if all(not proc.is_alive() for proc in workers):
                    break
                continue

            kind = msg.get("type")
            gpu = int(msg.get("gpu", -1))

            if kind == "worker_ready":
                text = f"GPU {gpu}: model ready | waiting for WSI"
                dashboard.set_gpu(gpu, text)
                dashboard.plain(text)

            elif kind == "wsi_start":
                text = f"GPU {gpu}  {msg['case_id']} | 0/{int(msg['total']):,} (  0.0%) | starting"
                dashboard.set_gpu(gpu, text)
                dashboard.plain(text)

            elif kind == "progress":
                dashboard.progress(msg)
                if not dashboard.enabled:
                    dashboard.plain(base.format_progress(msg))

            elif kind == "alignment":
                text = gp.format_alignment(msg)
                dashboard.set_event(text)
                dashboard.plain(text)

            elif kind == "warning":
                text = f"GPU {gpu} WARN {msg['case_id']}: {msg['message']}"
                dashboard.set_event(text)
                dashboard.plain(text)

            elif kind == "done":
                dashboard.finished += 1
                dashboard.completed_tiles += int(msg["tiles"])
                done_text = base.format_done(msg)
                dashboard.set_gpu(
                    gpu,
                    f"GPU {gpu}: DONE {msg['case_id']} | {float(msg['tile_per_s']):.1f} tile/s | waiting next WSI",
                )
                dashboard.set_event(f"[{dashboard.finished}/{total_tasks}] {done_text}")
                dashboard.plain(f"[{dashboard.finished}/{total_tasks}] {done_text}")

            elif kind == "task_error":
                dashboard.finished += 1
                dashboard.failed += 1
                text = f"GPU {gpu} ERROR {msg['case_id']}: {msg['error']}"
                dashboard.set_gpu(gpu, text)
                dashboard.set_event(text)
                dashboard.plain(text)
                error_details.append(str(msg.get("trace", "")))

            elif kind == "worker_error":
                worker_errors += 1
                text = f"GPU {gpu} WORKER ERROR: {msg['error']}"
                dashboard.set_gpu(gpu, text)
                dashboard.set_event(text)
                dashboard.plain(text)
                error_details.append(str(msg.get("trace", "")))

        for proc in workers:
            proc.join()

    finally:
        dashboard.close()

    elapsed = time.perf_counter() - dashboard.start
    if dashboard.finished < total_tasks:
        if error_details:
            print("\n".join(x for x in error_details if x), file=sys.stderr, flush=True)
        raise RuntimeError(
            f"All GPU workers exited with {total_tasks - dashboard.finished} WSI tasks unreported. "
            f"Worker errors: {worker_errors}."
        )

    print("\nFinished", flush=True)
    print(f"  WSIs:       {dashboard.finished - dashboard.failed}/{total_tasks} succeeded", flush=True)
    print(f"  tiles:      {dashboard.completed_tiles:,}", flush=True)
    print(f"  wall time:  {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"  aggregate:  {dashboard.completed_tiles / elapsed:.1f} tile/s", flush=True)

    if dashboard.failed or worker_errors:
        if error_details:
            print("\n".join(x for x in error_details if x), file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
