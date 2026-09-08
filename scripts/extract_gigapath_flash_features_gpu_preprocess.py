#!/usr/bin/env python3
"""Experimental GigaPath-Flash extraction with GPU resize/crop preprocessing.

This keeps the proven multi-GPU producer/consumer design but moves the expensive
512->256 bicubic resize and 224 center crop from CPU parser threads to the GPU.
CPU workers only decode the WSI patch and convert OpenSlide RGBA to RGB.

Before normal extraction, the first non-empty WSI on every GPU can validate the
new GPU preprocessing against the previous PIL reference path on a small sample.
The run aborts if feature cosine similarity is below --alignment-min-cosine.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse task discovery, coordinate parsing, model loading, output format, and
# progress helpers from the stable CPU-preprocess implementation.
import extract_gigapath_flash_features as base


RAW_SIZE = base.LEVEL0_READ_SIZE          # 512 @ 40x
RESIZE_SIZE = base.RESIZE_SIZE             # 256 @ 20x
MODEL_INPUT_SIZE = base.MODEL_INPUT_SIZE   # 224
CROP_OFFSET = base.CROP_OFFSET             # 16


@dataclass(frozen=True)
class WorkerConfig:
    parser_threads: int
    batch_size: int
    queue_batches: int
    amp_dtype: str
    feature_dtype: str
    compile_model: bool
    progress_interval: float
    validate_preprocess: int
    alignment_min_cosine: float


@dataclass
class StagedBatch:
    payload: base.BatchPayload
    images_gpu: torch.Tensor  # uint8 [B,3,224,224]
    ready_event: torch.cuda.Event


def _read_raw_patch_into(dest: torch.Tensor, x_l0: int, y_l0: int) -> None:
    """Decode one 512x512 patch into CHW uint8 pinned memory; no resize/crop."""
    reader = getattr(base._READER_TLS, "reader", None)
    if reader is None:
        raise RuntimeError("OpenSlide reader was not initialized in parser thread")

    rgba = reader.read_region((int(x_l0), int(y_l0)), 0, (RAW_SIZE, RAW_SIZE))

    # Keep exactly the same alpha-compositing behavior as the previous CPU path.
    if rgba.mode == "RGBA":
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = rgba.convert("RGB")

    arr = np.array(rgb, dtype=np.uint8, copy=True)
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    dest.copy_(torch.from_numpy(chw))


def _produce_raw_batches(
    slide_path: str,
    coords_l0: np.ndarray,
    batch_size: int,
    parser_threads: int,
    out_q: queue.Queue,
    stop_event: threading.Event,
    stats: dict[str, float],
) -> None:
    """CPU producer: OpenSlide decode only; bounded queue provides backpressure."""
    try:
        base._close_registered_readers()
        with ThreadPoolExecutor(
            max_workers=parser_threads,
            thread_name_prefix="patch-reader",
            initializer=base._init_slide_reader,
            initargs=(slide_path,),
        ) as pool:
            n = len(coords_l0)
            for start in range(0, n, batch_size):
                if stop_event.is_set():
                    break

                end = min(start + batch_size, n)
                size = end - start
                t0 = time.perf_counter()

                images = torch.empty(
                    (size, 3, RAW_SIZE, RAW_SIZE),
                    dtype=torch.uint8,
                    pin_memory=True,
                )

                futures = [
                    pool.submit(
                        _read_raw_patch_into,
                        images[i],
                        int(coords_l0[start + i, 0]),
                        int(coords_l0[start + i, 1]),
                    )
                    for i in range(size)
                ]
                for future in futures:
                    future.result()

                stats["decode_s"] += time.perf_counter() - t0
                stats["tiles"] += float(size)

                if not base._put_with_backpressure(
                    out_q,
                    base.BatchPayload(images=images, start=start, end=end),
                    stop_event,
                    stats,
                ):
                    return

        if not stop_event.is_set():
            base._put_with_backpressure(out_q, base._END, stop_event, stats)

    except Exception as exc:
        if not stop_event.is_set():
            failure = base.ProducerFailure(message=str(exc), trace=traceback.format_exc())
            base._put_with_backpressure(out_q, failure, stop_event, stats)
    finally:
        base._close_registered_readers()


def _gpu_resize_crop_uint8(raw_gpu: torch.Tensor) -> torch.Tensor:
    """Match PIL BICUBIC downsampling as closely as PyTorch supports.

    PyTorch documents bicubic + antialias=True + align_corners=False as matching
    Pillow for downsampling. Round/clamp restores the uint8 image semantics of the
    original PIL path before normalization.
    """
    x = raw_gpu.to(dtype=torch.float32)
    x = F.interpolate(
        x,
        size=(RESIZE_SIZE, RESIZE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    x = x.round_().clamp_(0.0, 255.0).to(dtype=torch.uint8)
    return x[
        :,
        :,
        CROP_OFFSET : CROP_OFFSET + MODEL_INPUT_SIZE,
        CROP_OFFSET : CROP_OFFSET + MODEL_INPUT_SIZE,
    ].contiguous()


def _stage_batch(
    payload: base.BatchPayload,
    device: torch.device,
    preprocess_stream: torch.cuda.Stream,
) -> StagedBatch:
    """Asynchronously copy raw uint8 patches and preprocess them on the GPU."""
    with torch.cuda.stream(preprocess_stream):
        raw_gpu = payload.images.to(device=device, non_blocking=True)
        images_gpu = _gpu_resize_crop_uint8(raw_gpu)
        ready = torch.cuda.Event(blocking=False)
        ready.record(preprocess_stream)
    return StagedBatch(payload=payload, images_gpu=images_gpu, ready_event=ready)


def _cpu_reference_preprocess(raw_cpu: torch.Tensor, n: int) -> torch.Tensor:
    """Original PIL resize/crop path used only for the one-time alignment check."""
    refs: list[torch.Tensor] = []
    for i in range(min(n, len(raw_cpu))):
        arr = raw_cpu[i].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr, mode="RGB")
        img = img.resize((RESIZE_SIZE, RESIZE_SIZE), resample=Image.Resampling.BICUBIC)
        img = img.crop(
            (
                CROP_OFFSET,
                CROP_OFFSET,
                CROP_OFFSET + MODEL_INPUT_SIZE,
                CROP_OFFSET + MODEL_INPUT_SIZE,
            )
        )
        arr224 = np.array(img, dtype=np.uint8, copy=True)
        refs.append(torch.from_numpy(np.ascontiguousarray(arr224.transpose(2, 0, 1))))
    return torch.stack(refs, dim=0)


def _normalize_uint8(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    x = images.to(dtype=torch.float32)
    x.mul_(1.0 / 255.0)
    x.sub_(mean).div_(std)
    return x


def _validate_alignment(
    raw_cpu: torch.Tensor,
    gpu_uint8: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    amp_dtype: torch.dtype,
    sample_count: int,
    min_cosine: float,
) -> dict[str, float]:
    n = min(sample_count, len(raw_cpu))
    if n <= 0:
        return {}

    ref_cpu = _cpu_reference_preprocess(raw_cpu, n)
    new_cpu = gpu_uint8[:n].detach().cpu()

    diff = (ref_cpu.to(torch.int16) - new_cpu.to(torch.int16)).abs()
    max_pixel_diff = float(diff.max().item())
    mean_pixel_diff = float(diff.float().mean().item())
    exact_pixel_pct = float((diff == 0).float().mean().item() * 100.0)

    ref_gpu = ref_cpu.pin_memory().to(device=device, non_blocking=True)
    ref_x = _normalize_uint8(ref_gpu, mean, std)
    new_x = _normalize_uint8(gpu_uint8[:n], mean, std)

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=True,
    ):
        ref_feat = model(ref_x)
        new_feat = model(new_x)

    if isinstance(ref_feat, (tuple, list)):
        ref_feat = ref_feat[0]
    if isinstance(new_feat, (tuple, list)):
        new_feat = new_feat[0]

    cosine = F.cosine_similarity(ref_feat.float(), new_feat.float(), dim=1)
    cosine_mean = float(cosine.mean().item())
    cosine_min = float(cosine.min().item())
    feature_max_abs = float((ref_feat.float() - new_feat.float()).abs().max().item())

    if cosine_mean < min_cosine:
        raise RuntimeError(
            "GPU preprocessing alignment check failed: "
            f"mean cosine={cosine_mean:.8f} < required {min_cosine:.8f}. "
            "No feature file will be written for this WSI."
        )

    return {
        "samples": float(n),
        "max_pixel_diff": max_pixel_diff,
        "mean_pixel_diff": mean_pixel_diff,
        "exact_pixel_pct": exact_pixel_pct,
        "cosine_mean": cosine_mean,
        "cosine_min": cosine_min,
        "feature_max_abs": feature_max_abs,
    }


def run_one_wsi(
    task: base.WSITask,
    model: torch.nn.Module,
    device: torch.device,
    cfg: WorkerConfig,
    gpu_id: int,
    result_q: Any,
    validate_this_wsi: bool,
) -> tuple[dict[str, float], bool]:
    coords_l0, coords_20x = base.load_coords(Path(task.coord_path))
    n_tiles = len(coords_l0)

    if task.num_tiles != n_tiles:
        result_q.put(
            {
                "type": "warning",
                "gpu": gpu_id,
                "case_id": task.case_id,
                "message": f"summary says {task.num_tiles} tiles, coordinate CSV has {n_tiles}",
            }
        )

    if n_tiles == 0:
        empty = torch.empty((0, base.FEATURE_DIM), dtype=base._storage_dtype(cfg.feature_dtype))
        base.save_feature_file(task, empty, coords_20x, coords_l0, cfg.feature_dtype)
        return {
            "tiles": 0.0,
            "wall_s": 0.0,
            "tile_per_s": 0.0,
            "gpu_wait_s": 0.0,
            "producer_blocked_s": 0.0,
            "producer_decode_s": 0.0,
            "gpu_step_s": 0.0,
        }, False

    batch_q: queue.Queue = queue.Queue(maxsize=cfg.queue_batches)
    producer_stop = threading.Event()
    progress_stop = threading.Event()
    producer_stats = {"blocked_s": 0.0, "decode_s": 0.0, "tiles": 0.0}
    progress_state = {"processed": 0.0}

    producer = threading.Thread(
        target=_produce_raw_batches,
        args=(
            task.slide_path,
            coords_l0,
            cfg.batch_size,
            cfg.parser_threads,
            batch_q,
            producer_stop,
            producer_stats,
        ),
        name=f"producer-{task.case_id}",
        daemon=True,
    )

    reporter = threading.Thread(
        target=base._progress_reporter,
        args=(
            gpu_id,
            task.case_id,
            n_tiles,
            batch_q,
            producer_stats,
            progress_state,
            result_q,
            cfg.progress_interval,
            progress_stop,
            cfg.queue_batches,
        ),
        name=f"progress-{task.case_id}",
        daemon=True,
    )

    preprocess_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    mean = torch.tensor(base.IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(base.IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    amp_dtype = base._amp_dtype(cfg.amp_dtype)
    storage_dtype = base._storage_dtype(cfg.feature_dtype)

    features_cpu: list[torch.Tensor] = []
    gpu_wait_s = 0.0
    gpu_step_s = 0.0
    wall_start = time.perf_counter()
    alignment_checked = False

    result_q.put(
        {
            "type": "wsi_start",
            "gpu": gpu_id,
            "case_id": task.case_id,
            "total": n_tiles,
        }
    )
    producer.start()
    reporter.start()

    try:
        first_item, waited = base._get_queue_item(batch_q, blocking=True)
        gpu_wait_s += waited
        if base._handle_special_queue_item(first_item):
            raise RuntimeError(f"Producer ended before yielding patches for {task.case_id}")
        if not isinstance(first_item, base.BatchPayload):
            raise TypeError(f"Unexpected queue item: {type(first_item)}")

        current = _stage_batch(first_item, device, preprocess_stream)
        no_more_batches = False
        expected_start = 0

        while current is not None:
            current_stream.wait_event(current.ready_event)

            if current.payload.start != expected_start:
                raise RuntimeError(
                    f"Batch order error for {task.case_id}: expected {expected_start}, "
                    f"got {current.payload.start}"
                )

            batch_end = current.payload.end
            expected_start = batch_end

            if validate_this_wsi and not alignment_checked and cfg.validate_preprocess > 0:
                stats = _validate_alignment(
                    current.payload.images,
                    current.images_gpu,
                    model,
                    device,
                    mean,
                    std,
                    amp_dtype,
                    cfg.validate_preprocess,
                    cfg.alignment_min_cosine,
                )
                result_q.put(
                    {
                        "type": "alignment",
                        "gpu": gpu_id,
                        "case_id": task.case_id,
                        **stats,
                    }
                )
                alignment_checked = True

            next_staged: Optional[StagedBatch] = None
            if not no_more_batches:
                try:
                    next_item, _ = base._get_queue_item(batch_q, blocking=False)
                    special = base._handle_special_queue_item(next_item)
                    if special:
                        no_more_batches = True
                    else:
                        if not isinstance(next_item, base.BatchPayload):
                            raise TypeError(f"Unexpected queue item: {type(next_item)}")
                        next_staged = _stage_batch(next_item, device, preprocess_stream)
                except queue.Empty:
                    pass

            step_start = time.perf_counter()
            x = _normalize_uint8(current.images_gpu, mean, std)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            ):
                output = model(x)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2 or output.shape[1] != base.FEATURE_DIM:
                raise RuntimeError(
                    f"Unexpected model output shape {tuple(output.shape)} for {task.case_id}; "
                    f"expected [B, {base.FEATURE_DIM}]"
                )

            features_cpu.append(output.detach().to(device="cpu", dtype=storage_dtype))
            gpu_step_s += time.perf_counter() - step_start
            progress_state["processed"] = float(batch_end)
            del x, output, current

            if next_staged is not None:
                current = next_staged
                continue
            if no_more_batches:
                current = None
                continue

            next_item, waited = base._get_queue_item(batch_q, blocking=True)
            gpu_wait_s += waited
            special = base._handle_special_queue_item(next_item)
            if special:
                current = None
                no_more_batches = True
            else:
                if not isinstance(next_item, base.BatchPayload):
                    raise TypeError(f"Unexpected queue item: {type(next_item)}")
                current = _stage_batch(next_item, device, preprocess_stream)

        producer.join()
        if expected_start != n_tiles:
            raise RuntimeError(
                f"Feature count mismatch for {task.case_id}: consumed {expected_start}, expected {n_tiles}"
            )

        features = torch.cat(features_cpu, dim=0)
        if len(features) != n_tiles:
            raise RuntimeError(f"Feature tensor mismatch for {task.case_id}: {len(features)} vs {n_tiles}")

        base.save_feature_file(task, features, coords_20x, coords_l0, cfg.feature_dtype)
        wall_s = time.perf_counter() - wall_start
        return {
            "tiles": float(n_tiles),
            "wall_s": wall_s,
            "tile_per_s": n_tiles / wall_s if wall_s > 0 else 0.0,
            "gpu_wait_s": gpu_wait_s,
            "producer_blocked_s": producer_stats["blocked_s"],
            "producer_decode_s": producer_stats["decode_s"],
            "gpu_step_s": gpu_step_s,
        }, alignment_checked

    except Exception:
        producer_stop.set()
        producer.join(timeout=10.0)
        raise
    finally:
        producer_stop.set()
        progress_stop.set()
        reporter.join(timeout=2.0)


def gpu_worker(
    gpu_id: int,
    task_q: Any,
    result_q: Any,
    checkpoint_path: str,
    cfg: WorkerConfig,
) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        model = base.load_tile_encoder(checkpoint_path, device, cfg.compile_model)
        result_q.put({"type": "worker_ready", "gpu": gpu_id})
        alignment_done = cfg.validate_preprocess <= 0

        while True:
            task = task_q.get()
            if task is None:
                break
            if not isinstance(task, base.WSITask):
                raise TypeError(f"Unexpected task type: {type(task)}")

            try:
                stats, checked = run_one_wsi(
                    task,
                    model,
                    device,
                    cfg,
                    gpu_id,
                    result_q,
                    validate_this_wsi=not alignment_done,
                )
                alignment_done = alignment_done or checked
                result_q.put(
                    {
                        "type": "done",
                        "gpu": gpu_id,
                        "case_id": task.case_id,
                        **stats,
                    }
                )
            except Exception as exc:
                tmp = Path(task.output_path).with_suffix(Path(task.output_path).suffix + ".tmp")
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                result_q.put(
                    {
                        "type": "task_error",
                        "gpu": gpu_id,
                        "case_id": task.case_id,
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    }
                )
                torch.cuda.empty_cache()

        result_q.put({"type": "worker_done", "gpu": gpu_id})

    except Exception as exc:
        result_q.put(
            {
                "type": "worker_error",
                "gpu": gpu_id,
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--coords-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpus", type=str, default="4,5,6,7")
    p.add_argument("--parser-threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--queue-batches", type=int, default=2, help="Raw 512px batches are large; default 2")
    p.add_argument("--progress-interval", type=float, default=5.0)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--validate-preprocess", type=int, default=8, help="Alignment samples on first WSI per GPU; 0 disables")
    p.add_argument("--alignment-min-cosine", type=float, default=0.999, help="Abort a WSI if mean feature cosine is below this")
    p.add_argument("--compile", action="store_true", dest="compile_model")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    return p.parse_args()


def validate_args(args: argparse.Namespace, gpu_ids: Iterable[int]) -> None:
    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)
    if not args.coords_dir.is_dir():
        raise NotADirectoryError(args.coords_dir)
    if args.parser_threads < 1:
        raise ValueError("--parser-threads must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.queue_batches < 1:
        raise ValueError("--queue-batches must be >= 1")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be > 0")
    if args.validate_preprocess < 0:
        raise ValueError("--validate-preprocess must be >= 0")
    if not 0.0 < args.alignment_min_cosine <= 1.0:
        raise ValueError("--alignment-min-cosine must be in (0, 1]")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    n_cuda = torch.cuda.device_count()
    bad = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= n_cuda]
    if bad:
        raise ValueError(f"Requested GPU ids {bad}, but torch sees cuda:0..{n_cuda - 1}")


def format_alignment(msg: dict[str, Any]) -> str:
    return (
        f"[GPU {msg['gpu']}] ALIGN {msg['case_id']} | "
        f"n={int(msg['samples'])} | pixel max={msg['max_pixel_diff']:.0f}, "
        f"mean={msg['mean_pixel_diff']:.4f}, exact={msg['exact_pixel_pct']:.2f}% | "
        f"feature cosine mean={msg['cosine_mean']:.8f}, min={msg['cosine_min']:.8f} | "
        f"feature max-abs={msg['feature_max_abs']:.6f}"
    )


def main() -> None:
    args = parse_args()
    gpu_ids = base.parse_gpu_ids(args.gpus)
    validate_args(args, gpu_ids)

    checkpoint = base.resolve_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, missing_wsi, existing_output = base.discover_tasks(
        args.data_dir,
        args.coords_dir,
        args.output_dir,
        args.overwrite,
        args.max_cases,
    )

    raw_batch_mib = args.batch_size * 3 * RAW_SIZE * RAW_SIZE / (1024 ** 2)
    print("GigaPath-Flash GPU-preprocess producer/consumer extraction", flush=True)
    print(f"  GPUs:                  {gpu_ids}", flush=True)
    print(f"  concurrent WSIs:       {len(gpu_ids)}", flush=True)
    print(f"  parser threads / WSI:  {args.parser_threads}", flush=True)
    print(f"  total parser threads:  {args.parser_threads * len(gpu_ids)}", flush=True)
    print(f"  batch size / GPU:      {args.batch_size}", flush=True)
    print(f"  raw batch size:        {raw_batch_mib:.0f} MiB pinned", flush=True)
    print(f"  queue depth / GPU:     {args.queue_batches} raw batches", flush=True)
    print(f"  preprocess:            GPU bicubic antialias + center crop", flush=True)
    print(f"  alignment samples:     {args.validate_preprocess} / GPU", flush=True)
    print(f"  alignment min cosine:  {args.alignment_min_cosine}", flush=True)
    print(f"  pending WSIs:          {len(tasks)}", flush=True)
    print(f"  existing outputs skip: {existing_output}", flush=True)
    print(f"  missing WSI skip:      {missing_wsi}", flush=True)
    print(f"  checkpoint:            {checkpoint}", flush=True)

    if not tasks:
        print("Nothing to process.", flush=True)
        return

    cfg = WorkerConfig(
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
            target=gpu_worker,
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
    finished = 0
    failed = 0
    total_tiles = 0
    start = time.perf_counter()
    worker_errors = 0

    while finished < total_tasks:
        try:
            msg = result_q.get(timeout=1.0)
        except queue.Empty:
            if all(not proc.is_alive() for proc in workers):
                break
            continue

        kind = msg.get("type")
        if kind == "worker_ready":
            print(f"[GPU {msg['gpu']}] model ready", flush=True)
        elif kind == "wsi_start":
            print(f"[GPU {msg['gpu']}] START {msg['case_id']} | {int(msg['total']):,} tiles", flush=True)
        elif kind == "progress":
            print(base.format_progress(msg), flush=True)
        elif kind == "alignment":
            print(format_alignment(msg), flush=True)
        elif kind == "warning":
            print(f"[GPU {msg['gpu']}] WARN {msg['case_id']}: {msg['message']}", file=sys.stderr, flush=True)
        elif kind == "done":
            finished += 1
            total_tiles += int(msg["tiles"])
            print(f"[{finished}/{total_tasks}] {base.format_done(msg)}", flush=True)
        elif kind == "task_error":
            finished += 1
            failed += 1
            print(
                f"[{finished}/{total_tasks}] [GPU {msg['gpu']}] ERROR {msg['case_id']}: {msg['error']}",
                file=sys.stderr,
                flush=True,
            )
            print(msg.get("trace", ""), file=sys.stderr, flush=True)
        elif kind == "worker_error":
            worker_errors += 1
            print(f"[GPU {msg['gpu']}] WORKER ERROR: {msg['error']}\n{msg['trace']}", file=sys.stderr, flush=True)

    for proc in workers:
        proc.join()

    elapsed = time.perf_counter() - start
    if finished < total_tasks:
        raise RuntimeError(
            f"All GPU workers exited with {total_tasks - finished} WSI tasks unreported. "
            f"Worker errors: {worker_errors}."
        )

    print("\nFinished", flush=True)
    print(f"  WSIs:       {finished - failed}/{total_tasks} succeeded", flush=True)
    print(f"  tiles:      {total_tiles:,}", flush=True)
    print(f"  wall time:  {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"  aggregate:  {total_tiles / elapsed:.1f} tile/s", flush=True)

    if failed or worker_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
