#!/usr/bin/env python3
"""High-throughput multi-GPU GigaPath-Flash feature extraction.

Pipeline
--------
- One long-lived process per GPU; each GPU owns one WSI until completion.
- CPU threads only decode 512x512 level-0 patches with OpenSlide.
- GPU performs bicubic 512->256 resize, 224 center crop, normalization, and inference.
- CPU/GPU handoff uses a bounded pinned-memory queue with backpressure.
- First non-empty WSI on each GPU validates GPU preprocessing against the prior
  PIL CPU preprocessing path; the WSI aborts if feature cosine is too low.
- Existing .pt outputs are skipped unless --overwrite is supplied.
- Progress is rendered in-place as a fixed terminal dashboard when stdout is a TTY.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_REPO = "prov-gigapath/prov-gigapath-flash"
FEATURE_DIM = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LEVEL0_READ_SIZE = 512
RESIZE_SIZE = 256
MODEL_INPUT_SIZE = 224
CROP_OFFSET = (RESIZE_SIZE - MODEL_INPUT_SIZE) // 2

_END = object()
_READER_TLS = threading.local()
_READER_REGISTRY: list[Any] = []
_READER_REGISTRY_LOCK = threading.Lock()


@dataclass(frozen=True)
class WSITask:
    case_id: str
    slide_path: str
    coord_path: str
    output_path: str
    num_tiles: int


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
class BatchPayload:
    images: torch.Tensor
    start: int
    end: int


@dataclass
class ProducerFailure:
    message: str
    trace: str


@dataclass
class StagedBatch:
    payload: BatchPayload
    images_gpu: torch.Tensor
    ready_event: torch.cuda.Event


def parse_gpu_ids(text: str) -> list[int]:
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate GPU ids in --gpus: {ids}")
    return ids


def load_summary_counts(coords_dir: Path) -> dict[str, int]:
    path = coords_dir / "summary.csv"
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["case_id"]] = int(row["num_tiles"])
            except (KeyError, TypeError, ValueError):
                pass
    return out


def count_coord_rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def discover_tasks(
    data_dir: Path,
    coords_dir: Path,
    output_dir: Path,
    overwrite: bool,
    max_cases: Optional[int],
) -> tuple[list[WSITask], int, int]:
    counts = load_summary_counts(coords_dir)
    tasks: list[WSITask] = []
    missing_wsi = 0
    existing_output = 0

    for coord_path in sorted(coords_dir.glob("*.csv")):
        if coord_path.name == "summary.csv":
            continue
        case_id = coord_path.stem
        slide_path = data_dir / f"{case_id}.tif"
        output_path = output_dir / f"{case_id}.pt"

        if not slide_path.is_file():
            missing_wsi += 1
            continue
        if output_path.is_file() and not overwrite:
            existing_output += 1
            continue

        num_tiles = counts.get(case_id)
        if num_tiles is None:
            num_tiles = count_coord_rows(coord_path)

        tasks.append(
            WSITask(
                case_id=case_id,
                slide_path=str(slide_path),
                coord_path=str(coord_path),
                output_path=str(output_path),
                num_tiles=num_tiles,
            )
        )

    tasks.sort(key=lambda x: x.num_tiles, reverse=True)
    if max_cases is not None:
        tasks = tasks[:max_cases]
    return tasks, missing_wsi, existing_output


def load_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty coordinate CSV: {path}") from exc

        col = {name: i for i, name in enumerate(header)}
        required = ("x_l0", "y_l0", "x_20x", "y_20x")
        missing = [x for x in required if x not in col]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        rows = [
            (
                int(row[col["x_l0"]]),
                int(row[col["y_l0"]]),
                int(row[col["x_20x"]]),
                int(row[col["y_20x"]]),
            )
            for row in reader
            if row
        ]

    if not rows:
        e = np.empty((0, 2), dtype=np.int64)
        return e, e.copy()

    arr = np.asarray(rows, dtype=np.int64)
    return arr[:, :2], arr[:, 2:4]


def _init_slide_reader(slide_path: str) -> None:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "openslide-python and the system OpenSlide library are required."
        ) from exc

    reader = openslide.OpenSlide(slide_path)
    _READER_TLS.reader = reader
    with _READER_REGISTRY_LOCK:
        _READER_REGISTRY.append(reader)


def _close_registered_readers() -> None:
    global _READER_REGISTRY
    with _READER_REGISTRY_LOCK:
        readers = _READER_REGISTRY
        _READER_REGISTRY = []
    for reader in readers:
        try:
            reader.close()
        except Exception:
            pass


def _read_raw_patch_into(dest: torch.Tensor, x_l0: int, y_l0: int) -> None:
    reader = getattr(_READER_TLS, "reader", None)
    if reader is None:
        raise RuntimeError("OpenSlide reader was not initialized in parser thread")

    rgba = reader.read_region(
        (int(x_l0), int(y_l0)),
        0,
        (LEVEL0_READ_SIZE, LEVEL0_READ_SIZE),
    )

    if rgba.mode == "RGBA":
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = rgba.convert("RGB")

    arr = np.array(rgb, dtype=np.uint8, copy=True)
    dest.copy_(torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))))


def _put_with_backpressure(
    out_q: queue.Queue,
    item: Any,
    stop_event: threading.Event,
    stats: dict[str, float],
) -> bool:
    t0 = time.perf_counter()
    while not stop_event.is_set():
        try:
            out_q.put(item, timeout=0.25)
            stats["blocked_s"] += time.perf_counter() - t0
            return True
        except queue.Full:
            pass
    stats["blocked_s"] += time.perf_counter() - t0
    return False


def _produce_raw_batches(
    slide_path: str,
    coords_l0: np.ndarray,
    batch_size: int,
    parser_threads: int,
    out_q: queue.Queue,
    stop_event: threading.Event,
    stats: dict[str, float],
) -> None:
    try:
        _close_registered_readers()
        with ThreadPoolExecutor(
            max_workers=parser_threads,
            thread_name_prefix="patch-reader",
            initializer=_init_slide_reader,
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
                    (size, 3, LEVEL0_READ_SIZE, LEVEL0_READ_SIZE),
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
                for fut in futures:
                    fut.result()

                stats["decode_s"] += time.perf_counter() - t0
                stats["tiles"] += float(size)

                if not _put_with_backpressure(
                    out_q,
                    BatchPayload(images, start, end),
                    stop_event,
                    stats,
                ):
                    return

        if not stop_event.is_set():
            _put_with_backpressure(out_q, _END, stop_event, stats)

    except Exception as exc:
        if not stop_event.is_set():
            _put_with_backpressure(
                out_q,
                ProducerFailure(str(exc), traceback.format_exc()),
                stop_event,
                stats,
            )
    finally:
        _close_registered_readers()


def _get_queue_item(batch_q: queue.Queue, blocking: bool) -> tuple[Any, float]:
    t0 = time.perf_counter()
    item = batch_q.get() if blocking else batch_q.get_nowait()
    return item, time.perf_counter() - t0


def _handle_special_queue_item(item: Any) -> Optional[bool]:
    if item is _END:
        return True
    if isinstance(item, ProducerFailure):
        raise RuntimeError(f"CPU producer failed: {item.message}\n{item.trace}")
    return None


def _gpu_resize_crop_uint8(raw_gpu: torch.Tensor) -> torch.Tensor:
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
        CROP_OFFSET:CROP_OFFSET + MODEL_INPUT_SIZE,
        CROP_OFFSET:CROP_OFFSET + MODEL_INPUT_SIZE,
    ].contiguous()


def _stage_batch(
    payload: BatchPayload,
    device: torch.device,
    preprocess_stream: torch.cuda.Stream,
) -> StagedBatch:
    with torch.cuda.stream(preprocess_stream):
        raw_gpu = payload.images.to(device=device, non_blocking=True)
        images_gpu = _gpu_resize_crop_uint8(raw_gpu)
        ready = torch.cuda.Event(blocking=False)
        ready.record(preprocess_stream)
    return StagedBatch(payload, images_gpu, ready)


def _cpu_reference_preprocess(raw_cpu: torch.Tensor, n: int) -> torch.Tensor:
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
        refs.append(
            torch.from_numpy(np.ascontiguousarray(arr224.transpose(2, 0, 1)))
        )
    return torch.stack(refs, dim=0)


def _storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


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
    if cosine_mean < min_cosine:
        raise RuntimeError(
            "GPU preprocessing alignment failed: "
            f"mean cosine={cosine_mean:.8f} < required {min_cosine:.8f}"
        )

    return {
        "samples": float(n),
        "max_pixel_diff": float(diff.max().item()),
        "mean_pixel_diff": float(diff.float().mean().item()),
        "exact_pixel_pct": float((diff == 0).float().mean().item() * 100.0),
        "cosine_mean": cosine_mean,
        "cosine_min": float(cosine.min().item()),
        "feature_max_abs": float(
            (ref_feat.float() - new_feat.float()).abs().max().item()
        ),
    }


def save_feature_file(
    task: WSITask,
    features: torch.Tensor,
    coords_20x: np.ndarray,
    coords_l0: np.ndarray,
    feature_dtype: str,
) -> None:
    output = Path(task.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "case_id": task.case_id,
        "model": MODEL_REPO,
        "features": features.to(dtype=_storage_dtype(feature_dtype)).contiguous(),
        "coords": torch.from_numpy(
            coords_20x.astype(np.int32, copy=False)
        ).contiguous(),
        "coords_l0": torch.from_numpy(
            coords_l0.astype(np.int32, copy=False)
        ).contiguous(),
        "tile_size_20x": RESIZE_SIZE,
        "model_input_size": MODEL_INPUT_SIZE,
        "level0_read_size": LEVEL0_READ_SIZE,
    }
    torch.save(payload, tmp)
    os.replace(tmp, output)


def _progress_reporter(
    gpu_id: int,
    case_id: str,
    total_tiles: int,
    batch_q: queue.Queue,
    producer_stats: dict[str, float],
    progress_state: dict[str, float],
    result_q: Any,
    interval: float,
    stop_event: threading.Event,
    queue_max: int,
) -> None:
    start = time.perf_counter()
    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - start
        processed = int(progress_state["processed"])
        produced = min(int(producer_stats["tiles"]), total_tiles)
        result_q.put(
            {
                "type": "progress",
                "gpu": gpu_id,
                "case_id": case_id,
                "processed": processed,
                "produced": produced,
                "total": total_tiles,
                "rate": processed / elapsed if elapsed > 0 else 0.0,
                "queue": batch_q.qsize(),
                "queue_max": queue_max,
            }
        )


def load_tile_encoder(
    checkpoint_path: str,
    device: torch.device,
    compile_model: bool,
) -> torch.nn.Module:
    try:
        import gigapath.tile_encoder as tile_encoder
    except ImportError as exc:
        raise RuntimeError(
            "Could not import scripts/gigapath/tile_encoder.py. "
            "Ensure it exists and timm>=1.0.3 is installed."
        ) from exc

    model = tile_encoder.create_model(checkpoint_path)
    model.eval().to(device)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model


def run_one_wsi(
    task: WSITask,
    model: torch.nn.Module,
    device: torch.device,
    cfg: WorkerConfig,
    gpu_id: int,
    result_q: Any,
    validate_this_wsi: bool,
) -> tuple[dict[str, float], bool]:
    coords_l0, coords_20x = load_coords(Path(task.coord_path))
    n_tiles = len(coords_l0)

    if task.num_tiles != n_tiles:
        result_q.put(
            {
                "type": "warning",
                "gpu": gpu_id,
                "case_id": task.case_id,
                "message": (
                    f"summary says {task.num_tiles}, coordinate CSV has {n_tiles}"
                ),
            }
        )

    if n_tiles == 0:
        empty = torch.empty((0, FEATURE_DIM), dtype=_storage_dtype(cfg.feature_dtype))
        save_feature_file(task, empty, coords_20x, coords_l0, cfg.feature_dtype)
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
        daemon=True,
    )
    reporter = threading.Thread(
        target=_progress_reporter,
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
        daemon=True,
    )

    preprocess_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    mean = torch.tensor(
        IMAGENET_MEAN, dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD, dtype=torch.float32, device=device
    ).view(1, 3, 1, 1)
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    storage_dtype = _storage_dtype(cfg.feature_dtype)

    features_cpu: list[torch.Tensor] = []
    gpu_wait_s = 0.0
    gpu_step_s = 0.0
    alignment_checked = False
    wall_start = time.perf_counter()

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
        first, waited = _get_queue_item(batch_q, True)
        gpu_wait_s += waited
        if _handle_special_queue_item(first):
            raise RuntimeError(
                f"Producer ended before yielding patches for {task.case_id}"
            )
        if not isinstance(first, BatchPayload):
            raise TypeError(f"Unexpected queue item: {type(first)}")

        current = _stage_batch(first, device, preprocess_stream)
        expected_start = 0
        no_more_batches = False

        while current is not None:
            current_stream.wait_event(current.ready_event)

            if current.payload.start != expected_start:
                raise RuntimeError(
                    f"Batch order error for {task.case_id}: "
                    f"expected {expected_start}, got {current.payload.start}"
                )

            batch_end = current.payload.end
            expected_start = batch_end

            if (
                validate_this_wsi
                and not alignment_checked
                and cfg.validate_preprocess > 0
            ):
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
                    nxt, _ = _get_queue_item(batch_q, False)
                    special = _handle_special_queue_item(nxt)
                    if special:
                        no_more_batches = True
                    else:
                        if not isinstance(nxt, BatchPayload):
                            raise TypeError(
                                f"Unexpected queue item: {type(nxt)}"
                            )
                        next_staged = _stage_batch(
                            nxt, device, preprocess_stream
                        )
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
            if output.ndim != 2 or output.shape[1] != FEATURE_DIM:
                raise RuntimeError(
                    f"Unexpected output shape {tuple(output.shape)} "
                    f"for {task.case_id}; expected [B,{FEATURE_DIM}]"
                )

            features_cpu.append(
                output.detach().to(device="cpu", dtype=storage_dtype)
            )
            gpu_step_s += time.perf_counter() - step_start
            progress_state["processed"] = float(batch_end)
            del x, output, current

            if next_staged is not None:
                current = next_staged
                continue
            if no_more_batches:
                current = None
                continue

            nxt, waited = _get_queue_item(batch_q, True)
            gpu_wait_s += waited
            special = _handle_special_queue_item(nxt)
            if special:
                current = None
                no_more_batches = True
            else:
                if not isinstance(nxt, BatchPayload):
                    raise TypeError(f"Unexpected queue item: {type(nxt)}")
                current = _stage_batch(nxt, device, preprocess_stream)

        producer.join()

        if expected_start != n_tiles:
            raise RuntimeError(
                f"Feature count mismatch for {task.case_id}: "
                f"consumed {expected_start}, expected {n_tiles}"
            )

        features = torch.cat(features_cpu, dim=0)
        if len(features) != n_tiles:
            raise RuntimeError(
                f"Feature tensor mismatch for {task.case_id}: "
                f"{len(features)} vs {n_tiles}"
            )

        save_feature_file(task, features, coords_20x, coords_l0, cfg.feature_dtype)
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

        model = load_tile_encoder(checkpoint_path, device, cfg.compile_model)
        result_q.put({"type": "worker_ready", "gpu": gpu_id})
        alignment_done = cfg.validate_preprocess <= 0

        while True:
            task = task_q.get()
            if task is None:
                break
            if not isinstance(task, WSITask):
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
                tmp = Path(task.output_path).with_suffix(
                    Path(task.output_path).suffix + ".tmp"
                )
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


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Local GigaPath-Flash checkpoint not found: {path}"
        )
    return path


def format_done(msg: dict[str, Any]) -> str:
    wall = float(msg["wall_s"])
    wait = float(msg["gpu_wait_s"])
    blocked = float(msg["producer_blocked_s"])
    wait_pct = 100.0 * wait / wall if wall > 0 else 0.0
    block_pct = 100.0 * blocked / wall if wall > 0 else 0.0
    return (
        f"GPU {msg['gpu']} {msg['case_id']}: {int(msg['tiles']):,} tiles"
        f" | {float(msg['tile_per_s']):.1f} tile/s"
        f" | {wall:.1f}s"
        f" | GPU-wait={wait:.2f}s ({wait_pct:.1f}%)"
        f" | producer-blocked={blocked:.2f}s ({block_pct:.1f}%)"
    )


def format_alignment(msg: dict[str, Any]) -> str:
    return (
        f"GPU {msg['gpu']} ALIGN {msg['case_id']}"
        f" | n={int(msg['samples'])}"
        f" | pixel max={msg['max_pixel_diff']:.0f}"
        f", mean={msg['mean_pixel_diff']:.4f}"
        f", exact={msg['exact_pixel_pct']:.2f}%"
        f" | feature cosine mean={msg['cosine_mean']:.8f}"
        f", min={msg['cosine_min']:.8f}"
    )


class TerminalDashboard:
    def __init__(
        self,
        gpu_ids: list[int],
        total_tasks: int,
        total_task_tiles: int,
    ) -> None:
        self.gpu_ids = list(gpu_ids)
        self.total_tasks = total_tasks
        self.total_task_tiles = total_task_tiles
        self.finished = 0
        self.failed = 0
        self.completed_tiles = 0
        self.start = time.perf_counter()
        self.gpu_lines = {
            gpu: f"GPU {gpu}: loading model..." for gpu in self.gpu_ids
        }
        self.event = "event: -"
        self.enabled = (
            sys.stdout.isatty()
            and os.environ.get("TERM", "") != "dumb"
        )
        self._initialized = False
        self._closed = False
        self._nlines = 2 + len(self.gpu_ids)

    def _fit(self, text: str) -> str:
        width = shutil.get_terminal_size(fallback=(180, 40)).columns
        if width > 8 and len(text) >= width:
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
        return [self._fit(x) for x in lines]

    def render(self) -> None:
        if self._closed or not self.enabled:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--coords-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpus", type=str, default="4,5,6,7")
    p.add_argument("--parser-threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--queue-batches",
        type=int,
        default=2,
        help="Bounded raw 512px batches per GPU (default: 2)",
    )
    p.add_argument("--progress-interval", type=float, default=2.0)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    p.add_argument(
        "--validate-preprocess",
        type=int,
        default=8,
        help="Alignment samples on first WSI per GPU; 0 disables",
    )
    p.add_argument(
        "--alignment-min-cosine",
        type=float,
        default=0.999,
    )
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
        raise ValueError("--alignment-min-cosine must be in (0,1]")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    n_cuda = torch.cuda.device_count()
    bad = [x for x in gpu_ids if x < 0 or x >= n_cuda]
    if bad:
        raise ValueError(
            f"Requested GPU ids {bad}, but torch sees cuda:0..{n_cuda - 1}"
        )


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)
    validate_args(args, gpu_ids)
    checkpoint = resolve_checkpoint(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, missing_wsi, existing_output = discover_tasks(
        args.data_dir,
        args.coords_dir,
        args.output_dir,
        args.overwrite,
        args.max_cases,
    )

    raw_batch_mib = (
        args.batch_size * 3 * LEVEL0_READ_SIZE * LEVEL0_READ_SIZE / 1024**2
    )

    print("GigaPath-Flash multi-GPU extraction", flush=True)
    print(f"  GPUs:                  {gpu_ids}", flush=True)
    print(f"  concurrent WSIs:       {len(gpu_ids)}", flush=True)
    print(f"  parser threads / WSI:  {args.parser_threads}", flush=True)
    print(
        f"  total parser threads:  {args.parser_threads * len(gpu_ids)}",
        flush=True,
    )
    print(f"  batch size / GPU:      {args.batch_size}", flush=True)
    print(f"  raw batch size:        {raw_batch_mib:.0f} MiB pinned", flush=True)
    print(
        f"  queue depth / GPU:     {args.queue_batches} raw batches",
        flush=True,
    )
    print("  preprocess:            GPU bicubic + antialias + crop", flush=True)
    print(
        f"  progress display:      in-place dashboard ({args.progress_interval:g}s)",
        flush=True,
    )
    print(
        f"  alignment samples:     {args.validate_preprocess} / GPU",
        flush=True,
    )
    print(
        f"  alignment min cosine:  {args.alignment_min_cosine}",
        flush=True,
    )
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
            args=(gpu, task_q, result_q, str(checkpoint), cfg),
            name=f"gigapath-gpu-{gpu}",
        )
        for gpu in gpu_ids
    ]
    for proc in workers:
        proc.start()
    for task in tasks:
        task_q.put(task)
    for _ in workers:
        task_q.put(None)

    total_tasks = len(tasks)
    total_task_tiles = sum(x.num_tiles for x in tasks)
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
                if all(not p.is_alive() for p in workers):
                    break
                continue

            kind = msg.get("type")
            gpu = int(msg.get("gpu", -1))

            if kind == "worker_ready":
                text = f"GPU {gpu}: model ready | waiting for WSI"
                dashboard.set_gpu(gpu, text)
                dashboard.plain(text)

            elif kind == "wsi_start":
                text = (
                    f"GPU {gpu}  {msg['case_id']}"
                    f" | 0/{int(msg['total']):,} (  0.0%) | starting"
                )
                dashboard.set_gpu(gpu, text)
                dashboard.plain(text)

            elif kind == "progress":
                dashboard.progress(msg)
                if not dashboard.enabled:
                    processed = int(msg["processed"])
                    total = int(msg["total"])
                    pct = 100.0 * processed / total if total else 100.0
                    dashboard.plain(
                        f"GPU {gpu} {msg['case_id']}"
                        f" | {processed:,}/{total:,} ({pct:.1f}%)"
                        f" | {float(msg['rate']):.1f} tile/s"
                        f" | CPU={int(msg['produced']):,}/{total:,}"
                        f" | queue={int(msg['queue'])}/{int(msg['queue_max'])}"
                    )

            elif kind == "alignment":
                text = format_alignment(msg)
                dashboard.set_event(text)
                dashboard.plain(text)

            elif kind == "warning":
                text = f"GPU {gpu} WARN {msg['case_id']}: {msg['message']}"
                dashboard.set_event(text)
                dashboard.plain(text)

            elif kind == "done":
                dashboard.finished += 1
                dashboard.completed_tiles += int(msg["tiles"])
                done_text = format_done(msg)
                dashboard.set_gpu(
                    gpu,
                    f"GPU {gpu}: DONE {msg['case_id']}"
                    f" | {float(msg['tile_per_s']):.1f} tile/s"
                    f" | waiting next WSI",
                )
                dashboard.set_event(
                    f"[{dashboard.finished}/{total_tasks}] {done_text}"
                )
                dashboard.plain(
                    f"[{dashboard.finished}/{total_tasks}] {done_text}"
                )

            elif kind == "task_error":
                dashboard.finished += 1
                dashboard.failed += 1
                text = (
                    f"GPU {gpu} ERROR {msg['case_id']}: {msg['error']}"
                )
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
            print(
                "\n".join(x for x in error_details if x),
                file=sys.stderr,
                flush=True,
            )
        raise RuntimeError(
            f"All GPU workers exited with "
            f"{total_tasks - dashboard.finished} WSI tasks unreported. "
            f"Worker errors: {worker_errors}."
        )

    print("\nFinished", flush=True)
    print(
        f"  WSIs:       "
        f"{dashboard.finished - dashboard.failed}/{total_tasks} succeeded",
        flush=True,
    )
    print(f"  tiles:      {dashboard.completed_tiles:,}", flush=True)
    print(f"  wall time:  {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(
            f"  aggregate:  "
            f"{dashboard.completed_tiles / elapsed:.1f} tile/s",
            flush=True,
        )

    if dashboard.failed or worker_errors:
        if error_details:
            print(
                "\n".join(x for x in error_details if x),
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
