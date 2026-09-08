#!/usr/bin/env python3
"""Extract frozen GigaPath-Flash slide embeddings from existing tile features.

This script does NOT read WSIs and does NOT run the tile encoder again. It consumes
.pt files produced by extract_gigapath_flash_features.py:

    features: [N, 384]
    coords:   [N, 2]  # 20X pixel coordinates

and runs the official Prov-GigaPath Flash slide encoder
(gigapath_slide_enc12l384d, LongNet-12/384) with global pooling. One 384-d slide
embedding is written per case.

The official slide encoder package must be installed locally (or supplied with
--gigapath-source), and --checkpoint must point to a local slide_encoder.pth.
There is deliberately no Hugging Face auto-download.

Output files contain both:

    slide_embedding: [384]
    features:        [1, 384]

The second key is an intentional compatibility view: the existing
train_survival_mil.py can train "slide embedding + Cox" by using these output
files with --model mean. Since each bag has exactly one vector, its mean pooling
is the slide embedding itself.
"""

from __future__ import annotations

import argparse
import importlib
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

MODEL_REPO = "prov-gigapath/prov-gigapath-flash"
MODEL_ARCH = "gigapath_slide_enc12l384d"
FEATURE_DIM = 384
TILE_SIZE_20X = 256
DEFAULT_SLIDE_NGRIDS = 1000
DEFAULT_MAX_WSI_SIZE = 262144


@dataclass(frozen=True)
class SlideTask:
    case_id: str
    input_path: str
    output_path: str
    num_tiles: int


@dataclass(frozen=True)
class WorkerConfig:
    checkpoint: str
    gigapath_source: Optional[str]
    amp: str
    slide_ngrids: int
    max_wsi_size: int
    overwrite: bool


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_gpu_ids(text: str) -> list[int]:
    ids = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate GPU ids: {ids}")
    return ids


def _external_gigapath_root(explicit_source: Optional[str]) -> Path:
    """Return a root whose child is the *official* gigapath Python package.

    Running scripts from this repository puts owl/scripts first on sys.path, where
    a small local gigapath package exists for the tile encoder. The slide encoder
    requires the complete official Prov-GigaPath package, so we explicitly locate
    either a user-supplied source checkout or an installed `gigapath` distribution.
    """
    if explicit_source:
        root = Path(explicit_source).expanduser().resolve()
        if (root / "gigapath" / "slide_encoder.py").is_file():
            return root
        raise FileNotFoundError(
            f"--gigapath-source must be the Prov-GigaPath repository root; "
            f"missing {root / 'gigapath' / 'slide_encoder.py'}"
        )

    try:
        dist = distribution("gigapath")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Official Prov-GigaPath Python package is not installed. Install the "
            "official repository (no model weights are downloaded by this script), "
            "or pass --gigapath-source /path/to/prov-gigapath."
        ) from exc

    root = Path(dist.locate_file("")).resolve()
    if not (root / "gigapath" / "slide_encoder.py").is_file():
        raise RuntimeError(
            f"Installed distribution 'gigapath' was found at {root}, but "
            "gigapath/slide_encoder.py is missing."
        )
    return root


def import_official_slide_encoder(explicit_source: Optional[str]):
    root = _external_gigapath_root(explicit_source)

    # Ensure the complete official package wins over owl/scripts/gigapath.
    root_s = str(root)
    sys.path = [p for p in sys.path if p != root_s]
    sys.path.insert(0, root_s)
    for name in list(sys.modules):
        if name == "gigapath" or name.startswith("gigapath."):
            del sys.modules[name]

    try:
        mod = importlib.import_module("gigapath.slide_encoder")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import official gigapath.slide_encoder. The official "
            "LongNet implementation requires dependencies including fairscale and "
            "einops. See the run instructions for the one-time install command."
        ) from exc

    mod_path = Path(mod.__file__).resolve()
    if root not in mod_path.parents:
        raise RuntimeError(
            f"Imported the wrong gigapath package: {mod_path}; expected under {root}"
        )
    return mod, root


def _check_flash_attention_available() -> str:
    """Fail loudly rather than silently changing LongNet attention semantics."""
    try:
        flash_mod = importlib.import_module("torchscale.component.flash_attention")
    except Exception as exc:
        raise RuntimeError(
            "Could not import Prov-GigaPath TorchScale flash-attention wrapper."
        ) from exc

    if getattr(flash_mod, "flash_attn_func", None) is None:
        raise RuntimeError(
            "The official LongNet slide encoder requires a working flash-attention "
            "backend on this GPU, but none was found. Install a flash-attn build "
            "compatible with the current PyTorch/CUDA/RTX 5090 environment. The "
            "script intentionally does not substitute a different attention kernel."
        )

    fn = getattr(flash_mod.flash_attn_func, "__module__", "unknown")
    return str(fn)


def _normalize_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        state = obj
    else:
        raise ValueError(
            "slide checkpoint must be a state dict or a dict containing key 'model'"
        )

    # Accept DDP-saved checkpoints without weakening architecture validation.
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def load_slide_model(
    checkpoint: Path,
    device: torch.device,
    gigapath_source: Optional[str],
    slide_ngrids: int,
    max_wsi_size: int,
):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local slide encoder checkpoint not found: {checkpoint}")

    slide_encoder, source_root = import_official_slide_encoder(gigapath_source)
    flash_backend = _check_flash_attention_available()

    # Importing slide_encoder registers the official timm architecture.
    import timm

    model = timm.create_model(
        MODEL_ARCH,
        pretrained=False,
        in_chans=FEATURE_DIM,
        global_pool=True,
        slide_ngrids=slide_ngrids,
        max_wsi_size=max_wsi_size,
    )

    obj = _torch_load(checkpoint)
    state = _normalize_state_dict(obj)

    probe = state.get("patch_embed.proj.weight")
    if not isinstance(probe, torch.Tensor) or tuple(probe.shape) != (FEATURE_DIM, FEATURE_DIM):
        got = None if probe is None else tuple(probe.shape)
        raise RuntimeError(
            f"Checkpoint does not look like GigaPath-Flash slide encoder: "
            f"patch_embed.proj.weight={got}, expected {(FEATURE_DIM, FEATURE_DIM)}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Slide checkpoint/model mismatch. "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]}"
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return model, source_root, flash_backend


def inspect_tile_feature(path: Path, slide_ngrids: int) -> tuple[int, torch.Tensor, torch.Tensor, dict]:
    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dict")
    if "features" not in obj or "coords" not in obj:
        raise ValueError(f"{path}: required keys are 'features' and 'coords'")

    x = obj["features"]
    coords = obj["coords"]
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"{path}: features must be [N,{FEATURE_DIM}], got {getattr(x, 'shape', None)}")
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: coords must be [N,2], got {getattr(coords, 'shape', None)}")
    if len(x) != len(coords):
        raise ValueError(f"{path}: feature/coord count mismatch {len(x)} vs {len(coords)}")
    if len(x) == 0:
        raise ValueError(f"{path}: empty tile feature bag")
    if not torch.isfinite(x.float()).all():
        raise ValueError(f"{path}: tile features contain NaN/Inf")

    coords_i64 = coords.to(dtype=torch.int64, copy=True).contiguous()
    if (coords_i64 < 0).any():
        raise ValueError(f"{path}: negative 20X coordinate")
    if (coords_i64 % TILE_SIZE_20X != 0).any():
        bad = int((coords_i64 % TILE_SIZE_20X != 0).any(dim=1).sum().item())
        raise ValueError(
            f"{path}: {bad} coordinates are not aligned to {TILE_SIZE_20X}px 20X grid"
        )

    grid = torch.div(coords_i64, TILE_SIZE_20X, rounding_mode="floor")
    max_grid = int(grid.max().item())
    if max_grid >= slide_ngrids:
        raise ValueError(
            f"{path}: coordinate grid index {max_grid} exceeds slide_ngrids={slide_ngrids}. "
            "Do not silently change slide_ngrids unless the pretrained positional "
            "encoding convention is intentionally being changed."
        )

    meta = {
        "source_model": obj.get("model"),
        "tile_size_20x": obj.get("tile_size_20x"),
        "model_input_size": obj.get("model_input_size"),
    }
    return len(x), x.contiguous(), coords_i64, meta


def discover_tasks(input_dir: Path, output_dir: Path, overwrite: bool, max_cases: Optional[int]) -> tuple[list[SlideTask], int]:
    inputs = sorted(input_dir.glob("*.pt"))
    if not inputs:
        raise FileNotFoundError(f"No .pt tile-feature files in {input_dir}")

    tasks: list[SlideTask] = []
    skipped = 0
    for inp in inputs:
        out = output_dir / inp.name
        if out.is_file() and not overwrite:
            skipped += 1
            continue
        # Cheap count without retaining the tensor beyond discovery. This also catches
        # corrupt feature files before GPU workers are launched.
        obj = _torch_load(inp)
        if not isinstance(obj, dict) or not isinstance(obj.get("features"), torch.Tensor):
            raise ValueError(f"{inp}: invalid tile feature file")
        n = int(len(obj["features"]))
        tasks.append(SlideTask(inp.stem, str(inp), str(out), n))

    # Long slides first so the four workers finish closer together.
    tasks.sort(key=lambda t: t.num_tiles, reverse=True)
    if max_cases is not None:
        tasks = tasks[:max_cases]
    return tasks, skipped


def _amp_dtype(name: str) -> Optional[torch.dtype]:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "none":
        return None
    raise ValueError(name)


@torch.inference_mode()
def encode_one(
    model: torch.nn.Module,
    task: SlideTask,
    device: torch.device,
    amp: str,
    slide_ngrids: int,
) -> tuple[torch.Tensor, int, dict]:
    n, x_cpu, coords_cpu, meta = inspect_tile_feature(Path(task.input_path), slide_ngrids)

    # Float32 host -> device is intentional. Autocast controls the model compute dtype,
    # while avoiding a mixed fp16/bf16 input ambiguity from stored tile features.
    x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=False).unsqueeze(0)
    coords = coords_cpu.to(device=device, dtype=torch.float32, non_blocking=False).unsqueeze(0)

    dtype = _amp_dtype(amp)
    enabled = dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=enabled):
        outputs = model(x, coords, all_layer_embed=False)

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise RuntimeError(f"Unexpected slide encoder output type/length: {type(outputs)}, {len(outputs) if isinstance(outputs, (list, tuple)) else 'n/a'}")
    z = outputs[0]
    if z.ndim != 2 or tuple(z.shape) != (1, FEATURE_DIM):
        raise RuntimeError(f"Unexpected slide embedding shape {tuple(z.shape)}, expected (1,{FEATURE_DIM})")
    z = z[0].float().cpu().contiguous()
    if not torch.isfinite(z).all():
        raise RuntimeError(f"{task.case_id}: slide embedding contains NaN/Inf")
    return z, n, meta


def save_slide_feature(task: SlideTask, embedding: torch.Tensor, num_tiles: int, meta: dict, cfg: WorkerConfig) -> None:
    out = Path(task.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    payload = {
        "case_id": task.case_id,
        "model": MODEL_REPO,
        "slide_model_arch": MODEL_ARCH,
        "pooling": "global_mean_after_longnet",
        "slide_embedding": embedding.to(torch.float32).contiguous(),
        # Compatibility with train_survival_mil.py --model mean.
        "features": embedding.to(torch.float32).view(1, FEATURE_DIM).contiguous(),
        "num_tiles": int(num_tiles),
        "tile_feature_model": meta.get("source_model"),
        "tile_size_20x": meta.get("tile_size_20x", TILE_SIZE_20X),
        "source_tile_feature": task.input_path,
        "slide_ngrids": cfg.slide_ngrids,
        "max_wsi_size": cfg.max_wsi_size,
    }
    torch.save(payload, tmp)
    os.replace(tmp, out)


def gpu_worker(gpu_id: int, task_q: Any, result_q: Any, cfg: WorkerConfig) -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        model, source_root, flash_backend = load_slide_model(
            Path(cfg.checkpoint),
            device,
            cfg.gigapath_source,
            cfg.slide_ngrids,
            cfg.max_wsi_size,
        )
        result_q.put({
            "type": "ready",
            "gpu": gpu_id,
            "source_root": str(source_root),
            "flash_backend": flash_backend,
        })

        while True:
            task = task_q.get()
            if task is None:
                break
            started = time.perf_counter()
            try:
                z, n, meta = encode_one(model, task, device, cfg.amp, cfg.slide_ngrids)
                save_slide_feature(task, z, n, meta, cfg)
                result_q.put({
                    "type": "done",
                    "gpu": gpu_id,
                    "case_id": task.case_id,
                    "tiles": n,
                    "elapsed_s": time.perf_counter() - started,
                })
            except Exception as exc:
                tmp = Path(task.output_path).with_suffix(Path(task.output_path).suffix + ".tmp")
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                result_q.put({
                    "type": "error",
                    "gpu": gpu_id,
                    "case_id": task.case_id,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                })
                torch.cuda.empty_cache()

        result_q.put({"type": "worker_done", "gpu": gpu_id})
    except Exception as exc:
        result_q.put({
            "type": "worker_error",
            "gpu": gpu_id,
            "error": str(exc),
            "trace": traceback.format_exc(),
        })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Tile-feature .pt directory")
    p.add_argument("--output-dir", type=Path, required=True, help="Slide-feature .pt directory")
    p.add_argument("--checkpoint", type=Path, required=True, help="Local GigaPath-Flash slide_encoder.pth")
    p.add_argument(
        "--gigapath-source",
        type=Path,
        default=None,
        help="Optional local clone of official prov-gigapath repo. If omitted, use installed 'gigapath' distribution.",
    )
    p.add_argument("--gpus", type=str, default="4,5,6,7")
    p.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    p.add_argument("--slide-ngrids", type=int, default=DEFAULT_SLIDE_NGRIDS)
    p.add_argument("--max-wsi-size", type=int, default=DEFAULT_MAX_WSI_SIZE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    return p.parse_args()


def validate_args(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Local slide encoder checkpoint not found: {args.checkpoint}. "
            "This script never auto-downloads model weights."
        )
    if args.slide_ngrids < 1:
        raise ValueError("--slide-ngrids must be >= 1")
    if args.max_wsi_size < TILE_SIZE_20X:
        raise ValueError("--max-wsi-size is too small")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    n_cuda = torch.cuda.device_count()
    bad = [g for g in gpu_ids if g < 0 or g >= n_cuda]
    if bad:
        raise ValueError(f"Requested GPU ids {bad}, but torch sees cuda:0..{n_cuda-1}")


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)
    validate_args(args, gpu_ids)

    source = None if args.gigapath_source is None else str(args.gigapath_source.expanduser().resolve())
    checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks, skipped = discover_tasks(args.input_dir, args.output_dir, args.overwrite, args.max_cases)

    print("GigaPath-Flash slide embedding extraction", flush=True)
    print(f"  input tile features:    {args.input_dir}", flush=True)
    print(f"  output slide features:  {args.output_dir}", flush=True)
    print(f"  model arch:             {MODEL_ARCH}", flush=True)
    print(f"  pooling:                global (CLS is not used)", flush=True)
    print(f"  checkpoint:             {checkpoint}", flush=True)
    print(f"  GigaPath source:        {source or 'installed distribution'}", flush=True)
    print(f"  GPUs:                   {gpu_ids}", flush=True)
    print(f"  amp:                    {args.amp}", flush=True)
    print(f"  slide_ngrids:           {args.slide_ngrids}", flush=True)
    print(f"  max_wsi_size:           {args.max_wsi_size}", flush=True)
    print(f"  pending cases:          {len(tasks)}", flush=True)
    print(f"  existing outputs skip:  {skipped}", flush=True)

    if not tasks:
        print("Nothing to process.", flush=True)
        return

    cfg = WorkerConfig(
        checkpoint=str(checkpoint),
        gigapath_source=source,
        amp=args.amp,
        slide_ngrids=args.slide_ngrids,
        max_wsi_size=args.max_wsi_size,
        overwrite=args.overwrite,
    )

    ctx = torch.multiprocessing.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    workers = [
        ctx.Process(target=gpu_worker, args=(gpu, task_q, result_q, cfg), name=f"gigapath-slide-gpu-{gpu}")
        for gpu in gpu_ids
    ]
    for p in workers:
        p.start()
    for task in tasks:
        task_q.put(task)
    for _ in workers:
        task_q.put(None)

    total = len(tasks)
    done = 0
    failed = 0
    ready = 0
    tile_total = 0
    started = time.perf_counter()
    traces: list[str] = []

    while done + failed < total:
        try:
            msg = result_q.get(timeout=1.0)
        except queue.Empty:
            if all(not p.is_alive() for p in workers):
                break
            continue

        kind = msg.get("type")
        if kind == "ready":
            ready += 1
            print(
                f"GPU {msg['gpu']} ready | official={msg['source_root']} | flash={msg['flash_backend']}",
                flush=True,
            )
        elif kind == "done":
            done += 1
            tile_total += int(msg["tiles"])
            dt = float(msg["elapsed_s"])
            rate = int(msg["tiles"]) / dt if dt > 0 else 0.0
            print(
                f"[{done + failed}/{total}] GPU {msg['gpu']} {msg['case_id']} "
                f"| {int(msg['tiles']):,} tiles | {dt:.2f}s | {rate:.1f} tile/s",
                flush=True,
            )
        elif kind == "error":
            failed += 1
            print(
                f"[{done + failed}/{total}] GPU {msg['gpu']} ERROR {msg['case_id']}: {msg['error']}",
                file=sys.stderr,
                flush=True,
            )
            traces.append(str(msg.get("trace", "")))
        elif kind == "worker_error":
            print(f"GPU {msg['gpu']} WORKER ERROR: {msg['error']}", file=sys.stderr, flush=True)
            traces.append(str(msg.get("trace", "")))

    for p in workers:
        p.join()

    elapsed = time.perf_counter() - started
    print("\nFinished", flush=True)
    print(f"  succeeded: {done}/{total}", flush=True)
    print(f"  failed:    {failed}", flush=True)
    print(f"  tiles:     {tile_total:,}", flush=True)
    print(f"  wall time: {elapsed:.1f}s", flush=True)

    if done + failed < total:
        if traces:
            print("\n".join(t for t in traces if t), file=sys.stderr, flush=True)
        raise RuntimeError(f"Workers exited before reporting {total - done - failed} tasks")
    if failed:
        if traces:
            print("\n".join(t for t in traces if t), file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
