#!/usr/bin/env python3
"""5-fold survival baselines on pre-extracted WSI tile features.

Models
------
1. mean:  Mean pooling of tile embeddings -> linear Cox risk head.
2. abmil: Gated Attention MIL (Ilse-style) -> linear Cox risk head.

The script is tailored to the OWL / LEOPARD public training labels
(case_id, event, follow_up_years) and the .pt files produced by
extract_gigapath_flash_features.py.

Important Cox detail
--------------------
Cox partial likelihood needs a risk set containing multiple patients. The mean
baseline uses the entire training fold in each optimization step (exact fold-level
risk sets). ABMIL uses multi-patient Cox mini-batches; within each training WSI a
random subset of tiles is sampled each epoch. Validation always uses every tile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

FEATURE_DIM_DEFAULT = 384


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    feature_path: str
    event: int
    follow_up_years: float


@dataclass
class TrainConfig:
    model: str
    feature_dim: int
    folds: int
    seed: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    cox_batch_size: int
    train_tiles: int
    attention_dim: int
    dropout: float
    eval_chunk_tiles: int
    amp: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_feature_tensor(path: Path, expected_dim: Optional[int] = None) -> torch.Tensor:
    obj = _torch_load(path)
    if not isinstance(obj, dict) or "features" not in obj:
        raise ValueError(f"{path}: expected a dict containing key 'features'")
    x = obj["features"]
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{path}: features must be a 2D torch.Tensor")
    if len(x) == 0:
        raise ValueError(f"{path}: empty feature bag")
    if expected_dim is not None and x.shape[1] != expected_dim:
        raise ValueError(f"{path}: feature dim={x.shape[1]}, expected {expected_dim}")
    if not torch.isfinite(x.float()).all():
        raise ValueError(f"{path}: features contain NaN/Inf")
    return x.contiguous()


def _find_column(header: Sequence[str], aliases: Sequence[str]) -> str:
    normalized = {str(x).strip().lower(): str(x) for x in header}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    raise KeyError(f"Could not find any of columns {aliases}; got {list(header)}")


def _parse_event(value: str) -> int:
    s = str(value).strip().lower()
    if s in {"1", "1.0", "true", "yes", "event", "bcr", "occurred"}:
        return 1
    if s in {"0", "0.0", "false", "no", "censored", "censor", "none"}:
        return 0
    v = float(s)
    if v not in (0.0, 1.0):
        raise ValueError(f"event must be 0/1, got {value!r}")
    return int(v)


def load_labels(path: Path) -> list[tuple[str, int, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sample = path.read_text(encoding="utf-8-sig", errors="strict")[:8192]
    if not sample.strip():
        raise ValueError(f"Empty label file: {path}")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel

    rows: list[tuple[str, int, float]] = []
    seen: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header in {path}")
        case_col = _find_column(reader.fieldnames, ("case_id", "case", "sample_id"))
        event_col = _find_column(reader.fieldnames, ("event", "bcr_event", "status"))
        time_col = _find_column(reader.fieldnames, ("follow_up_years", "time_years", "time", "survival_years"))
        for lineno, row in enumerate(reader, start=2):
            raw_case = str(row.get(case_col, "")).strip()
            if not raw_case:
                continue
            event = _parse_event(row[event_col])
            t = float(row[time_col])
            if not math.isfinite(t) or t < 0:
                raise ValueError(f"{path}:{lineno}: invalid follow-up time {t}")
            if raw_case in seen:
                raise ValueError(f"Duplicate case_id in labels: {raw_case}")
            seen.add(raw_case)
            rows.append((raw_case, event, t))
    if not rows:
        raise ValueError(f"No label rows in {path}")
    return rows


def _case_aliases(case_id: str) -> list[str]:
    s = Path(str(case_id).strip()).stem
    if s.endswith("_tissue"):
        s = s[: -len("_tissue")]
    aliases = [s]
    if s.startswith("case_"):
        aliases.append(s[len("case_"):])
    else:
        aliases.append("case_" + s)
    return list(dict.fromkeys(aliases))


def build_records(labels: Sequence[tuple[str, int, float]], features_dir: Path) -> tuple[list[CaseRecord], list[str], list[str]]:
    feature_paths = sorted(features_dir.glob("*.pt"))
    if not feature_paths:
        raise FileNotFoundError(f"No .pt features found in {features_dir}")

    alias_to_path: dict[str, Path] = {}
    for path in feature_paths:
        for alias in _case_aliases(path.stem):
            if alias in alias_to_path and alias_to_path[alias] != path:
                raise ValueError(f"Feature alias collision: {alias}")
            alias_to_path[alias] = path

    records: list[CaseRecord] = []
    labels_without_features: list[str] = []
    used_paths: set[Path] = set()
    for raw_case, event, t in labels:
        path = None
        for alias in _case_aliases(raw_case):
            if alias in alias_to_path:
                path = alias_to_path[alias]
                break
        if path is None:
            labels_without_features.append(raw_case)
            continue
        used_paths.add(path)
        records.append(CaseRecord(case_id=path.stem, feature_path=str(path), event=int(event), follow_up_years=float(t)))

    features_without_labels = [p.stem for p in feature_paths if p not in used_paths]
    records.sort(key=lambda r: r.case_id)
    return records, labels_without_features, features_without_labels


def make_stratified_folds(records: Sequence[CaseRecord], n_splits: int, seed: int) -> list[list[int]]:
    if n_splits < 2:
        raise ValueError("--folds must be >= 2")
    groups = {0: [], 1: []}
    for i, r in enumerate(records):
        groups[r.event].append(i)
    if min(len(groups[0]), len(groups[1])) < n_splits:
        raise ValueError(
            f"Not enough cases per event class for {n_splits} folds: event=1:{len(groups[1])}, event=0:{len(groups[0])}"
        )

    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.asarray(groups[cls], dtype=np.int64)
        rng.shuffle(idx)
        for j, item in enumerate(idx.tolist()):
            folds[j % n_splits].append(int(item))
    for f in folds:
        f.sort()
    return folds


def save_splits(records: Sequence[CaseRecord], folds: Sequence[Sequence[int]], path: Path) -> None:
    fold_of: dict[int, int] = {}
    for fold, indices in enumerate(folds):
        for idx in indices:
            fold_of[int(idx)] = fold
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "fold", "event", "follow_up_years", "feature_path"])
        for i, r in enumerate(records):
            w.writerow([r.case_id, fold_of[i], r.event, r.follow_up_years, r.feature_path])


class FeatureStore:
    def __init__(self, records: Sequence[CaseRecord], cache_features: str, feature_dim: int):
        self.records = list(records)
        self.cache_features = cache_features
        self.feature_dim = feature_dim
        self.cache: dict[str, torch.Tensor] = {}

    def warm(self) -> None:
        if self.cache_features != "ram":
            return
        total_bytes = 0
        started = time.perf_counter()
        print("Caching tile features in RAM...", flush=True)
        for i, r in enumerate(self.records, start=1):
            x = load_feature_tensor(Path(r.feature_path), self.feature_dim)
            self.cache[r.case_id] = x
            total_bytes += x.numel() * x.element_size()
            if i % 25 == 0 or i == len(self.records):
                print(f"  [{i}/{len(self.records)}] {total_bytes / 1024**3:.2f} GiB cached", flush=True)
        print(
            f"RAM cache ready: {total_bytes / 1024**3:.2f} GiB in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def get(self, record: CaseRecord) -> torch.Tensor:
        if record.case_id in self.cache:
            return self.cache[record.case_id]
        return load_feature_tensor(Path(record.feature_path), self.feature_dim)


def cox_breslow_loss(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow handling of tied times."""
    risk = risk.reshape(-1).float()
    times = times.reshape(-1).float()
    events = events.reshape(-1).float()
    if risk.numel() != times.numel() or risk.numel() != events.numel():
        raise ValueError("risk/times/events length mismatch")
    nevents = events.sum()
    if float(nevents.detach().cpu()) <= 0:
        raise ValueError("Cox loss batch contains no observed events")

    order = torch.argsort(times, descending=True)
    r = risk[order]
    t = times[order]
    e = events[order]
    log_riskset = torch.logcumsumexp(r, dim=0)

    _, counts = torch.unique_consecutive(t, return_counts=True)
    start = 0
    loglik = r.new_zeros(())
    for count_t in counts.tolist():
        end = start + int(count_t)
        e_group = e[start:end]
        d = e_group.sum()
        if float(d.detach().cpu()) > 0:
            loglik = loglik + (r[start:end] * e_group).sum()
            loglik = loglik - d * log_riskset[end - 1]
        start = end
    return -loglik / nevents.clamp_min(1.0)


def harrell_c_index(times: Sequence[float], events: Sequence[int], risks: Sequence[float]) -> float:
    """Harrell C-index where larger risk means earlier recurrence."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=np.int64)
    r = np.asarray(risks, dtype=np.float64)
    concordant = 0.0
    comparable = 0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        js = np.flatnonzero(t > t[i])
        for j in js:
            comparable += 1
            if r[i] > r[j]:
                concordant += 1.0
            elif r[i] == r[j]:
                concordant += 0.5
    if comparable == 0:
        return float("nan")
    return concordant / comparable


class MeanCox(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.risk = nn.Linear(feature_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.risk(self.norm(pooled.float())).squeeze(-1)


class GatedABMILCox(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.feature_dim = feature_dim
        self.tile_norm = nn.LayerNorm(feature_dim)
        self.attn_v = nn.Linear(feature_dim, attention_dim)
        self.attn_u = nn.Linear(feature_dim, attention_dim)
        self.attn_w = nn.Linear(attention_dim, 1)
        self.slide_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.risk = nn.Linear(feature_dim, 1)

    def encode_tiles(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tile_norm(x.float())
        a = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))
        logits = self.attn_w(a).squeeze(-1)
        return h, logits

    def score_embedding(self, z: torch.Tensor) -> torch.Tensor:
        return self.risk(self.dropout(self.slide_norm(z.float()))).squeeze(-1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        h, logits = self.encode_tiles(x)
        attention = torch.softmax(logits.float(), dim=0).to(dtype=h.dtype)
        z = torch.sum(attention[:, None] * h, dim=0)
        risk = self.score_embedding(z)
        if return_attention:
            return risk, attention
        return risk


def _amp_context(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp != "none"
    if amp == "bf16":
        dtype = torch.bfloat16
    elif amp == "fp16":
        dtype = torch.float16
    elif amp == "none":
        dtype = torch.float32
    else:
        raise ValueError(amp)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def sample_tiles(x: torch.Tensor, max_tiles: int, rng: np.random.Generator) -> torch.Tensor:
    if max_tiles <= 0 or len(x) <= max_tiles:
        return x
    idx = rng.choice(len(x), size=max_tiles, replace=False)
    idx.sort()
    return x[torch.from_numpy(idx.astype(np.int64, copy=False))]


@torch.inference_mode()
def predict_abmil_slide(
    model: GatedABMILCox,
    features_cpu: torch.Tensor,
    device: torch.device,
    chunk_tiles: int,
    amp: str,
) -> float:
    """Exact full-bag attention via streaming softmax, without loading all tiles on GPU."""
    model.eval()
    running_max: Optional[torch.Tensor] = None
    denom = torch.zeros((), dtype=torch.float32, device=device)
    numer = torch.zeros(model.feature_dim, dtype=torch.float32, device=device)

    for start in range(0, len(features_cpu), chunk_tiles):
        chunk = features_cpu[start:start + chunk_tiles].to(device, non_blocking=True)
        with _amp_context(device, amp):
            h, logits = model.encode_tiles(chunk)
        h = h.float()
        logits = logits.float()
        chunk_max = logits.max()
        if running_max is None:
            new_max = chunk_max
            old_scale = torch.zeros((), dtype=torch.float32, device=device)
        else:
            new_max = torch.maximum(running_max, chunk_max)
            old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(logits - new_max)
        numer = numer * old_scale + torch.sum(weights[:, None] * h, dim=0)
        denom = denom * old_scale + weights.sum()
        running_max = new_max

    z = numer / denom.clamp_min(1e-12)
    with _amp_context(device, amp):
        risk = model.score_embedding(z)
    return float(risk.float().item())


def evaluate_mean(
    model: MeanCox,
    pooled: torch.Tensor,
    records: Sequence[CaseRecord],
    indices: Sequence[int],
    device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    with torch.inference_mode():
        risks = model(pooled[idx]).float().cpu().numpy().tolist()
    times = [records[i].follow_up_years for i in indices]
    events = [records[i].event for i in indices]
    return harrell_c_index(times, events, risks), [float(x) for x in risks]


def evaluate_abmil(
    model: GatedABMILCox,
    store: FeatureStore,
    records: Sequence[CaseRecord],
    indices: Sequence[int],
    device: torch.device,
    chunk_tiles: int,
    amp: str,
) -> tuple[float, list[float]]:
    risks: list[float] = []
    for i in indices:
        x = store.get(records[i])
        risks.append(predict_abmil_slide(model, x, device, chunk_tiles, amp))
    times = [records[i].follow_up_years for i in indices]
    events = [records[i].event for i in indices]
    return harrell_c_index(times, events, risks), risks


def _best_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def write_predictions(
    path: Path,
    records: Sequence[CaseRecord],
    indices: Sequence[int],
    risks: Sequence[float],
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "fold", "event", "follow_up_years", "risk"])
        for idx, risk in zip(indices, risks):
            r = records[idx]
            w.writerow([r.case_id, fold, r.event, r.follow_up_years, float(risk)])


def train_mean_fold(
    fold: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    pooled_cpu: torch.Tensor,
    records: Sequence[CaseRecord],
    output_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    model = MeanCox(cfg.feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pooled = pooled_cpu.to(device)
    tr = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    times = torch.tensor([records[i].follow_up_years for i in train_idx], dtype=torch.float32, device=device)
    events = torch.tensor([records[i].event for i in train_idx], dtype=torch.float32, device=device)

    best_c = -float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risks = model(pooled[tr])
        loss = cox_breslow_loss(risks, times, events)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        val_c, _ = evaluate_mean(model, pooled, records, val_idx, device)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_cindex": val_c})
        print(f"[mean][fold {fold}] epoch {epoch:03d} loss={loss.item():.5f} val_c={val_c:.4f}", flush=True)
        if math.isfinite(val_c) and val_c > best_c + 1e-6:
            best_c = val_c
            best_epoch = epoch
            best_state = _best_state_dict(model)
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold}: no valid validation C-index")
    model.load_state_dict(best_state)
    val_c, val_risks = evaluate_mean(model, pooled, records, val_idx, device)

    ckpt = {
        "model": "mean",
        "fold": fold,
        "feature_dim": cfg.feature_dim,
        "model_state": best_state,
        "best_epoch": best_epoch,
        "val_cindex": val_c,
        "config": asdict(cfg),
        "train_case_ids": [records[i].case_id for i in train_idx],
        "val_case_ids": [records[i].case_id for i in val_idx],
    }
    torch.save(ckpt, output_dir / f"fold_{fold}.pt")
    write_predictions(output_dir / f"fold_{fold}_val_predictions.csv", records, val_idx, val_risks, fold)
    (output_dir / f"fold_{fold}_history.json").write_text(json.dumps(history, indent=2))
    return {"fold": fold, "cindex": val_c, "best_epoch": best_epoch, "risks": val_risks}


def _epoch_batches(indices: Sequence[int], batch_size: int, rng: np.random.Generator) -> list[list[int]]:
    arr = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(arr)
    return [arr[i:i + batch_size].tolist() for i in range(0, len(arr), batch_size)]


def train_abmil_fold(
    fold: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    store: FeatureStore,
    records: Sequence[CaseRecord],
    output_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    model = GatedABMILCox(cfg.feature_dim, cfg.attention_dim, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_c = -float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng = np.random.default_rng(cfg.seed + fold * 100_000 + epoch)
        batches = _epoch_batches(train_idx, cfg.cox_batch_size, rng)
        epoch_losses: list[float] = []
        skipped_no_event = 0

        for batch in batches:
            batch_events = [records[i].event for i in batch]
            if sum(batch_events) == 0:
                skipped_no_event += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            risks: list[torch.Tensor] = []
            for i in batch:
                x_cpu = sample_tiles(store.get(records[i]), cfg.train_tiles, rng)
                x = x_cpu.to(device, non_blocking=True)
                with _amp_context(device, cfg.amp):
                    risk = model(x)
                risks.append(risk.float())
            risk_tensor = torch.stack(risks)
            times = torch.tensor([records[i].follow_up_years for i in batch], dtype=torch.float32, device=device)
            events = torch.tensor(batch_events, dtype=torch.float32, device=device)
            loss = cox_breslow_loss(risk_tensor, times, events)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        if not epoch_losses:
            raise RuntimeError(f"Fold {fold} epoch {epoch}: no Cox batches with events")

        val_c, _ = evaluate_abmil(model, store, records, val_idx, device, cfg.eval_chunk_tiles, cfg.amp)
        mean_loss = float(np.mean(epoch_losses))
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_cindex": val_c,
            "skipped_no_event_batches": skipped_no_event,
        })
        print(
            f"[abmil][fold {fold}] epoch {epoch:03d} loss={mean_loss:.5f} val_c={val_c:.4f} skipped_no_event={skipped_no_event}",
            flush=True,
        )

        if math.isfinite(val_c) and val_c > best_c + 1e-6:
            best_c = val_c
            best_epoch = epoch
            best_state = _best_state_dict(model)
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold}: no valid validation C-index")
    model.load_state_dict(best_state)
    val_c, val_risks = evaluate_abmil(model, store, records, val_idx, device, cfg.eval_chunk_tiles, cfg.amp)

    ckpt = {
        "model": "abmil",
        "fold": fold,
        "feature_dim": cfg.feature_dim,
        "attention_dim": cfg.attention_dim,
        "dropout": cfg.dropout,
        "model_state": best_state,
        "best_epoch": best_epoch,
        "val_cindex": val_c,
        "config": asdict(cfg),
        "train_case_ids": [records[i].case_id for i in train_idx],
        "val_case_ids": [records[i].case_id for i in val_idx],
    }
    torch.save(ckpt, output_dir / f"fold_{fold}.pt")
    write_predictions(output_dir / f"fold_{fold}_val_predictions.csv", records, val_idx, val_risks, fold)
    (output_dir / f"fold_{fold}_history.json").write_text(json.dumps(history, indent=2))
    return {"fold": fold, "cindex": val_c, "best_epoch": best_epoch, "risks": val_risks}


def prepare_mean_features(store: FeatureStore, records: Sequence[CaseRecord], feature_dim: int) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    print("Computing mean-pooled slide embeddings...", flush=True)
    for i, r in enumerate(records, start=1):
        x = store.get(r)
        if x.shape[1] != feature_dim:
            raise ValueError(f"{r.case_id}: feature dim {x.shape[1]} != {feature_dim}")
        pooled.append(x.float().mean(dim=0))
        if i % 50 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}]", flush=True)
    return torch.stack(pooled, dim=0)


def run_model(
    model_name: str,
    records: Sequence[CaseRecord],
    folds: Sequence[Sequence[int]],
    store: FeatureStore,
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    lr = (1e-3 if model_name == "mean" else 2e-4) if args.lr is None else args.lr
    cfg = TrainConfig(
        model=model_name,
        feature_dim=args.feature_dim,
        folds=args.folds,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        lr=lr,
        weight_decay=args.weight_decay,
        cox_batch_size=args.cox_batch_size,
        train_tiles=args.train_tiles,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        eval_chunk_tiles=args.eval_chunk_tiles,
        amp=args.amp,
    )

    out = output_root / model_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    pooled = None
    if model_name == "mean":
        pooled = prepare_mean_features(store, records, args.feature_dim).to(device)

    oof_risk = np.full(len(records), np.nan, dtype=np.float64)
    fold_metrics: list[dict] = []
    all_idx = set(range(len(records)))

    for fold, val_idx in enumerate(folds):
        val_idx = list(val_idx)
        train_idx = sorted(all_idx.difference(val_idx))
        ne_train = sum(records[i].event for i in train_idx)
        ne_val = sum(records[i].event for i in val_idx)
        print(
            f"\n=== {model_name.upper()} fold {fold}/{len(folds)-1} ===\n"
            f"train={len(train_idx)} (events={ne_train}) | val={len(val_idx)} (events={ne_val})",
            flush=True,
        )
        seed_everything(args.seed + fold)
        if model_name == "mean":
            assert pooled is not None
            result = train_mean_fold(fold, train_idx, val_idx, pooled, records, out, cfg, device)
        elif model_name == "abmil":
            result = train_abmil_fold(fold, train_idx, val_idx, store, records, out, cfg, device)
        else:
            raise ValueError(model_name)
        for idx, risk in zip(val_idx, result["risks"]):
            oof_risk[idx] = risk
        fold_metrics.append({
            "fold": fold,
            "cindex": result["cindex"],
            "best_epoch": result["best_epoch"],
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "events_train": ne_train,
            "events_val": ne_val,
        })

    if np.isnan(oof_risk).any():
        raise RuntimeError("OOF predictions are incomplete")
    times = [r.follow_up_years for r in records]
    events = [r.event for r in records]
    oof_c = harrell_c_index(times, events, oof_risk.tolist())
    fold_cs = [float(x["cindex"]) for x in fold_metrics]
    summary = {
        "model": model_name,
        "n_cases": len(records),
        "n_events": int(sum(events)),
        "fold_cindex": fold_cs,
        "mean_fold_cindex": float(np.mean(fold_cs)),
        "std_fold_cindex": float(np.std(fold_cs, ddof=1)) if len(fold_cs) > 1 else 0.0,
        "oof_cindex": float(oof_c),
        "folds": fold_metrics,
        "config": asdict(cfg),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_predictions(out / "oof_predictions.csv", records, list(range(len(records))), oof_risk.tolist(), -1)

    print(
        f"\n{model_name.upper()} RESULT: fold C-index={summary['mean_fold_cindex']:.4f} +/- {summary['std_fold_cindex']:.4f} | OOF C-index={summary['oof_cindex']:.4f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=("mean", "abmil", "both"), default="both")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--feature-dim", type=int, default=FEATURE_DIM_DEFAULT)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=None, help="Default: mean=1e-3, abmil=2e-4")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cox-batch-size", type=int, default=32, help="Patients per ABMIL Cox mini-batch")
    p.add_argument("--train-tiles", type=int, default=8192, help="Random tiles/WSI/epoch for ABMIL; <=0 uses all")
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--eval-chunk-tiles", type=int, default=32768)
    p.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    p.add_argument("--cache-features", choices=("none", "ram"), default="none")
    p.add_argument("--strict-match", action="store_true", help="Fail if labels/features are not one-to-one")
    p.add_argument("--check-only", action="store_true", help="Only validate label/feature matching")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.features_dir.is_dir():
        raise NotADirectoryError(args.features_dir)
    if not args.labels.is_file():
        raise FileNotFoundError(args.labels)
    if args.feature_dim < 1:
        raise ValueError("--feature-dim must be >= 1")
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    if args.cox_batch_size < 2:
        raise ValueError("--cox-batch-size must be >= 2")
    if args.eval_chunk_tiles < 1:
        raise ValueError("--eval-chunk-tiles must be >= 1")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0,1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    labels = load_labels(args.labels)
    records, labels_without_features, features_without_labels = build_records(labels, args.features_dir)
    if not records:
        raise RuntimeError("No labels matched any feature files")

    print("OWL / LEOPARD survival training", flush=True)
    print(f"  labels:                 {len(labels)}", flush=True)
    print(f"  feature files:          {len(list(args.features_dir.glob('*.pt')))}", flush=True)
    print(f"  matched cases:          {len(records)}", flush=True)
    print(f"  matched events:         {sum(r.event for r in records)}", flush=True)
    print(f"  labels without feature: {len(labels_without_features)}", flush=True)
    if labels_without_features:
        print("    " + ", ".join(labels_without_features[:20]), flush=True)
    print(f"  features without label: {len(features_without_labels)}", flush=True)
    if features_without_labels:
        print("    " + ", ".join(features_without_labels[:20]), flush=True)

    if args.strict_match and (labels_without_features or features_without_labels):
        raise RuntimeError("Label/feature matching is not one-to-one (--strict-match)")

    first_x = load_feature_tensor(Path(records[0].feature_path), args.feature_dim)
    print(f"  first feature bag:      {records[0].case_id} {tuple(first_x.shape)} {first_x.dtype}", flush=True)

    folds = make_stratified_folds(records, args.folds, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(records, folds, args.output_dir / "splits.csv")
    for k, idx in enumerate(folds):
        print(f"  fold {k}:                {len(idx)} cases, events={sum(records[i].event for i in idx)}", flush=True)

    if args.check_only:
        print("Check complete; no training started.", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"  device:                 {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"  device:                 {device}", flush=True)

    store = FeatureStore(records, args.cache_features, args.feature_dim)
    store.warm()

    models = ("mean", "abmil") if args.model == "both" else (args.model,)
    summaries = []
    started = time.perf_counter()
    for name in models:
        summaries.append(run_model(name, records, folds, store, args.output_dir, args, device))
    comparison = {"elapsed_s": time.perf_counter() - started, "models": summaries}
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== Comparison ===", flush=True)
    for s in summaries:
        print(f"{s['model']:>5}: mean-fold={s['mean_fold_cindex']:.4f} std={s['std_fold_cindex']:.4f} oof={s['oof_cindex']:.4f}", flush=True)


if __name__ == "__main__":
    main()
