#!/usr/bin/env python3
"""Generate Prov-GigaPath compatible patch coordinates from OWL tissue masks.

Dataset assumptions (OWL / LEOPARD):
- case_xxx.tif: WSI, level-0 is 40x.
- case_xxx_tissue.tif: tissue mask whose level-0 is 16x downsampled
  relative to the WSI level-0, i.e. approximately 2.5x.

Prov-GigaPath official tiling is 256x256 at 20x (approximately 0.5 MPP),
with non-overlapping tiles and tissue occupancy > 0.1.
For this dataset that maps to:
- WSI level-0 (40x): patch = stride = 512 px.
- Tissue mask (2.5x): window = stride = 32 px.

The script only reads the low-resolution tissue masks. It writes one CSV per
slide containing both raw WSI level-0 coordinates (for image reading) and
20x GigaPath coordinates (for the GigaPath slide encoder).
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


@dataclass(frozen=True)
class TilingConfig:
    level0_magnification: float = 40.0
    target_magnification: float = 20.0
    target_tile_size: int = 256
    mask_downsample: int = 16
    occupancy_threshold: float = 0.1
    mask_threshold: float = 0.0

    @property
    def level0_tile_size(self) -> int:
        value = self.target_tile_size * self.level0_magnification / self.target_magnification
        rounded = int(round(value))
        if not np.isclose(value, rounded):
            raise ValueError(f"level-0 tile size is not an integer: {value}")
        return rounded

    @property
    def mask_tile_size(self) -> int:
        if self.level0_tile_size % self.mask_downsample != 0:
            raise ValueError(
                "level-0 tile size must be divisible by mask_downsample: "
                f"{self.level0_tile_size} vs {self.mask_downsample}"
            )
        return self.level0_tile_size // self.mask_downsample


def load_tissue_mask(mask_path: Path, threshold: float) -> np.ndarray:
    """Load the highest-resolution image from a TIFF tissue mask as HxW bool."""
    with tifffile.TiffFile(mask_path) as tif:
        mask = tif.series[0].levels[0].asarray()

    mask = np.squeeze(mask)

    # Be tolerant of RGB/RGBA or channel-first masks.
    if mask.ndim == 3:
        if mask.shape[-1] in (1, 3, 4):
            mask = np.max(mask, axis=-1)
        elif mask.shape[0] in (1, 3, 4):
            mask = np.max(mask, axis=0)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape} for {mask_path}")

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D tissue mask, got shape {mask.shape}: {mask_path}")

    return mask > threshold


def compute_occupancy_grid(tissue: np.ndarray, tile_size: int) -> np.ndarray:
    """Compute non-overlapping tissue occupancy, padding only the bottom/right edges.

    Edge padding is background (False), so a partial edge tile must still satisfy
    the same occupancy threshold as a full tile.
    """
    height, width = tissue.shape
    pad_h = (-height) % tile_size
    pad_w = (-width) % tile_size

    if pad_h or pad_w:
        tissue = np.pad(
            tissue,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=False,
        )

    grid_h = tissue.shape[0] // tile_size
    grid_w = tissue.shape[1] // tile_size

    # Shape: (grid_h, tile_h, grid_w, tile_w) -> mean over each tile.
    occupancy = tissue.reshape(grid_h, tile_size, grid_w, tile_size).mean(axis=(1, 3))
    return occupancy.astype(np.float32, copy=False)


def write_case_coords(
    case_id: str,
    mask_path: Path,
    slide_path: Path,
    output_path: Path,
    cfg: TilingConfig,
) -> dict:
    if not slide_path.is_file():
        raise FileNotFoundError(f"Missing WSI for {case_id}: {slide_path}")

    tissue = load_tissue_mask(mask_path, cfg.mask_threshold)
    mask_h, mask_w = tissue.shape
    occupancy = compute_occupancy_grid(tissue, cfg.mask_tile_size)

    # Official GigaPath uses a strict '>' occupancy test.
    grid_y, grid_x = np.where(occupancy > cfg.occupancy_threshold)
    ratios = occupancy[grid_y, grid_x]

    # Coordinates used to read the original 40x WSI.
    x_l0 = grid_x.astype(np.int64) * cfg.level0_tile_size
    y_l0 = grid_y.astype(np.int64) * cfg.level0_tile_size

    # Coordinates in the official 20x / 256-pixel GigaPath grid.
    # GigaPath slide_encoder.coords_to_pos() divides these by tile_size=256.
    x_20x = grid_x.astype(np.int64) * cfg.target_tile_size
    y_20x = grid_y.astype(np.int64) * cfg.target_tile_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "x_l0",
                "y_l0",
                "x_20x",
                "y_20x",
                "tissue_ratio",
            ]
        )
        writer.writerows(
            (
                case_id,
                int(x0),
                int(y0),
                int(x20),
                int(y20),
                f"{float(ratio):.6f}",
            )
            for x0, y0, x20, y20, ratio in zip(x_l0, y_l0, x_20x, y_20x, ratios)
        )

    tmp_path.replace(output_path)

    return {
        "case_id": case_id,
        "slide_file": slide_path.name,
        "mask_file": mask_path.name,
        "mask_width": mask_w,
        "mask_height": mask_h,
        "num_tiles": len(grid_x),
    }


def iter_cases(data_dir: Path, recursive: bool = False) -> Iterable[tuple[str, Path, Path]]:
    pattern = "**/*_tissue.tif" if recursive else "*_tissue.tif"
    for mask_path in sorted(data_dir.glob(pattern)):
        case_id = mask_path.name[: -len("_tissue.tif")]
        slide_path = mask_path.with_name(f"{case_id}.tif")
        yield case_id, mask_path, slide_path


def process_one(args: tuple[str, Path, Path, Path, TilingConfig]) -> dict:
    case_id, mask_path, slide_path, output_dir, cfg = args
    return write_case_coords(
        case_id=case_id,
        mask_path=mask_path,
        slide_path=slide_path,
        output_path=output_dir / f"{case_id}.csv",
        cfg=cfg,
    )


def save_summary(rows: list[dict], output_dir: Path, cfg: TilingConfig) -> None:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "slide_file",
                "mask_file",
                "mask_width",
                "mask_height",
                "num_tiles",
                "level0_tile_size",
                "target_tile_size",
                "mask_tile_size",
                "occupancy_threshold",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["case_id"]):
            writer.writerow(
                {
                    **row,
                    "level0_tile_size": cfg.level0_tile_size,
                    "target_tile_size": cfg.target_tile_size,
                    "mask_tile_size": cfg.mask_tile_size,
                    "occupancy_threshold": cfg.occupancy_threshold,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official-style GigaPath patch coordinates from OWL tissue masks."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing *.tif and *_tissue.tif")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-case coordinate CSV files")
    parser.add_argument("--workers", type=int, default=1, help="Number of mask files processed in parallel (default: 1)")
    parser.add_argument("--recursive", action="store_true", help="Search for *_tissue.tif recursively")
    parser.add_argument("--occupancy-threshold", type=float, default=0.1, help="Keep tiles with tissue ratio > threshold (default: 0.1)")
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="Mask pixels > threshold are tissue (default: 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)
    if not 0.0 <= args.occupancy_threshold <= 1.0:
        raise ValueError("--occupancy-threshold must be in [0, 1]")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    cfg = TilingConfig(
        occupancy_threshold=args.occupancy_threshold,
        mask_threshold=args.mask_threshold,
    )

    cases = list(iter_cases(args.data_dir, recursive=args.recursive))
    if not cases:
        raise RuntimeError(f"No *_tissue.tif files found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("GigaPath coordinate generation")
    print(f"  cases:               {len(cases)}")
    print(f"  target:              {cfg.target_tile_size}px @ {cfg.target_magnification:g}x")
    print(f"  WSI level-0 tile:    {cfg.level0_tile_size}px @ {cfg.level0_magnification:g}x")
    print(f"  tissue-mask window:  {cfg.mask_tile_size}px (downsample={cfg.mask_downsample}x)")
    print(f"  stride:              same as tile size (non-overlapping)")
    print(f"  occupancy:           > {cfg.occupancy_threshold}")

    jobs = [
        (case_id, mask_path, slide_path, args.output_dir, cfg)
        for case_id, mask_path, slide_path in cases
    ]

    rows: list[dict] = []
    errors: list[tuple[str, Exception]] = []

    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            case_id = job[0]
            try:
                row = process_one(job)
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {case_id}: {row['num_tiles']} tiles")
            except Exception as exc:  # continue other slides, report all failures at the end
                errors.append((case_id, exc))
                print(f"[{i}/{len(jobs)}] ERROR {case_id}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_case = {pool.submit(process_one, job): job[0] for job in jobs}
            done = 0
            for future in as_completed(future_to_case):
                done += 1
                case_id = future_to_case[future]
                try:
                    row = future.result()
                    rows.append(row)
                    print(f"[{done}/{len(jobs)}] {case_id}: {row['num_tiles']} tiles")
                except Exception as exc:
                    errors.append((case_id, exc))
                    print(f"[{done}/{len(jobs)}] ERROR {case_id}: {exc}")

    save_summary(rows, args.output_dir, cfg)

    total_tiles = sum(row["num_tiles"] for row in rows)
    print(f"\nFinished: {len(rows)}/{len(cases)} slides, {total_tiles} tiles")
    print(f"Coordinates: {args.output_dir}")
    print(f"Summary:     {args.output_dir / 'summary.csv'}")

    if errors:
        print("\nFailures:")
        for case_id, exc in errors:
            print(f"  {case_id}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
