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

The script reads the low-resolution tissue masks to generate coordinates. It
also saves a low-resolution WSI thumbnail with the selected patch grid overlaid
for visual quality control.
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
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TilingConfig:
    level0_magnification: float = 40.0
    target_magnification: float = 20.0
    target_tile_size: int = 256
    mask_downsample: int = 16
    occupancy_threshold: float = 0.1
    mask_threshold: float = 0.0
    overlay_max_size: int = 2048
    save_overlay: bool = True

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


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a TIFF level array to uint8 RGB."""
    array = np.squeeze(array)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            else:
                array = array[..., :3]
        elif array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array[:3], 0, -1)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
        else:
            raise ValueError(f"Unsupported thumbnail shape: {array.shape}")
    else:
        raise ValueError(f"Unsupported thumbnail shape: {array.shape}")

    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)

    lo = float(np.percentile(array[finite], 0.5))
    hi = float(np.percentile(array[finite], 99.5))
    if hi <= lo:
        hi = lo + 1.0
    array = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def load_wsi_thumbnail(slide_path: Path, max_size: int) -> tuple[Image.Image, tuple[int, int]]:
    """Read a suitable low-resolution TIFF pyramid level and return an RGB thumbnail.

    Returns:
        thumbnail: PIL RGB image with longest side <= max_size.
        level0_size: (width, height) of the original WSI level-0.
    """
    with tifffile.TiffFile(slide_path) as tif:
        levels = tif.series[0].levels

        level_shapes: list[tuple[int, int, int]] = []
        for i, level in enumerate(levels):
            shape = level.shape
            if len(shape) < 2:
                continue
            if len(shape) == 2:
                h, w = shape
            elif shape[-1] in (1, 3, 4):
                h, w = shape[-3], shape[-2]
            elif shape[0] in (1, 3, 4):
                h, w = shape[-2], shape[-1]
            else:
                h, w = shape[-2], shape[-1]
            level_shapes.append((i, int(w), int(h)))

        if not level_shapes:
            raise ValueError(f"Cannot determine TIFF pyramid dimensions: {slide_path}")

        level0_w = level_shapes[0][1]
        level0_h = level_shapes[0][2]

        # Prefer the highest-resolution level that already fits near the requested
        # thumbnail size. If all pyramid levels are larger, use the lowest level.
        fitting = [item for item in level_shapes if max(item[1], item[2]) <= max_size]
        if fitting:
            chosen_idx, _, _ = max(fitting, key=lambda item: max(item[1], item[2]))
        else:
            chosen_idx, _, _ = min(level_shapes, key=lambda item: max(item[1], item[2]))

        thumb_array = levels[chosen_idx].asarray()

    thumb = Image.fromarray(_to_uint8_rgb(thumb_array), mode="RGB")
    if max(thumb.size) > max_size:
        scale = max_size / max(thumb.size)
        new_size = (
            max(1, int(round(thumb.width * scale))),
            max(1, int(round(thumb.height * scale))),
        )
        thumb = thumb.resize(new_size, Image.Resampling.BILINEAR)

    return thumb, (level0_w, level0_h)


def save_patch_overlay(
    slide_path: Path,
    output_path: Path,
    x_l0: np.ndarray,
    y_l0: np.ndarray,
    level0_tile_size: int,
    max_size: int,
) -> None:
    """Save a thumbnail of the WSI with selected level-0 patch boxes overlaid."""
    thumbnail, (level0_w, level0_h) = load_wsi_thumbnail(slide_path, max_size=max_size)

    scale_x = thumbnail.width / level0_w
    scale_y = thumbnail.height / level0_h

    base = thumbnail.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    box_w = max(1, int(round(level0_tile_size * scale_x)))
    box_h = max(1, int(round(level0_tile_size * scale_y)))
    line_width = 1 if max(base.size) <= 1200 else 2

    for x0, y0 in zip(x_l0, y_l0):
        x1 = int(round(int(x0) * scale_x))
        y1 = int(round(int(y0) * scale_y))
        x2 = min(base.width - 1, x1 + box_w)
        y2 = min(base.height - 1, y1 + box_h)
        if x2 < 0 or y2 < 0 or x1 >= base.width or y1 >= base.height:
            continue
        draw.rectangle(
            [max(0, x1), max(0, y1), x2, y2],
            fill=(0, 255, 0, 35),
            outline=(0, 220, 0, 220),
            width=line_width,
        )

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path, quality=90)


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
    overlay_path: Path,
    cfg: TilingConfig,
) -> dict:
    # The main process filters missing WSIs before submitting jobs. Keep this
    # check as a safeguard in case a file disappears while the script is running.
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

    if cfg.save_overlay:
        save_patch_overlay(
            slide_path=slide_path,
            output_path=overlay_path,
            x_l0=x_l0,
            y_l0=y_l0,
            level0_tile_size=cfg.level0_tile_size,
            max_size=cfg.overlay_max_size,
        )

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
        overlay_path=output_dir / "overlays" / f"{case_id}.jpg",
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
    parser.add_argument("--overlay-max-size", type=int, default=2048, help="Maximum side length of overlay thumbnails (default: 2048)")
    parser.add_argument("--no-overlay", action="store_true", help="Do not generate WSI patch overlay thumbnails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)
    if not 0.0 <= args.occupancy_threshold <= 1.0:
        raise ValueError("--occupancy-threshold must be in [0, 1]")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.overlay_max_size < 256:
        raise ValueError("--overlay-max-size must be >= 256")

    cfg = TilingConfig(
        occupancy_threshold=args.occupancy_threshold,
        mask_threshold=args.mask_threshold,
        overlay_max_size=args.overlay_max_size,
        save_overlay=not args.no_overlay,
    )

    discovered_cases = list(iter_cases(args.data_dir, recursive=args.recursive))
    if not discovered_cases:
        raise RuntimeError(f"No *_tissue.tif files found in {args.data_dir}")

    # A tissue mask may arrive before its corresponding WSI during dataset sync.
    # Missing WSIs are expected in that situation, so skip them rather than
    # treating them as processing failures.
    missing_cases = [case for case in discovered_cases if not case[2].is_file()]
    cases = [case for case in discovered_cases if case[2].is_file()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_overlay:
        (args.output_dir / "overlays").mkdir(parents=True, exist_ok=True)

    print("GigaPath coordinate generation")
    print(f"  tissue masks found:  {len(discovered_cases)}")
    print(f"  cases with WSI:      {len(cases)}")
    print(f"  missing WSI skipped: {len(missing_cases)}")
    print(f"  target:              {cfg.target_tile_size}px @ {cfg.target_magnification:g}x")
    print(f"  WSI level-0 tile:    {cfg.level0_tile_size}px @ {cfg.level0_magnification:g}x")
    print(f"  tissue-mask window:  {cfg.mask_tile_size}px (downsample={cfg.mask_downsample}x)")
    print(f"  stride:              same as tile size (non-overlapping)")
    print(f"  occupancy:           > {cfg.occupancy_threshold}")
    print(f"  overlays:            {'enabled' if cfg.save_overlay else 'disabled'}")

    for case_id, _, slide_path in missing_cases:
        print(f"[SKIP] {case_id}: missing WSI {slide_path}")

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
    print(f"\nFinished: {len(rows)}/{len(cases)} available slides, {total_tiles} tiles")
    print(f"Skipped missing WSI: {len(missing_cases)}")
    print(f"Coordinates: {args.output_dir}")
    if cfg.save_overlay:
        print(f"Overlays:    {args.output_dir / 'overlays'}")
    print(f"Summary:     {args.output_dir / 'summary.csv'}")

    if errors:
        print("\nFailures:")
        for case_id, exc in errors:
            print(f"  {case_id}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
