"""Minimal local GigaPath-Flash tile encoder loader.

No network access and no dependency on the full Prov-GigaPath package.
The architecture matches the released GigaPath-Flash tile encoder:
DINOv2-small style ViT-S/16, 384-d embedding, SwiGLU FFN, LayerScale.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    from timm.layers import SwiGLUPacked
    from timm.models.vision_transformer import VisionTransformer
except ImportError as exc:
    raise RuntimeError(
        "timm is required to instantiate the GigaPath-Flash tile encoder. "
        "Install it with: python -m pip install 'timm>=1.0.3'"
    ) from exc


_TILE_ENC_ARGS = dict(
    img_size=224,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=2048 / 384.0,
    mlp_layer=SwiGLUPacked,
    act_layer=nn.SiLU,
    init_values=1e-5,
    num_classes=0,
    global_pool="token",
    class_token=True,
    reg_tokens=0,
)


def build_model(**kwargs) -> VisionTransformer:
    model_args = dict(_TILE_ENC_ARGS)
    model_args.update(kwargs)
    return VisionTransformer(**model_args)


def _unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj

    # Released checkpoints are normally a bare state_dict. Accept common
    # wrappers as well so locally converted checkpoints work without changes.
    if "patch_embed.proj.weight" in obj:
        return obj

    for key in ("model", "state_dict"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    return obj


def create_model(pretrained: str, **kwargs) -> VisionTransformer:
    checkpoint = Path(pretrained).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GigaPath-Flash checkpoint not found: {checkpoint}")

    model = build_model(**kwargs)
    state_dict = _unwrap_state_dict(torch.load(checkpoint, map_location="cpu"))

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Unsupported checkpoint object type {type(state_dict)!r}: {checkpoint}"
        )

    # Handle DataParallel / DDP-style prefixes if present.
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # For this fixed architecture, substantial key mismatch almost certainly
    # means the wrong checkpoint was supplied. Fail early rather than silently
    # extracting invalid features.
    if missing_keys or unexpected_keys:
        missing_preview = ", ".join(missing_keys[:10])
        unexpected_preview = ", ".join(unexpected_keys[:10])
        raise RuntimeError(
            "Checkpoint does not match the expected GigaPath-Flash ViT-S/16 architecture. "
            f"missing={len(missing_keys)} [{missing_preview}] ; "
            f"unexpected={len(unexpected_keys)} [{unexpected_preview}]"
        )

    return model
