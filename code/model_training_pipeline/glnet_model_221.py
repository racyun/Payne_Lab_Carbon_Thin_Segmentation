# -*- coding: utf-8 -*-
"""
Global-Local dual-branch segmentation model (GLNet-style) on top of SwinV2 + UPerNet.

One SHARED SwinV2 backbone is run twice per forward:
  - global branch: the whole image downsampled to a full frame  -> 4-stage feature pyramid
  - local branch:  one full-resolution crop of that image       -> 4-stage feature pyramid
For each pyramid stage, the region of the GLOBAL feature map that corresponds to the crop
(via the crop's normalized bbox, roi_align) is resized to the local feature size, concatenated
with the local features, and 1x1-convolved back to the stage's channel count. The fused pyramid
feeds the reused UPerNet decode head, producing per-pixel logits for the CROP region.

This lets the model combine whole-image context (grains that span the frame) with full-resolution
local detail, addressing the "512 crops chop whole grains" problem. Reuses the existing SSL
backbone via load_ssl_backbone_checkpoint (the module exposes .backbone).

Smoke test:  python code/model_training_pipeline/glnet_model_221.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from swin_training_pipeline_221 import (
    BACKBONE_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    get_model_semantic_segmentation,
)


class GlobalLocalUperNet(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX,
                 backbone_id: str = BACKBONE_ID, out_indices: list[int] | None = None):
        super().__init__()
        base = get_model_semantic_segmentation(num_classes, ignore_index, backbone_id, out_indices)
        self.backbone = base.backbone          # shared, run twice per forward
        self.decode_head = base.decode_head    # reused UPerNet head
        self.num_classes = num_classes

        # Per-stage channel dims (derived from a dummy forward so we don't depend on .channels).
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(pixel_values=torch.zeros(1, 3, 256, 256)).feature_maps
        chans = [int(f.shape[1]) for f in feats]
        # Fuse concat(local, global-region) -> back to the stage's channel count.
        self.fuse = nn.ModuleList([nn.Conv2d(2 * c, c, kernel_size=1) for c in chans])

    @staticmethod
    def _global_region(gf: torch.Tensor, bbox_norm: torch.Tensor, out_hw) -> torch.Tensor:
        """Extract, per sample, the bbox region of the global feature map, resized to out_hw."""
        b, _, h, w = gf.shape
        boxes = bbox_norm.to(gf.dtype).clone()
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        idx = torch.arange(b, device=gf.device, dtype=gf.dtype).unsqueeze(1)
        rois = torch.cat([idx, boxes], dim=1)  # [B, 5] = (batch_idx, x0, y0, x1, y1)
        return roi_align(gf, rois, output_size=(int(out_hw[0]), int(out_hw[1])),
                         spatial_scale=1.0, aligned=True)

    def forward(self, global_img: torch.Tensor, local_crop: torch.Tensor,
                bbox_norm: torch.Tensor) -> torch.Tensor:
        g_feats = self.backbone(pixel_values=global_img).feature_maps
        l_feats = self.backbone(pixel_values=local_crop).feature_maps
        fused = []
        for i, (gf, lf) in enumerate(zip(g_feats, l_feats)):
            region = self._global_region(gf, bbox_norm, lf.shape[-2:])
            fused.append(self.fuse[i](torch.cat([lf, region], dim=1)))
        logits = self.decode_head(fused)  # [B, num_classes, H/4, W/4] of the local crop
        logits = F.interpolate(logits, size=local_crop.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    # --- Tiled-eval helpers: encode ONE global image once, reuse for a batch of tiles. ---
    def encode_global(self, global_img: torch.Tensor):
        """global_img: [1,3,H,W] -> tuple of 4 global feature maps (batch 1)."""
        return self.backbone(pixel_values=global_img).feature_maps

    @staticmethod
    def _global_region_shared(gf: torch.Tensor, bbox_norm: torch.Tensor, out_hw) -> torch.Tensor:
        """gf: [1,C,H,W] (one global); bbox_norm: [b,4] -> [b,C,out_h,out_w] (all from that global)."""
        b = bbox_norm.shape[0]
        _, _, h, w = gf.shape
        boxes = bbox_norm.to(gf.dtype).clone()
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        idx = torch.zeros(b, 1, device=gf.device, dtype=gf.dtype)  # all rois point to global 0
        rois = torch.cat([idx, boxes], dim=1)
        return roi_align(gf, rois, output_size=(int(out_hw[0]), int(out_hw[1])),
                         spatial_scale=1.0, aligned=True)

    def forward_local(self, g_feats, local_crop: torch.Tensor, bbox_norm: torch.Tensor) -> torch.Tensor:
        """Predict a BATCH of tiles reusing precomputed global features (from encode_global)."""
        l_feats = self.backbone(pixel_values=local_crop).feature_maps
        fused = []
        for i, (gf, lf) in enumerate(zip(g_feats, l_feats)):
            region = self._global_region_shared(gf, bbox_norm, lf.shape[-2:])
            fused.append(self.fuse[i](torch.cat([lf, region], dim=1)))
        logits = self.decode_head(fused)
        logits = F.interpolate(logits, size=local_crop.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def build_glnet(num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                backbone_id: str = BACKBONE_ID, out_indices: list[int] | None = None) -> GlobalLocalUperNet:
    return GlobalLocalUperNet(num_classes, ignore_index, backbone_id, out_indices)


if __name__ == "__main__":
    # Smoke test: build the model and run one dual-branch forward, reporting shapes + peak memory.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    model = build_glnet().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M | fuse stages: {[m.in_channels for m in model.fuse]}")

    gh, gw, crop = 1024, 1280, 512
    global_img = torch.randn(1, 3, gh, gw, device=dev)
    local_crop = torch.randn(1, 3, crop, crop, device=dev)
    bbox = torch.tensor([[0.25, 0.25, 0.25 + crop / gw, 0.25 + crop / gh]], device=dev)  # normalized

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    logits = model(global_img, local_crop, bbox)
    loss = logits.float().mean()
    loss.backward()  # exercise the backward path (memory realism)
    print(f"logits: {tuple(logits.shape)}  (expected [1, {NUM_CLASSES}, {crop}, {crop}])")
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
              f"(global {gh}x{gw} + crop {crop}x{crop}, batch 1, with backward)")
    print("smoke test OK")
