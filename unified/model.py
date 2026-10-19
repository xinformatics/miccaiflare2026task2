from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from unified.specs import MODALITIES, TASK_SPECS, TaskSpec


def norm(channels: int) -> nn.Module:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = norm(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class ModalitySetStem(nn.Module):
    """Encode an arbitrary ordered set of named modalities into fixed-width features."""

    def __init__(self, width: int, embedding_dim: int = 24):
        super().__init__()
        self.image_stem = nn.Sequential(
            nn.Conv3d(1, width, 3, padding=1, bias=False),
            norm(width),
            nn.SiLU(inplace=True),
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            norm(width),
            nn.SiLU(inplace=True),
        )
        self.modality_embedding = nn.Embedding(len(MODALITIES), embedding_dim)
        self.embedding_projection = nn.Linear(embedding_dim, width)
        self.attention = nn.Sequential(
            nn.Linear(width + embedding_dim, width),
            nn.SiLU(inplace=True),
            nn.Linear(width, 1),
        )

    def forward(self, images: torch.Tensor, modality_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, depth, height, width = images.shape
        if modality_ids.shape != (batch, channels):
            raise ValueError(f"modality_ids must be {(batch, channels)}, got {tuple(modality_ids.shape)}")
        encoded = self.image_stem(images.reshape(batch * channels, 1, depth, height, width))
        encoded = encoded.reshape(batch, channels, encoded.shape[1], depth, height, width)
        embeddings = self.modality_embedding(modality_ids)
        bias = self.embedding_projection(embeddings)[..., None, None, None]
        encoded = encoded + bias
        pooled = encoded.mean(dim=(-3, -2, -1))
        scores = self.attention(torch.cat((pooled, embeddings), dim=-1)).squeeze(-1)
        weights = scores.softmax(dim=1)
        fused = (encoded * weights[..., None, None, None, None]).sum(dim=1)
        return fused, weights


class GlanceFocusAutoMSC(nn.Module):
    """Single shared Glance/Focus student with dataset-conditioned output heads."""

    def __init__(
        self,
        base_width: int = 32,
        task_embedding_dim: int = 32,
        clinical_dim: int = 8,
        task_specs: Mapping[str, TaskSpec] | None = None,
        task_embedding_count: int = 4096,
    ):
        super().__init__()
        self.task_specs = dict(TASK_SPECS if task_specs is None else task_specs)
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8, base_width * 10]
        self.stem = ModalitySetStem(widths[0])
        self.encoder = nn.ModuleList(
            [
                ResidualBlock(widths[0], widths[0]),
                ResidualBlock(widths[0], widths[1], 2),
                ResidualBlock(widths[1], widths[2], 2),
                ResidualBlock(widths[2], widths[3], 2),
                ResidualBlock(widths[3], widths[4], 2),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                ResidualBlock(widths[4] + widths[3], widths[3]),
                ResidualBlock(widths[3] + widths[2], widths[2]),
                ResidualBlock(widths[2] + widths[1], widths[1]),
                ResidualBlock(widths[1] + widths[0], widths[0]),
            ]
        )
        task_keys = list(self.task_specs)
        if task_embedding_count == len(task_keys):
            # Compatibility path for previously trained checkpoints.
            self.task_to_index = {key: index for index, key in enumerate(task_keys)}
        else:
            self.task_to_index = {
                key: spec.dataset_id % task_embedding_count for key, spec in self.task_specs.items()
            }
        self.task_embedding = nn.Embedding(task_embedding_count, task_embedding_dim)
        self.glance_encoder = nn.Sequential(
            ResidualBlock(widths[0], widths[1], 2),
            ResidualBlock(widths[1], widths[2], 2),
            ResidualBlock(widths[2], widths[3], 2),
        )
        self.glance_mask = nn.Conv3d(widths[3], 2, 1)
        self.segmentation_heads = nn.ModuleDict(
            {key: nn.Conv3d(widths[0], spec.segmentation_classes, 1) for key, spec in self.task_specs.items()}
        )
        cls_features = widths[3] + widths[4] + widths[0] + task_embedding_dim
        self.classification_trunk = nn.Sequential(
            nn.Linear(cls_features, widths[4]),
            nn.LayerNorm(widths[4]),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classification_heads = nn.ModuleDict()
        self.clinical_heads = nn.ModuleDict()
        for task_key, spec in self.task_specs.items():
            for head_name, class_count in spec.classification_heads.items():
                name = f"{task_key}__{head_name}"
                self.classification_heads[name] = nn.Linear(widths[4], class_count)
                self.clinical_heads[name] = nn.Linear(clinical_dim, class_count)
                nn.init.zeros_(self.clinical_heads[name].weight)
                nn.init.zeros_(self.clinical_heads[name].bias)

    def forward(
        self,
        images: torch.Tensor,
        modality_ids: torch.Tensor,
        task_key: str,
        glance_images: torch.Tensor | None = None,
        clinical_features: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor | dict[str, torch.Tensor]]:
        if task_key not in self.task_specs:
            raise KeyError(f"Unknown task {task_key}")
        stem, modality_weights = self.stem(images, modality_ids)

        skips = []
        x = stem
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        bottleneck = skips[-1]

        x = bottleneck
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        segmentation = self.segmentation_heads[task_key](x)

        if glance_images is None:
            glance_shape = tuple(min(64, size) for size in stem.shape[-3:])
            glance_input = F.adaptive_avg_pool3d(stem, glance_shape)
        else:
            glance_input, _ = self.stem(glance_images, modality_ids)
        glance_features = self.glance_encoder(glance_input)
        coarse_segmentation = self.glance_mask(glance_features)

        foreground_probability = segmentation.softmax(dim=1)[:, 1:].sum(dim=1, keepdim=True)
        foreground_probability = foreground_probability.clamp_min(1e-4)
        lesion_features = (x * foreground_probability).sum(dim=(-3, -2, -1))
        lesion_features = lesion_features / foreground_probability.sum(dim=(-3, -2, -1))
        global_features = glance_features.mean(dim=(-3, -2, -1))
        bottleneck_features = bottleneck.mean(dim=(-3, -2, -1))
        task_indices = torch.full(
            (images.shape[0],), self.task_to_index[task_key], dtype=torch.long, device=images.device
        )
        task_features = self.task_embedding(task_indices)
        classification_features = self.classification_trunk(
            torch.cat((global_features, bottleneck_features, lesion_features, task_features), dim=1)
        )
        if clinical_features is None:
            clinical_features = classification_features.new_zeros((images.shape[0], 8))
        classifications = {}
        for head_name in self.task_specs[task_key].classification_heads:
            name = f"{task_key}__{head_name}"
            classifications[head_name] = (
                self.classification_heads[name](classification_features)
                + self.clinical_heads[name](clinical_features)
            )
        return {
            "segmentation": segmentation,
            "coarse_segmentation": coarse_segmentation,
            "classifications": classifications,
            "modality_weights": modality_weights,
        }
