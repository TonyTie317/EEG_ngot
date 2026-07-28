"""A compact, corrected ST-GCN for small facial graph datasets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Add one self-loop and apply symmetric degree normalization."""
    if adjacency.ndim not in (2, 3):
        raise ValueError(f"adjacency must be [V,V] or [B,V,V], got {adjacency.shape}")
    num_nodes = adjacency.shape[-1]
    identity = torch.eye(num_nodes, dtype=adjacency.dtype, device=adjacency.device)
    if adjacency.ndim == 3:
        identity = identity.unsqueeze(0)
    with_self = adjacency + identity
    degree = with_self.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * with_self * inv_sqrt.unsqueeze(-2)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            aggregated = torch.einsum("bctv,vw->bctw", inputs, adjacency)
        else:
            aggregated = torch.einsum("bctv,bvw->bctw", inputs, adjacency)
        return self.projection(aggregated)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_kernel: int,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd")
        self.graph_conv = SpatialGraphConv(in_channels, out_channels)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(temporal_kernel // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(inputs)
        output = self.graph_conv(inputs, adjacency)
        output = self.temporal(output) + residual
        return functional.relu(output, inplace=True)


class FacialSTGCN(nn.Module):
    """Small ST-GCN with positive learnable weights on anatomical edges.

    Input shape is ``[batch, nodes, time, features]``.
    """

    def __init__(
        self,
        *,
        num_features: int = 10,
        num_nodes: int = 15,
        num_classes: int = 3,
        hidden_channels: int = 32,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_nodes = int(num_nodes)
        hidden = int(hidden_channels)
        self.input_norm = nn.BatchNorm1d(self.num_features * self.num_nodes)
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    self.num_features,
                    hidden,
                    temporal_kernel=7,
                    residual=False,
                ),
                STGCNBlock(
                    hidden,
                    hidden,
                    temporal_kernel=5,
                    dropout=dropout,
                ),
                STGCNBlock(
                    hidden,
                    hidden * 2,
                    temporal_kernel=5,
                    stride=2,
                    dropout=dropout,
                ),
                STGCNBlock(
                    hidden * 2,
                    hidden * 2,
                    temporal_kernel=3,
                    stride=2,
                    dropout=dropout,
                ),
            ]
        )
        # sigmoid(0) * 2 = 1: every existing edge starts at its anatomical
        # weight. A zero edge stays zero; the model does not invent anatomy.
        self.edge_logits = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes))
                for _ in self.blocks
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"inputs must be [B,V,T,F], got {inputs.shape}")
        batch, num_nodes, time, num_features = inputs.shape
        if num_nodes != self.num_nodes or num_features != self.num_features:
            raise ValueError(
                f"Expected V={self.num_nodes}, F={self.num_features}; "
                f"got V={num_nodes}, F={num_features}"
            )

        # Correct layout before BatchNorm: [B,V,T,F] -> [B,V*F,T].
        output = inputs.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch, num_nodes * num_features, time)
        output = self.input_norm(output)
        output = output.view(batch, num_nodes, num_features, time)
        output = output.permute(0, 2, 3, 1).contiguous()

        for block, edge_logits in zip(self.blocks, self.edge_logits):
            symmetric_logits = (edge_logits + edge_logits.transpose(0, 1)) / 2.0
            positive_scale = 2.0 * torch.sigmoid(symmetric_logits)
            weighted = adjacency * positive_scale
            normalized = normalize_adjacency(weighted)
            output = block(output, normalized)

        pooled = output.mean(dim=(-2, -1))
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
