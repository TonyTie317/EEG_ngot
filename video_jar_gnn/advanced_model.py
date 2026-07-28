"""Low-capacity repeat-set classifiers for condition-level video learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .model import STGCNBlock, normalize_adjacency


def _phase_means(sequence: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,C,T]`` into approximately 0--20%, 20--50%, 50--100%."""
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {sequence.shape}")
    time = sequence.shape[-1]
    if time < 3:
        pooled = sequence.mean(dim=-1)
        return torch.cat((pooled, pooled, pooled), dim=1)
    first = max(1, min(time - 2, int(round(time * 0.2))))
    second = max(first + 1, min(time - 1, int(round(time * 0.5))))
    return torch.cat(
        (
            sequence[:, :, :first].mean(dim=-1),
            sequence[:, :, first:second].mean(dim=-1),
            sequence[:, :, second:].mean(dim=-1),
        ),
        dim=1,
    )


class STGCNEncoder(nn.Module):
    """The corrected spatial-temporal graph backbone without a class head."""

    def __init__(
        self,
        *,
        num_features: int,
        num_nodes: int,
        hidden_channels: int,
        dropout: float,
        temporal_pooling: str = "global",
    ):
        super().__init__()
        if temporal_pooling not in {"global", "segments"}:
            raise ValueError("temporal_pooling must be global or segments")
        hidden = int(hidden_channels)
        self.num_features = int(num_features)
        self.num_nodes = int(num_nodes)
        self.temporal_pooling = temporal_pooling
        self.output_dim = hidden * 2 * (3 if temporal_pooling == "segments" else 1)
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
        self.edge_logits = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes))
                for _ in self.blocks
            ]
        )

    def forward(
        self, inputs: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,V,T,F], got {inputs.shape}")
        batch, nodes, time, features = inputs.shape
        if nodes != self.num_nodes or features != self.num_features:
            raise ValueError(
                f"Expected V={self.num_nodes}, F={self.num_features}; "
                f"got V={nodes}, F={features}"
            )
        output = inputs.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch, nodes * features, time)
        output = self.input_norm(output)
        output = output.view(batch, nodes, features, time)
        output = output.permute(0, 2, 3, 1).contiguous()
        for block, edge_logits in zip(self.blocks, self.edge_logits):
            symmetric = (edge_logits + edge_logits.transpose(0, 1)) / 2.0
            weighted = adjacency * (2.0 * torch.sigmoid(symmetric))
            output = block(output, normalize_adjacency(weighted))
        sequence = output.mean(dim=-1)
        if self.temporal_pooling == "segments":
            return _phase_means(sequence)
        return sequence.mean(dim=-1)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.gelu(inputs + self.layers(inputs))


class TCNEncoder(nn.Module):
    """Multi-scale temporal baseline over flattened node features."""

    def __init__(
        self,
        *,
        num_features: int,
        num_nodes: int,
        hidden_channels: int,
        dropout: float,
        temporal_pooling: str = "global",
    ):
        super().__init__()
        if temporal_pooling not in {"global", "segments"}:
            raise ValueError("temporal_pooling must be global or segments")
        input_channels = int(num_features) * int(num_nodes)
        hidden = int(hidden_channels)
        self.temporal_pooling = temporal_pooling
        self.output_dim = hidden * (4 if temporal_pooling == "segments" else 2)
        self.input_norm = nn.BatchNorm1d(input_channels)
        self.projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            TemporalResidualBlock(
                hidden, kernel_size=7, dilation=1, dropout=dropout
            ),
            TemporalResidualBlock(
                hidden, kernel_size=5, dilation=2, dropout=dropout
            ),
            TemporalResidualBlock(
                hidden, kernel_size=3, dilation=4, dropout=dropout
            ),
        )

    def forward(
        self, inputs: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> torch.Tensor:
        del adjacency
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,V,T,F], got {inputs.shape}")
        output = inputs.permute(0, 1, 3, 2).flatten(1, 2)
        output = self.blocks(self.projection(self.input_norm(output)))
        if self.temporal_pooling == "segments":
            return torch.cat((_phase_means(output), output.amax(dim=-1)), dim=1)
        return torch.cat(
            (output.mean(dim=-1), output.amax(dim=-1)), dim=1
        )


class GRUEncoder(nn.Module):
    """One-layer bidirectional GRU baseline kept deliberately small."""

    def __init__(
        self,
        *,
        num_features: int,
        num_nodes: int,
        hidden_channels: int,
        dropout: float,
        temporal_pooling: str = "global",
    ):
        super().__init__()
        del dropout
        if temporal_pooling not in {"global", "segments"}:
            raise ValueError("temporal_pooling must be global or segments")
        input_features = int(num_features) * int(num_nodes)
        hidden = int(hidden_channels)
        self.temporal_pooling = temporal_pooling
        self.output_dim = hidden * 2
        self.input_norm = nn.LayerNorm(input_features)
        self.projection = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = hidden * (8 if temporal_pooling == "segments" else 4)
        self.output_projection = nn.Linear(pooled_dim, self.output_dim)

    def forward(
        self, inputs: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> torch.Tensor:
        del adjacency
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B,V,T,F], got {inputs.shape}")
        sequence = inputs.permute(0, 2, 1, 3).flatten(2, 3)
        sequence = self.projection(self.input_norm(sequence))
        output, _ = self.gru(sequence)
        temporal = output.transpose(1, 2)
        if self.temporal_pooling == "segments":
            pooled = torch.cat(
                (_phase_means(temporal), temporal.amax(dim=-1)), dim=1
            )
        else:
            pooled = torch.cat(
                (temporal.mean(dim=-1), temporal.amax(dim=-1)), dim=1
            )
        return self.output_projection(pooled)


def build_encoder(
    name: str,
    *,
    num_features: int,
    num_nodes: int,
    hidden_channels: int,
    dropout: float,
    temporal_pooling: str = "global",
) -> nn.Module:
    classes = {
        "stgcn": STGCNEncoder,
        "tcn": TCNEncoder,
        "gru": GRUEncoder,
    }
    if name not in classes:
        raise ValueError(f"Unknown encoder {name!r}")
    return classes[name](
        num_features=num_features,
        num_nodes=num_nodes,
        hidden_channels=hidden_channels,
        dropout=dropout,
        temporal_pooling=temporal_pooling,
    )


class RepeatSetClassifier(nn.Module):
    """Encode only valid repeats, pool invariantly, then classify once."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_classes: int,
        hidden_channels: int,
        dropout: float,
        aggregation: str,
        objective: str,
    ):
        super().__init__()
        if aggregation not in {"mean", "mean_std"}:
            raise ValueError("aggregation must be mean or mean_std")
        if objective not in {"ce", "ordinal"}:
            raise ValueError("objective must be ce or ordinal")
        if objective == "ordinal" and num_classes != 3:
            raise ValueError("ordinal objective is currently defined for JAR3")
        self.encoder = encoder
        self.embedding_dim = int(encoder.output_dim)
        self.num_classes = int(num_classes)
        self.aggregation = aggregation
        self.objective = objective
        pooled_dim = self.embedding_dim * (2 if aggregation == "mean_std" else 1)
        hidden = int(hidden_channels)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if objective == "ordinal":
            self.score_head = nn.Linear(hidden, 1)
            self.threshold_base = nn.Parameter(torch.tensor(-0.5))
            self.threshold_gap_raw = nn.Parameter(torch.tensor(0.0))
            self.class_head = None
        else:
            self.class_head = nn.Linear(hidden, num_classes)
            self.score_head = None

    def _pool(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * weights).sum(dim=1) / counts
        if self.aggregation == "mean":
            return mean
        variance = (
            torch.square(embeddings - mean.unsqueeze(1)) * weights
        ).sum(dim=1) / counts
        return torch.cat((mean, torch.sqrt(variance + 1e-6)), dim=1)

    def forward(
        self,
        graphs: torch.Tensor,
        adjacency: torch.Tensor,
        repeat_mask: torch.Tensor,
    ) -> torch.Tensor:
        if graphs.ndim != 5:
            raise ValueError(f"graphs must be [B,R,V,T,F], got {graphs.shape}")
        if repeat_mask.shape != graphs.shape[:2]:
            raise ValueError("repeat_mask shape is incompatible with graphs")
        if not torch.all(repeat_mask.any(dim=1)):
            raise ValueError("Every condition must contain at least one repeat")
        batch, repeats, nodes, time, features = graphs.shape
        flat_mask = repeat_mask.reshape(-1)
        valid_graphs = graphs.reshape(
            batch * repeats, nodes, time, features
        )[flat_mask]
        expanded_adjacency = (
            adjacency[:, None]
            .expand(batch, repeats, nodes, nodes)
            .reshape(batch * repeats, nodes, nodes)[flat_mask]
        )
        valid_embeddings = self.encoder(valid_graphs, expanded_adjacency)
        flat_embeddings = valid_embeddings.new_zeros(
            (batch * repeats, self.embedding_dim)
        )
        flat_embeddings[flat_mask] = valid_embeddings
        embeddings = flat_embeddings.view(batch, repeats, self.embedding_dim)
        pooled = self._pool(embeddings, repeat_mask)
        hidden = self.shared_head(pooled)
        if self.objective == "ordinal":
            score = self.score_head(hidden).squeeze(1)
            threshold_0 = self.threshold_base
            threshold_1 = threshold_0 + functional.softplus(
                self.threshold_gap_raw
            )
            return torch.stack(
                (score - threshold_0, score - threshold_1), dim=1
            )
        return self.class_head(hidden)

    def probabilities(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.objective == "ce":
            return torch.softmax(outputs, dim=1)
        exceed = torch.sigmoid(outputs)
        probabilities = torch.stack(
            (
                1.0 - exceed[:, 0],
                exceed[:, 0] - exceed[:, 1],
                exceed[:, 1],
            ),
            dim=1,
        )
        return probabilities.clamp_min(0.0) / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

    def predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return class decisions using the objective's proper decision rule."""
        if self.objective == "ce":
            return outputs.argmax(dim=1)
        # CORAL/cumulative ordinal prediction is the number of surpassed
        # thresholds. Using argmax over derived class probabilities can make
        # the middle class mathematically impossible when thresholds are close.
        return (outputs > 0.0).sum(dim=1).long()
