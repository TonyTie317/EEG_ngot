#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved ST-GCN for AU-based emotion recognition
Inspired by original ST-GCN paper with adaptations for facial AU data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBlock(nn.Module):
    """
    Single ST-GCN block: Spatial GCN + Temporal Conv + Residual
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        kernel_size: (temporal_k, spatial_k) - spatial_k is number of A partitions
        stride: Temporal stride for downsampling
        dropout: Dropout rate
        residual: Whether to use residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=(9, 1), 
                 stride=1, dropout=0.0, residual=True):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # temporal kernel must be odd
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        # Spatial GCN
        self.gcn = SpatialGCN(in_channels, out_channels)
        
        # Temporal Conv with BatchNorm
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        """
        Args:
            x: [B, C, T, N] - batch, channels, time, nodes
            A: [B, N, N] or [N, N] - adjacency matrix
        Returns:
            x: [B, C_out, T', N]
            A: adjacency (unchanged)
        """
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolution
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, A):
        """
        Args:
            x: [B, C, T, N]
            A: [B, N, N] or [N, N]
        """
        B, C, T, N = x.shape
        
        # Normalize adjacency: D^(-1/2) * A * D^(-1/2)
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)
        
        # Add self-loops
        A = A + torch.eye(N, device=A.device).unsqueeze(0)
        
        # Degree matrix
        D = A.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        D_inv_sqrt = D.pow(-0.5)
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        A_norm = D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)
        
        # Graph convolution
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, N, C]
        x = x.view(B * T, N, C)
        
        # Message passing: X' = A_norm @ X
        x = torch.bmm(A_norm.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, N, N), x)
        
        x = x.view(B, T, N, C).permute(0, 3, 1, 2)  # [B, C, T, N]
        
        # Feature transformation
        x = self.conv(x)
        
        return x, A


class ImprovedSTGCN(nn.Module):
    """
    Improved ST-GCN for AU emotion recognition
    
    Args:
        in_channels: Input feature dimension (e.g., 10 for your AU features)
        num_classes: Number of emotion classes (e.g., 4)
        num_nodes: Number of AU nodes (e.g., 15)
        edge_importance_weighting: Use learnable edge weights
        dropout: Dropout rate
    """
    def __init__(self, in_channels=10, num_classes=4, num_nodes=15,
                 edge_importance_weighting=True, dropout=0.5):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Input BatchNorm (normalize over all node features)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        # ST-GCN layers with progressive channel expansion
        # Architecture: 64 -> 64 -> 128 -> 128 -> 256
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, 64, kernel_size=(9, 1), stride=1, 
                      dropout=0, residual=False),
            STGCNBlock(64, 64, kernel_size=(9, 1), stride=1, dropout=dropout),
            STGCNBlock(64, 128, kernel_size=(9, 1), stride=2, dropout=dropout),
            STGCNBlock(128, 128, kernel_size=(9, 1), stride=1, dropout=dropout),
            STGCNBlock(128, 256, kernel_size=(9, 1), stride=2, dropout=dropout),
        ])
        
        # Edge importance weighting (learnable)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(num_nodes, num_nodes))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # Classifier
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x, A):
        """
        Args:
            x: [B, N, T, F] - batch, nodes, time, features
            A: [N, N] - adjacency matrix
        Returns:
            logits: [B, num_classes]
        """
        B, N, T, F = x.shape
        
        # Reshape to [B, F, T, N] for processing
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, F, T, N]
        
        # Input normalization
        x_bn = x.permute(0, 3, 2, 1).contiguous()  # [B, N, T, F]
        x_bn = x_bn.view(B, N * F, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(B, N, F, T).permute(0, 2, 3, 1).contiguous()  # [B, F, T, N]
        
        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            # Apply edge importance weighting
            if isinstance(importance, nn.Parameter):
                A_weighted = A * importance
            else:
                A_weighted = A
            
            x, _ = gcn(x, A_weighted)
        
        # Global pooling over time and space
        # x: [B, 256, T', N]
        x = F.avg_pool2d(x, x.size()[2:])  # [B, 256, 1, 1]
        
        # Classification
        x = self.fcn(x)  # [B, num_classes, 1, 1]
        x = x.view(x.size(0), -1)  # [B, num_classes]
        
        return x


class HybridSTGCN(nn.Module):
    """
    Hybrid model: Lighter than full ST-GCN, heavier than SimpleSTGCN
    Good balance for small datasets
    """
    def __init__(self, in_channels=10, num_classes=4, num_nodes=15, 
                 hid=128, dropout=0.3):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        # 3 ST-GCN blocks (lighter than original)
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(in_channels, hid, kernel_size=(7, 1), stride=1, 
                      dropout=0, residual=False),
            STGCNBlock(hid, hid, kernel_size=(5, 1), stride=1, dropout=dropout),
            STGCNBlock(hid, hid*2, kernel_size=(3, 1), stride=2, dropout=dropout),
        ])
        
        # Edge importance
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(num_nodes, num_nodes))
            for _ in self.st_gcn_networks
        ])
        
        # Classifier
        self.fcn = nn.Sequential(
            nn.Conv2d(hid*2, hid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hid, num_classes, kernel_size=1)
        )
    
    def forward(self, x, A):
        """
        Args:
            x: [B, N, T, F]
            A: [N, N]
        Returns:
            [B, num_classes]
        """
        B, N, T, F = x.shape
        
        # Input BN
        x_flat = x.view(B, N * F, T)
        x_flat = self.input_bn(x_flat)
        x = x_flat.view(B, N, F, T).permute(0, 2, 3, 1)  # [B, F, T, N]
        
        # ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            A_weighted = A * importance
            x, _ = gcn(x, A_weighted)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        
        # Classify
        x = self.fcn(x)
        return x.view(x.size(0), -1)


# ============= Factory function =============
def create_stgcn(model_type='hybrid', in_channels=10, num_classes=4, 
                 num_nodes=15, **kwargs):
    """
    Factory function to create ST-GCN variants
    
    Args:
        model_type: 'simple', 'hybrid', 'full'
        in_channels: Feature dimension
        num_classes: Number of classes
        num_nodes: Number of graph nodes
    """
    if model_type == 'simple':
        from model_stgcn import SimpleSTGCN
        return SimpleSTGCN(in_channels, kwargs.get('hid', 128), num_classes)
    elif model_type == 'hybrid':
        return HybridSTGCN(in_channels, num_classes, num_nodes, **kwargs)
    elif model_type == 'full':
        return ImprovedSTGCN(in_channels, num_classes, num_nodes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test các models
    B, N, T, F = 8, 15, 300, 10
    x = torch.randn(B, N, T, F)
    A = torch.randn(N, N)
    
    print("Testing HybridSTGCN...")
    model = HybridSTGCN()
    out = model(x, A)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting ImprovedSTGCN...")
    model_full = ImprovedSTGCN()
    out = model_full(x, A)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_full.parameters()):,}")
