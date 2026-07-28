#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F  # đừng trùng tên biến với F!
#H: hidden size
class SimpleSTGCN(nn.Module):
    """
    Baseline: GCN theo không gian (AU graph) + Temporal 1D Conv.
    Input x: [B, N, T, F_in]; adj: [B or 1, N, N]
    """
    def __init__(self, in_feats: int, hid: int, num_classes: int):
        super().__init__()
        self.theta1 = nn.Linear(in_feats, hid, bias=False)        # feature lift y=xWT+b
        self.temporal1 = nn.Conv1d(hid, hid, kernel_size=5, padding=2)  # theo trục T
        self.theta2 = nn.Linear(hid, hid, bias=False)
        self.temporal2 = nn.Conv1d(hid, hid, kernel_size=3, padding=1)
        self.cls = nn.Linear(hid, num_classes)

    def spatial_gcn(self, x, adj):
        # x: [B, N, H]; adj: [B or 1, N, N]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.size(0) == 1 and x.size(0) > 1:
            adj = adj.expand(x.size(0), -1, -1)
        Ax = torch.bmm(adj, x)  # [B, N, H]
        return Ax

    def forward(self, x, adj):
        # x: [B, N, T, F_in]
        B, N, T, feat_dim = x.shape  #[B, 15, 660, 10  ]

        # Lift feature
        x = self.theta1(x)              # [B, N, T, H]
        x = F.relu(x)

        # Spatial step 1 (dùng mean theo thời gian để tạo tín hiệu không gian)
        x_sp = self.spatial_gcn(x.mean(dim=2), adj)  # [B, N, H]
        x = x + x_sp.unsqueeze(2)                    # residual: [B, N, T, H]

        # Temporal conv 1 (chạy riêng cho từng node)
        x = x.permute(0, 1, 3, 2)                    # [B, N, H, T]
        x = self.temporal1(x.reshape(B * N, x.size(2), T))  # [B*N, H, T]
        x = F.relu(x).reshape(B, N, -1, T).permute(0, 1, 3, 2)  # [B, N, T, H]

        # Spatial step 2
        x_sp2 = self.spatial_gcn(x.mean(dim=2), adj)  # [B, N, H]
        x = x + x_sp2.unsqueeze(2)                    # [B, N, T, H]

        # Temporal conv 2
        x = x.permute(0, 1, 3, 2)                     # [B, N, H, T]
        x = self.temporal2(x.reshape(B * N, x.size(2), T))
        x = F.relu(x).reshape(B, N, -1, T).permute(0, 1, 3, 2)  # [B, N, T, H]

        # Global average pooling theo N và T
        x = x.mean(dim=1).mean(dim=1)  # [B, H]
        return self.cls(x)             # [B, num_classes]F
