#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script to compare different ST-GCN variants
Tests: inference speed, memory usage, parameter count
"""

import time
import torch
import numpy as np
from model_stgcn import SimpleSTGCN
from model_stgcn_improved import create_stgcn

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, x, adj, device, num_runs=100):
    """Measure average inference time"""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x, adj)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_runs):
            _ = model(x, adj)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
    
    return elapsed / num_runs * 1000  # ms per inference

def measure_memory(model, x, adj, device):
    """Measure peak memory usage during forward pass"""
    if device != 'cuda':
        return 0.0
    
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    
    with torch.no_grad():
        _ = model(x, adj)
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return peak_mem

def test_model(model_type, model, x, adj, device):
    """Run all benchmarks for a model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}")
    print(f"{'='*60}")
    
    # Parameter count
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    # Inference time
    inf_time = measure_inference_time(model, x, adj, device)
    print(f"Inference time: {inf_time:.2f} ms")
    
    # Memory
    if device == 'cuda':
        mem = measure_memory(model, x, adj, device)
        print(f"Peak memory: {mem:.2f} MB")
    
    # Output shape
    model.eval()
    with torch.no_grad():
        out = model(x, adj)
    print(f"Output shape: {out.shape}")
    
    return {
        'model_type': model_type,
        'parameters': params,
        'inference_time_ms': inf_time,
        'peak_memory_mb': mem if device == 'cuda' else 0,
        'output_shape': out.shape
    }

def main():
    print("ST-GCN Model Benchmark")
    print("="*60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Data dimensions (typical for your dataset)
    B, N, T, F = 8, 15, 300, 10
    num_classes = 4
    
    print(f"Input shape: [B={B}, N={N}, T={T}, F={F}]")
    print(f"Number of classes: {num_classes}")
    
    # Create test data
    x = torch.randn(B, N, T, F).to(device)
    A = torch.randn(N, N).to(device)
    
    # Test models
    results = []
    
    # 1. SimpleSTGCN
    print("\n" + "="*60)
    print("1. SimpleSTGCN (Original)")
    model_simple = SimpleSTGCN(in_feats=F, hid=128, num_classes=num_classes).to(device)
    
    # For SimpleSTGCN, need normalized [B, N, N] adjacency
    if A.dim() == 2:
        adj_simple = A.unsqueeze(0).expand(B, -1, -1)
    adj_simple = adj_simple + torch.eye(N, device=device).unsqueeze(0)
    adj_simple = adj_simple / (adj_simple.sum(dim=-1, keepdim=True).clamp_min(1.0))
    
    result = test_model('SimpleSTGCN', model_simple, x, adj_simple, device)
    results.append(result)
    
    # 2. HybridSTGCN
    print("\n" + "="*60)
    print("2. HybridSTGCN (Recommended)")
    model_hybrid = create_stgcn(
        model_type='hybrid',
        in_channels=F,
        num_classes=num_classes,
        num_nodes=N,
        hid=128,
        dropout=0.3
    ).to(device)
    result = test_model('HybridSTGCN', model_hybrid, x, A, device)
    results.append(result)
    
    # 3. ImprovedSTGCN
    print("\n" + "="*60)
    print("3. ImprovedSTGCN (Full)")
    model_improved = create_stgcn(
        model_type='full',
        in_channels=F,
        num_classes=num_classes,
        num_nodes=N,
        edge_importance_weighting=True,
        dropout=0.5
    ).to(device)
    result = test_model('ImprovedSTGCN', model_improved, x, A, device)
    results.append(result)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Params':<12} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-"*60)
    
    for r in results:
        print(f"{r['model_type']:<20} "
              f"{r['parameters']:>11,} "
              f"{r['inference_time_ms']:>11.2f} "
              f"{r['peak_memory_mb']:>11.2f}")
    
    # Relative comparison (normalized to SimpleSTGCN)
    print("\n" + "="*60)
    print("RELATIVE TO SimpleSTGCN (baseline = 1.0x)")
    print("="*60)
    
    baseline = results[0]
    print(f"{'Model':<20} {'Params':<12} {'Speed':<12} {'Memory':<12}")
    print("-"*60)
    
    for r in results:
        param_ratio = r['parameters'] / baseline['parameters']
        time_ratio = r['inference_time_ms'] / baseline['inference_time_ms']
        mem_ratio = r['peak_memory_mb'] / baseline['peak_memory_mb'] if baseline['peak_memory_mb'] > 0 else 0
        
        print(f"{r['model_type']:<20} "
              f"{param_ratio:>11.2f}x "
              f"{time_ratio:>11.2f}x "
              f"{mem_ratio:>11.2f}x")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Dataset Size         | Recommended Model    | Reason
---------------------|---------------------|---------------------------
< 500 samples        | SimpleSTGCN         | Fast, low overfitting risk
500 - 2000 samples   | HybridSTGCN         | Best balance
> 2000 samples       | ImprovedSTGCN       | Full capacity, best accuracy
    
Hardware             | Recommended Model    | Reason
---------------------|---------------------|---------------------------
CPU only             | SimpleSTGCN         | Fastest inference
GPU (< 4GB VRAM)     | HybridSTGCN         | Moderate memory
GPU (>= 4GB VRAM)    | ImprovedSTGCN       | Full performance

Training Time        | Recommended Model    | Reason
---------------------|---------------------|---------------------------
Quick experiments    | SimpleSTGCN         | 2-3 min/epoch
Production           | HybridSTGCN         | 5-7 min/epoch
Research             | ImprovedSTGCN       | 10-15 min/epoch
    """)
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()
