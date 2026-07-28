#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize learned edge importance from trained ST-GCN models
Shows which AU connections the model learned are important
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

# AU node names (from b2_video_normalization.py)
AU_NODES = [
    "brow_left_inner", "brow_left_outer", "brow_right_inner", "brow_right_outer",
    "eye_left_upper", "eye_left_lower", "eye_right_upper", "eye_right_lower",
    "nose_bridge", "nose_alar_left", "nose_alar_right",
    "upper_lip", "lower_lip", "lip_corners", "chin_center",
]

def shorten_names(names):
    """Shorten AU names for better display"""
    mapping = {
        "brow_left_inner": "BL_in",
        "brow_left_outer": "BL_out",
        "brow_right_inner": "BR_in",
        "brow_right_outer": "BR_out",
        "eye_left_upper": "EL_up",
        "eye_left_lower": "EL_lo",
        "eye_right_upper": "ER_up",
        "eye_right_lower": "ER_lo",
        "nose_bridge": "Nose_br",
        "nose_alar_left": "Nose_L",
        "nose_alar_right": "Nose_R",
        "upper_lip": "Lip_up",
        "lower_lip": "Lip_lo",
        "lip_corners": "Lip_cor",
        "chin_center": "Chin",
    }
    return [mapping.get(n, n) for n in names]

def load_model_edge_importance(model_path, model_type='hybrid'):
    """Load edge importance weights from trained model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract edge_importance parameters
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Find edge_importance parameters
    edge_weights = []
    for key, value in state_dict.items():
        if 'edge_importance' in key:
            edge_weights.append(value.cpu().numpy())
    
    if len(edge_weights) == 0:
        print("⚠️  No edge_importance parameters found. Model may not have learned edge weights.")
        return None
    
    return edge_weights

def plot_edge_importance(edge_weights, layer_idx, au_nodes, save_path=None):
    """Plot edge importance matrix for a specific layer"""
    W = edge_weights[layer_idx]
    N = W.shape[0]
    
    # Shorten names
    short_names = shorten_names(au_nodes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot heatmap
    im = ax.imshow(W, cmap='hot', interpolation='nearest', aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Importance', rotation=270, labelpad=20)
    
    # Ticks
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    
    # Title
    ax.set_title(f'Learned Edge Importance - Layer {layer_idx+1}', fontsize=12, pad=20)
    ax.set_xlabel('Target AU Node', fontsize=10)
    ax.set_ylabel('Source AU Node', fontsize=10)
    
    # Grid
    ax.set_xticks(np.arange(N+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Annotate strong connections (top 20%)
    threshold = np.percentile(W.flatten(), 80)
    for i in range(N):
        for j in range(N):
            if W[i, j] > threshold and i != j:  # Exclude diagonal
                text = ax.text(j, i, f'{W[i,j]:.2f}',
                             ha="center", va="center", color="white", fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_all_layers(edge_weights, au_nodes, save_dir='edge_importance_viz'):
    """Plot edge importance for all layers"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    num_layers = len(edge_weights)
    print(f"Found {num_layers} ST-GCN layers with edge importance")
    
    for i in range(num_layers):
        save_path = save_dir / f'layer_{i+1}_edge_importance.png'
        plot_edge_importance(edge_weights, i, au_nodes, save_path)

def plot_edge_evolution(edge_weights, au_nodes, save_path='edge_importance_evolution.png'):
    """Plot how edge importance evolves across layers"""
    num_layers = len(edge_weights)
    N = edge_weights[0].shape[0]
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    short_names = shorten_names(au_nodes)
    
    for i, (ax, W) in enumerate(zip(axes, edge_weights)):
        im = ax.imshow(W, cmap='hot', interpolation='nearest')
        ax.set_title(f'Layer {i+1}', fontsize=10)
        
        if i == 0:
            ax.set_yticks(np.arange(N))
            ax.set_yticklabels(short_names, fontsize=6)
            ax.set_ylabel('Source AU', fontsize=8)
        else:
            ax.set_yticks([])
        
        ax.set_xticks(np.arange(N))
        ax.set_xticklabels(short_names, rotation=90, fontsize=6)
        ax.set_xlabel('Target AU', fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Edge Importance Evolution Across Layers', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {save_path}")
    plt.close()

def analyze_top_connections(edge_weights, au_nodes, top_k=10):
    """Analyze top-k most important connections across all layers"""
    print("\n" + "="*60)
    print(f"TOP {top_k} MOST IMPORTANT AU CONNECTIONS (Across All Layers)")
    print("="*60)
    
    all_edges = []
    for layer_idx, W in enumerate(edge_weights):
        N = W.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j:  # Exclude self-connections
                    all_edges.append({
                        'layer': layer_idx + 1,
                        'source': au_nodes[i],
                        'target': au_nodes[j],
                        'weight': W[i, j]
                    })
    
    # Sort by weight
    all_edges.sort(key=lambda x: x['weight'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Layer':<8} {'Connection':<50} {'Weight':<10}")
    print("-"*80)
    
    for rank, edge in enumerate(all_edges[:top_k], 1):
        connection = f"{edge['source']:20} → {edge['target']:20}"
        print(f"{rank:<6} {edge['layer']:<8} {connection:<50} {edge['weight']:.4f}")
    
    # Analyze patterns
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)
    
    # Most important source nodes
    source_counts = {}
    for edge in all_edges[:30]:  # Top 30 edges
        src = edge['source']
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print("\nMost influential AU nodes (appear most in top connections):")
    for node, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {node}: {count} times")

def compare_with_base_adjacency(edge_weights, base_adj_path='au_adjacency.npy'):
    """Compare learned weights with base adjacency structure"""
    if not Path(base_adj_path).exists():
        print(f"⚠️  Base adjacency not found: {base_adj_path}")
        return
    
    A_base = np.load(base_adj_path)
    
    print("\n" + "="*60)
    print("COMPARISON WITH BASE ADJACENCY")
    print("="*60)
    
    for i, W_learned in enumerate(edge_weights):
        # Compare where base has edges (A_base[i,j] = 1)
        base_edges = (A_base > 0) & (np.eye(A_base.shape[0]) == 0)  # Exclude diagonal
        
        learned_on_base = W_learned[base_edges].mean()
        learned_off_base = W_learned[~base_edges].mean()
        
        print(f"\nLayer {i+1}:")
        print(f"  Avg weight on base edges: {learned_on_base:.4f}")
        print(f"  Avg weight on new edges:  {learned_off_base:.4f}")
        print(f"  Ratio: {learned_on_base / (learned_off_base + 1e-8):.2f}x")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize ST-GCN edge importance")
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pt file)')
    parser.add_argument('--model-type', type=str, default='hybrid',
                       choices=['hybrid', 'full'],
                       help='Model type')
    parser.add_argument('--output-dir', type=str, default='edge_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--base-adj', type=str, default=None,
                       help='Path to base adjacency matrix (.npy)')
    args = parser.parse_args()
    
    print("="*60)
    print("ST-GCN Edge Importance Visualization")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Output: {args.output_dir}")
    
    # Load edge weights
    edge_weights = load_model_edge_importance(args.model, args.model_type)
    
    if edge_weights is None:
        print("❌ No edge importance found. Exiting.")
        return
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Individual layer plots
    print("\n📊 Generating individual layer plots...")
    plot_all_layers(edge_weights, AU_NODES, output_dir)
    
    # 2. Evolution across layers
    print("\n📊 Generating evolution plot...")
    plot_edge_evolution(edge_weights, AU_NODES, 
                       output_dir / 'edge_evolution.png')
    
    # 3. Analyze top connections
    analyze_top_connections(edge_weights, AU_NODES, top_k=15)
    
    # 4. Compare with base adjacency (if provided)
    if args.base_adj:
        compare_with_base_adjacency(edge_weights, args.base_adj)
    
    print("\n✅ Visualization complete!")
    print(f"Check outputs in: {output_dir}")

if __name__ == "__main__":
    # For quick testing without command line args
    import sys
    if len(sys.argv) == 1:
        print("Usage: python visualize_edge_importance.py --model <path_to_model.pt>")
        print("\nExample:")
        print("  python visualize_edge_importance.py --model stgcn_hybrid_best.pt --model-type hybrid")
    else:
        main()
