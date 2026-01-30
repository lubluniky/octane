#!/usr/bin/env python3
"""Generate CPU vs Metal GPU benchmark comparison charts."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['xtick.color'] = '#eee'
plt.rcParams['ytick.color'] = '#eee'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['font.family'] = 'monospace'

# Benchmark data from actual runs
matmul_sizes = [128, 512, 1024, 2048]
matmul_cpu = [68.79, 739.96, 4825.6, 40859]  # µs
matmul_metal = [4.89, 92.82, 733.59, 5966.7]  # µs

softmax_configs = ['64x512', '256x1024', '512x2048', '1024x4096']
softmax_cpu = [123.71, 966.77, 3827.1, 15494]  # µs
softmax_metal = [53.24, 354.97, 1630.2, 7043.8]  # µs

inference_batch = [512, 1024, 2048, 4096]
inference_cpu = [986.19, 1780.7, 3018.8, 6355.2]  # µs
inference_metal = [120.96, 224.97, 494.18, 1015.8]  # µs

mlp_cpu = 486.95  # µs
mlp_metal = 134.03  # µs

output_dir = os.path.dirname(os.path.abspath(__file__))
media_dir = os.path.join(os.path.dirname(output_dir), 'media')
os.makedirs(media_dir, exist_ok=True)

def create_matmul_chart():
    """Create matmul comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(matmul_sizes))
    width = 0.35

    # Time comparison
    bars1 = ax1.bar(x - width/2, matmul_cpu, width, label='CPU', color='#e94560', alpha=0.8)
    bars2 = ax1.bar(x + width/2, matmul_metal, width, label='Metal GPU', color='#00d4ff', alpha=0.8)

    ax1.set_xlabel('Matrix Size (NxN)', fontsize=12)
    ax1.set_ylabel('Time (µs)', fontsize=12)
    ax1.set_title('Matrix Multiplication: CPU vs Metal GPU', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}x{s}' for s in matmul_sizes])
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Speedup
    speedups = [c/m for c, m in zip(matmul_cpu, matmul_metal)]
    bars3 = ax2.bar(x, speedups, color='#00ff88', alpha=0.8)
    ax2.axhline(y=1, color='#e94560', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Matrix Size (NxN)', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Metal GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s}x{s}' for s in matmul_sizes])
    ax2.grid(True, alpha=0.3)

    # Add speedup labels
    for bar, speedup in zip(bars3, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, 'matmul_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"Saved: {os.path.join(media_dir, 'matmul_comparison.png')}")

def create_inference_chart():
    """Create inference comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(inference_batch))
    width = 0.35

    # Time comparison
    bars1 = ax1.bar(x - width/2, inference_cpu, width, label='CPU', color='#e94560', alpha=0.8)
    bars2 = ax1.bar(x + width/2, inference_metal, width, label='Metal GPU', color='#00d4ff', alpha=0.8)

    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Time (µs)', fontsize=12)
    ax1.set_title('Policy Network Inference: CPU vs Metal GPU', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(inference_batch)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Speedup
    speedups = [c/m for c, m in zip(inference_cpu, inference_metal)]
    bars3 = ax2.bar(x, speedups, color='#00ff88', alpha=0.8)
    ax2.axhline(y=1, color='#e94560', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Metal GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(inference_batch)
    ax2.grid(True, alpha=0.3)

    # Add speedup labels
    for bar, speedup in zip(bars3, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, 'inference_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"Saved: {os.path.join(media_dir, 'inference_comparison.png')}")

def create_combined_speedup_chart():
    """Create combined speedup overview chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Categories and speedups
    categories = [
        'MatMul\n128x128', 'MatMul\n512x512', 'MatMul\n1024x1024', 'MatMul\n2048x2048',
        'Inference\nBatch 512', 'Inference\nBatch 1024', 'Inference\nBatch 2048', 'Inference\nBatch 4096',
        'MLP\nForward'
    ]

    speedups = [
        matmul_cpu[0]/matmul_metal[0], matmul_cpu[1]/matmul_metal[1],
        matmul_cpu[2]/matmul_metal[2], matmul_cpu[3]/matmul_metal[3],
        inference_cpu[0]/inference_metal[0], inference_cpu[1]/inference_metal[1],
        inference_cpu[2]/inference_metal[2], inference_cpu[3]/inference_metal[3],
        mlp_cpu/mlp_metal
    ]

    colors = ['#e94560'] * 4 + ['#00d4ff'] * 4 + ['#00ff88']

    x = np.arange(len(categories))
    bars = ax.bar(x, speedups, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    ax.axhline(y=1, color='#ffffff', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Operation', fontsize=12)
    ax.set_ylabel('Speedup (x faster than CPU)', fontsize=12)
    ax.set_title('Metal GPU Acceleration on Apple Silicon (M-series)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e94560', alpha=0.8, label='Matrix Multiplication'),
        Patch(facecolor='#00d4ff', alpha=0.8, label='Policy Inference'),
        Patch(facecolor='#00ff88', alpha=0.8, label='MLP Forward Pass')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, 'metal_speedup_overview.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"Saved: {os.path.join(media_dir, 'metal_speedup_overview.png')}")

def create_softmax_chart():
    """Create softmax comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(softmax_configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, softmax_cpu, width, label='CPU', color='#e94560', alpha=0.8)
    bars2 = ax.bar(x + width/2, softmax_metal, width, label='Metal GPU', color='#00d4ff', alpha=0.8)

    ax.set_xlabel('Tensor Shape (Batch x Features)', fontsize=12)
    ax.set_ylabel('Time (µs)', fontsize=12)
    ax.set_title('Softmax Operation: CPU vs Metal GPU', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(softmax_configs)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add speedup annotations
    for i, (c, m) in enumerate(zip(softmax_cpu, softmax_metal)):
        speedup = c / m
        ax.annotate(f'{speedup:.1f}x', xy=(i, max(c, m) + 500), ha='center', fontsize=10,
                   color='#00ff88', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, 'softmax_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"Saved: {os.path.join(media_dir, 'softmax_comparison.png')}")

if __name__ == '__main__':
    print("Generating GPU benchmark charts...")
    create_matmul_chart()
    create_inference_chart()
    create_combined_speedup_chart()
    create_softmax_chart()
    print("\nAll charts generated successfully!")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY: Metal GPU vs CPU")
    print("="*60)
    print("\nMatrix Multiplication Speedups:")
    for size, cpu, metal in zip(matmul_sizes, matmul_cpu, matmul_metal):
        print(f"  {size:4d}x{size:<4d}: {cpu/metal:5.1f}x faster ({cpu:.1f}µs → {metal:.1f}µs)")

    print("\nPolicy Inference Speedups:")
    for batch, cpu, metal in zip(inference_batch, inference_cpu, inference_metal):
        print(f"  Batch {batch:4d}: {cpu/metal:5.1f}x faster ({cpu:.1f}µs → {metal:.1f}µs)")

    print(f"\nMLP Forward Pass:")
    print(f"  256x128→512→64: {mlp_cpu/mlp_metal:.1f}x faster ({mlp_cpu:.1f}µs → {mlp_metal:.1f}µs)")
    print("="*60)
