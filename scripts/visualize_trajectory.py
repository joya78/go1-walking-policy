#!/usr/bin/env python3
"""Visualize robot trajectory and create simple animations."""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import argparse
import sys
import os

# Add project to path
sys.path.insert(0, '/home/maxime/my_go1_project')

def plot_training_metrics(log_file):
    """Plot training metrics from TensorBoard logs."""
    from tensorboard.backend.event_processing import event_accumulator
    
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Get scalar tags
    tags = ea.Tags()['scalars']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Go1 Training Metrics', fontsize=16)
    
    # Plot key metrics
    metrics = ['Train/mean_reward', 'Train/mean_episode_length', 'Loss/value_function', 'Loss/surrogate']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        if metric in tags:
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values)
            ax.set_title(metric.replace('/', ' - '))
            ax.set_xlabel('Iteration')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_robot_trajectory_2d(positions, orientations, save_path='trajectory.png'):
    """Plot 2D robot trajectory from logged positions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D trajectory (top view)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Path')
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='X', label='End', zorder=5)
    
    # Add robot orientation arrows every N steps
    step = max(1, len(positions) // 20)
    for i in range(0, len(positions), step):
        yaw = orientations[i, 2]  # Assuming [roll, pitch, yaw]
        dx = 0.3 * np.cos(yaw)
        dy = 0.3 * np.sin(yaw)
        ax1.arrow(positions[i, 0], positions[i, 1], dx, dy, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Robot Trajectory (Top View)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Height over time
    time = np.arange(len(positions)) * 0.02  # Assuming 50Hz
    ax2.plot(time, positions[:, 2], 'g-', linewidth=2)
    ax2.axhline(y=positions[0, 2], color='r', linestyle='--', alpha=0.5, label=f'Initial: {positions[0, 2]:.3f}m')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Height (m)', fontsize=12)
    ax2.set_title('Robot Height Over Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Trajectory plot saved to: {save_path}")
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Go1 training results')
    parser.add_argument('--log_dir', type=str, 
                       default='/home/maxime/IsaacLab-main/logs/rsl_rl/unitree_go1_rough/2025-11-16_17-17-58',
                       help='Path to training log directory')
    parser.add_argument('--output_dir', type=str, default='/home/maxime/my_go1_project/videos',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find TensorBoard event file
    event_files = [f for f in os.listdir(args.log_dir) if f.startswith('events.out.tfevents')]
    
    if event_files:
        log_file = os.path.join(args.log_dir, event_files[0])
        print(f"[INFO] Plotting training metrics from: {log_file}")
        
        fig = plot_training_metrics(log_file)
        output_path = os.path.join(args.output_dir, 'training_metrics.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Training metrics saved to: {output_path}")
        plt.close()
    else:
        print("[WARNING] No TensorBoard event files found")
    
    print("\n[INFO] To see live metrics, access TensorBoard at:")
    print("       http://localhost:6006 (after SSH tunnel: ssh -L 6006:localhost:6006 maxime@server)")

if __name__ == "__main__":
    main()
