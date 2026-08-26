import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = {
    "CTDE-D2QN": "runs/CTDE_D2QN",
    "CTDE-D3QN": "runs/CTDE_D3QN",
    "CTDE-DQN": "runs/CTDE_DQN",
    "CTDE-MAPPO": "runs/CTDE_MAPPO"
}

REWARD_TAG = 'Reward/Team_Reward'
LOSS_TAG = 'Loss/Q_Tot_Huber_Loss'

def get_tensorboard_data(log_dir, tag):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    if tag not in event_acc.Tags()['scalars']:
        print(f"Warning: Tag '{tag}' not found in {log_dir}")
        return [], []
        
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def smooth_data(scalars, weight=0.95):
    """Increased default weight to 0.95 for smoother trend lines."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metric(metric_tag, title, ylabel, filename, apply_smoothing=True):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    all_data = {}
    
    # First pass: collect all raw data
    for label, log_dir in log_dirs.items():
        steps, values = get_tensorboard_data(log_dir, metric_tag)
        if steps and values:
            all_data[label] = (steps, values)
            
    # Second pass: process and plot scaled data
    for label, (steps, values) in all_data.items():
        max_step_for_this_method = max(steps)
        
        # Scale steps to a percentage (0 to 100)
        scaled_steps = [(s / max_step_for_this_method) * 100 for s in steps]
        
        if apply_smoothing:
            smoothed_values = smooth_data(values, weight=0.95)
            # 1. Plot the thick, smoothed foreground line and grab its assigned color
            line = plt.plot(scaled_steps, smoothed_values, label=label, linewidth=2.5)[0]
            # 2. Plot the raw, jagged data in the background using the same color and high transparency
            plt.plot(scaled_steps, values, color=line.get_color(), alpha=0.15, linewidth=1)
        else:
            # Plot only the raw data with solid opacity
            plt.plot(scaled_steps, values, label=label, linewidth=1.5, alpha=0.85)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Training Progress (%)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # Loss Plot (Smoothed)
    plot_metric(LOSS_TAG, 'Training Loss over Time (CTDE Methods)', 'Loss', 'ctde_loss_plot.png', apply_smoothing=True)
    
    # Reward Plots (Separated)
    plot_metric(REWARD_TAG, 'Episode Reward over Time (Smoothed)', 'Total Reward', 'ctde_reward_plot_smoothed.png', apply_smoothing=True)
    plot_metric(REWARD_TAG, 'Episode Reward over Time (Raw)', 'Total Reward', 'ctde_reward_plot_raw.png', apply_smoothing=False)