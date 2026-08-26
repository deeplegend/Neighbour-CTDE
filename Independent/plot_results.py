import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 1. Define the directories for your three algorithms
log_dirs = {
    "I-D2QN": "runs/INDEPENDENT_I-D2QN",
    "I-D3QN": "runs/INDEPENDENT_I-D3QN",
    "I-DQN": "runs/INDEPENDENT_I-DQN",
    "IPPO": "runs/INDEPENDENT_IPPO"
}

# 2. Define the exact metric tags used in TensorBoard
REWARD_TAG = 'Reward/Team_Reward' 
LOSS_TAG = 'Loss/Q_Loss'

def get_tensorboard_data(log_dir, tag):
    """Extracts data from a TensorBoard event file for a specific tag."""
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
    """Applies exponential moving average smoothing similar to TensorBoard."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_metric(metric_tag, title, ylabel, filename, apply_smoothing=True):
    """Plots and saves a combined graph for the given metric."""
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    all_data = {}
    max_dqn_step = 0
    
    # First pass: Collect data and find the maximum step for the DQN variants
    for label, log_dir in log_dirs.items():
        steps, values = get_tensorboard_data(log_dir, metric_tag)
        if steps and values:
            all_data[label] = (steps, values)
            # Track the max step of non-IPPO methods to use as our normalization target
            if label != "IPPO":
                max_dqn_step = max(max_dqn_step, max(steps))
                
    # Second pass: Process and plot the data
    for label, (steps, values) in all_data.items():
        plot_steps = steps
        
        # Normalize IPPO's x-axis to perfectly match the DQN range
        if label == "IPPO" and max_dqn_step > 0:
            max_ippo_step = max(steps)
            plot_steps = [(s / max_ippo_step) * max_dqn_step for s in steps]
            
        if apply_smoothing:
            smoothed_values = smooth_data(values, weight=0.95)
            # Plot solid smoothed data first to grab the assigned color
            line = plt.plot(plot_steps, smoothed_values, label=label, linewidth=2.5)[0]
            # Plot transparent raw data in the background matching the smoothed line's color
            plt.plot(plot_steps, values, color=line.get_color(), alpha=0.15, linewidth=1)
        else:
            # Plot only the raw data with higher opacity
            plt.plot(plot_steps, values, label=label, linewidth=1.5, alpha=0.85)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

# 3. Generate the plots
if __name__ == "__main__":
    # Loss Plot
    plot_metric(LOSS_TAG, 'Training Loss over Time', 'Loss', 'combined_loss_plot.png', apply_smoothing=True)
    
    # Reward Plots (Separated)
    plot_metric(REWARD_TAG, 'Episode Reward over Time (Smoothed)', 'Total Reward', 'reward_plot_smoothed.png', apply_smoothing=True)
    plot_metric(REWARD_TAG, 'Episode Reward over Time (Raw)', 'Total Reward', 'reward_plot_raw.png', apply_smoothing=False)