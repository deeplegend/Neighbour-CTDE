import os
import time
import subprocess
import traceback
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor

# =====================================================================
# HARDWARE SETUP (DirectML for AMD / CUDA for NVIDIA / CPU Fallback)
# =====================================================================
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using AMD GPU (DirectML): {torch_directml.device_name(0)}")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")

# Testing Configuration
TEST_STEPS = 1000
MODELS_DIR = "models"

# Evaluates both CTDE / Independent variants
ALGORITHMS = [
    "I-DQN", "I-D2QN", "I-D3QN", "IPPO"
]

def recompile_network(tls_type="static"):
    print(f"Recompiling network with {tls_type} controllers...")
    cmd = [
        "netconvert",
        "--node-files=sumo_env/square_grid.nod.xml",
        "--edge-files=sumo_env/square_grid.edg.xml",
        "--connection-files=sumo_env/square_grid.con.xml",
        "--output-file=sumo_env/square_grid.net.xml",
        "--tls.join", "true",
        "--tls.default-type", tls_type
    ]
    subprocess.run(cmd, capture_output=True)

def evaluate_traditional(controller_type="static"):
    """Runs a baseline traditional controller (no RL)."""
    recompile_network(controller_type)
    
    binary = checkBinary('sumo')
    cmd = [binary, "-c", "sumo_env/square_grid.sumocfg", "--no-step-log", "true"]
    traci.start(cmd)
    
    throughput = 0
    cumulative_delay = 0
    ev_cumulative_wait = 0
    ev_count = 0
    
    start_time = time.time()
    for step in range(TEST_STEPS):
        traci.simulationStep()
        throughput += traci.simulation.getArrivedNumber()
        
        for veh_id in traci.vehicle.getIDList():
            w = traci.vehicle.getWaitingTime(veh_id)
            cumulative_delay += w
            if "emergency" in traci.vehicle.getTypeID(veh_id):
                ev_cumulative_wait += w
                ev_count += 1
                
    inference_time = (time.time() - start_time) / TEST_STEPS
    traci.close()
    
    avg_ev_wait = ev_cumulative_wait / max(1, ev_count)
    avg_delay = cumulative_delay / TEST_STEPS
    
    return throughput, avg_delay, avg_ev_wait, inference_time

def find_weight_file(algorithm):
    candidates = [
        f"{MODELS_DIR}/{algorithm}_actor.pth",
        f"{MODELS_DIR}/{algorithm}.pth",
        f"{MODELS_DIR}/{algorithm.replace('I-', '')}_actor.pth",
        f"{MODELS_DIR}/{algorithm.replace('IPPO', 'MAPPO')}_actor.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def evaluate_rl_model(algorithm):
    try:
        weight_path = find_weight_file(algorithm)
        if not weight_path:
            print(f"  [!] Weights for {algorithm} not found in '{MODELS_DIR}/'. Skipping...")
            return 0, 0, 0, 0

        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
        fc1_weight_shape = checkpoint["fc1.weight"].shape
        loaded_obs_dim = fc1_weight_shape[1]
        
        max_neighbors = (loaded_obs_dim // 14) - 1
        
        env = MultiIntersectionSUMOEnv(
            use_gui=False, 
            max_steps=TEST_STEPS, 
            max_neighbors=max(0, max_neighbors)
        )
        
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        
        if "D3QN" in algorithm:
            actor = DuelingDQNActor(obs_dim, action_dim).to(device)
        elif "PPO" in algorithm:
            actor = MAPPOActor(obs_dim, action_dim).to(device)
        else:
            actor = DQNActor(obs_dim, action_dim).to(device)
            
        actor.load_state_dict(checkpoint)
        actor.eval()
        
        obs_dict, _ = env.reset()
        done = False
        inference_latencies = []
        
        while not done:
            actions = {}
            t0 = time.time()
            with torch.no_grad():
                for a in env.agents:
                    obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                    if "PPO" in algorithm:
                        action, _, _ = actor.get_action(obs_tensor)
                        actions[a] = action.item()
                    else:
                        q_vals = actor(obs_tensor)
                        actions[a] = q_vals.argmax(dim=1).item()
            inference_latencies.append(time.time() - t0)
            
            next_obs_dict, _, terms, _, _ = env.step(actions)
            done = all(terms.values())
            obs_dict = next_obs_dict
            
        metrics = env.get_episode_metrics()
        env.close()
        
        avg_inference_time = np.mean(inference_latencies) if inference_latencies else 0.0
        
        return metrics["throughput"], metrics["avg_delay"], metrics["avg_ev_wait"], avg_inference_time

    except Exception as e:
        print(f"  [ERROR] Failed during evaluation of {algorithm}: {e}")
        traceback.print_exc()
        try:
            traci.close()
        except Exception:
            pass
        return 0, 0, 0, 0

def plot_results(results):
    labels = list(results.keys())
    throughputs = [res[0] for res in results.values()]
    delays = [res[1] for res in results.values()]
    ev_waits = [res[2] for res in results.values()]
    
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars_tp = ax.bar(x - width, throughputs, width, label='System Throughput (Vehicles)', color='#2ca02c')
    ax.bar_label(bars_tp, fmt='%.1f', padding=3)
    
    bars_delay = ax.bar(x, delays, width, label='Avg Congestion Delay (s)', color='#1f77b4')
    ax.bar_label(bars_delay, fmt='%.2f', padding=3)
    
    bars_ev = ax.bar(x + width, ev_waits, width, label='Avg EV Delay (s)', color='#d62728')
    ax.bar_label(bars_ev, fmt='%.2f', padding=3)

    ax.set_ylabel('Performance Metric Values')
    ax.set_title('Comparative Benchmark: Traditional vs. Multi-Agent Controllers', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig("CTDE_Evaluation_Results.png", dpi=300)
    print("=> Saved comparison chart as 'CTDE_Evaluation_Results.png'")
    plt.show()

if __name__ == "__main__":
    print("==================================================")
    print("   MULTI-AGENT TRAFFIC BENCHMARK EVALUATION       ")
    print("==================================================")
    results = {}
    
    print("\nEvaluating Webster's Fixed-Time (Static)...")
    results["Static"] = evaluate_traditional("static")
    
    tested_names = set()
    for alg in ALGORITHMS:
        if alg in tested_names:
            continue
        print(f"\nEvaluating {alg}...")
        res = evaluate_rl_model(alg)
        if res[0] > 0 or res[1] > 0:  
            results[alg] = res
            tested_names.add(alg)
            
    print("\n" + "=" * 95)
    print(f"{'Method':<16} | {'Throughput':<14} | {'Avg Delay (s)':<16} | {'Avg EV Wait (s)':<16} | {'Inference Latency'}")
    print("-" * 95)
    for name, (tp, delay, ev_wait, inf) in results.items():
        print(f"{name:<16} | {tp:<14.1f} | {delay:<16.2f} | {ev_wait:<16.2f} | {inf*1000:.4f} ms")
    print("=" * 95)
    
    if len(results) > 1:
        plot_results(results)