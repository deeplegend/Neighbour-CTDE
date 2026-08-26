import os
import time
import subprocess
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing Configuration
TEST_STEPS = 1000
MODELS_DIR = "models"
ALGORITHMS = ["DQN", "D2QN", "D3QN", "MAPPO"]

def recompile_network(tls_type="static"):
    """Recompiles the SUMO network to switch between Static and Actuated controllers."""
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

def evaluate_rl_model(algorithm):
    """Loads frozen weights and tests the decentralized RL policy."""
    recompile_network("static") # Base network
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=TEST_STEPS)
    
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    
    if algorithm == "D3QN":
        actor = DuelingDQNActor(obs_dim, action_dim).to(device)
    elif algorithm == "MAPPO":
        actor = MAPPOActor(obs_dim, action_dim).to(device)
    else:
        actor = DQNActor(obs_dim, action_dim).to(device)
        
    weight_path = f"{MODELS_DIR}/{algorithm}_actor.pth"
    if not os.path.exists(weight_path):
        print(f"Weights for {algorithm} not found. Skipping...")
        return 0, 0, 0, 0
        
    actor.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
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
                if algorithm == "MAPPO":
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
    
    avg_inference_time = np.mean(inference_latencies)
    
    return (
        metrics["throughput"], 
        metrics["avg_delay"], 
        metrics["avg_ev_wait"], 
        avg_inference_time
    )

def plot_results(results):
    labels = list(results.keys())
    throughputs = [res[0] for res in results.values()]
    delays = [res[1] for res in results.values()]
    ev_waits = [res[2] for res in results.values()]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars_tp = ax1.bar(x, throughputs, width, label='System Throughput', color='#2ca02c')
    ax1.bar_label(bars_tp, fmt='%.1f', padding=3)
    
    ax1.set_ylabel('Total Vehicles Arrived')
    ax1.set_title('Throughput (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    bars_delay = ax2.bar(x - width/2, delays, width, label='Avg Congestion Delay', color='#1f77b4')
    ax2.bar_label(bars_delay, fmt='%.2f', padding=3)
    
    bars_ev = ax2.bar(x + width/2, ev_waits, width, label='Avg Emergency Vehicle Delay', color='#d62728')
    ax2.bar_label(bars_ev, fmt='%.2f', padding=3)
    
    ax2.set_ylabel('Delay Time (seconds)')
    ax2.set_title('Delays (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('CTDE Algorithms vs Traditional Controllers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("CTDE_Evaluation_Results.png", dpi=300, bbox_inches='tight')
    print("Saved comparison chart as 'CTDE_Evaluation_Results.png'")
    plt.show()

if __name__ == "__main__":
    print("=== Phase 5: Multi-Agent Benchmark Evaluation ===")
    results = {}
    
    print("\nEvaluating Webster's Fixed-Time (Static)...")
    results["Static"] = evaluate_traditional("static")
    
    for alg in ALGORITHMS:
        print(f"Evaluating {alg}...")
        res = evaluate_rl_model(alg)
        if res[0] > 0:
            results[alg] = res
            
    print("\n" + "="*80)
    print(f"{'Method':<12} | {'Throughput':<12} | {'Avg Delay':<12} | {'Avg EV Wait':<12} | {'Inference Latency'}")
    print("-" * 80)
    for name, (tp, delay, ev_wait, inf) in results.items():
        print(f"{name:<12} | {tp:<12.1f} | {delay:<12.2f} | {ev_wait:<12.2f} | {inf*1000:.4f} ms")
    print("="*80)
    
    if len(results) > 0:
        plot_results(results)