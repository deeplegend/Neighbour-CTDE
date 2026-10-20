import csv
import os
import time
import subprocess
from pathlib import Path
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor
from sumo_env.generate_traffic import generate_routefile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Testing Configuration
TEST_STEPS = 1000
EVALUATION_SEEDS = (1001, 1002, 1003, 1004, 1005)
MODELS_DIR = "models_5x5"
ALGORITHMS = ["DQN", "D2QN", "D3QN", "MAPPO"]
METRIC_NAMES = ("Throughput", "Avg Delay", "Avg EV Wait", "Inference Time")

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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"netconvert failed while preparing the {tls_type} network.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

def evaluate_traditional(controller_type="static", seed=42, route_file=None):
    """Runs a baseline traditional controller (no RL) on one seeded scenario."""
    recompile_network(controller_type)
    
    binary = checkBinary('sumo')
    cmd = [binary, "-c", "sumo_env/square_grid.sumocfg", "--no-step-log", "true"]
    cmd.extend(["--seed", str(seed)])
    if route_file is not None:
        cmd.extend(["--route-files", str(route_file)])

    throughput = 0
    cumulative_delay = 0
    ev_cumulative_wait = 0
    ev_count = 0
    
    traci.start(cmd)
    start_time = time.time()
    try:
        for _ in range(TEST_STEPS):
            traci.simulationStep()
            throughput += traci.simulation.getArrivedNumber()

            for veh_id in traci.vehicle.getIDList():
                w = traci.vehicle.getWaitingTime(veh_id)
                cumulative_delay += w
                if "emergency" in traci.vehicle.getTypeID(veh_id):
                    ev_cumulative_wait += w
                    ev_count += 1
    finally:
        inference_time = (time.time() - start_time) / TEST_STEPS
        traci.close()
    
    avg_ev_wait = ev_cumulative_wait / max(1, ev_count)
    avg_delay = cumulative_delay / TEST_STEPS
    
    return throughput, avg_delay, avg_ev_wait, inference_time

def evaluate_rl_model(algorithm, seed, route_file):
    """Loads frozen weights and tests the decentralized RL policy on one scenario."""
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
        raise FileNotFoundError(f"Weights for {algorithm} not found: {weight_path}")
        
    actor.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    actor.eval()
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs_dict, _ = env.reset(seed=seed, route_file=str(route_file))
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

def summarize_results(results):
    summary = {}
    for method, runs in results.items():
        values = np.asarray(runs, dtype=float)
        summary[method] = {
            "mean": np.mean(values, axis=0),
            "variance": np.var(values, axis=0, ddof=1),
        }
    return summary


def save_results(results, summary):
    raw_path = Path("CTDE_Evaluation_Raw.csv")
    summary_path = Path("CTDE_Evaluation_Summary.csv")

    with raw_path.open("w", newline="", encoding="utf-8") as raw_file:
        writer = csv.writer(raw_file)
        writer.writerow(("Method", "Seed", *METRIC_NAMES))
        for method, runs in results.items():
            for seed, metrics in zip(EVALUATION_SEEDS, runs):
                writer.writerow((method, seed, *metrics))

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        summary_headers = ["Method", "Seed Count"]
        for metric_name in METRIC_NAMES:
            summary_headers.extend((f"{metric_name} Mean", f"{metric_name} Variance"))
        writer.writerow(summary_headers)
        for method, values in summary.items():
            row = [method, len(results[method])]
            for mean, variance in zip(values["mean"], values["variance"]):
                row.extend((mean, variance))
            writer.writerow(row)
    return raw_path, summary_path


def plot_results(summary):
    labels = list(summary)
    means = [summary[name]["mean"] for name in labels]
    standard_deviations = [np.sqrt(summary[name]["variance"]) for name in labels]
    throughputs = [res[0] for res in means]
    delays = [res[1] for res in means]
    ev_waits = [res[2] for res in means]
    throughput_errors = [res[0] for res in standard_deviations]
    delay_errors = [res[1] for res in standard_deviations]
    ev_wait_errors = [res[2] for res in standard_deviations]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars_tp = ax1.bar(
        x, throughputs, width, yerr=throughput_errors, capsize=4,
        label='System Throughput', color='#2ca02c'
    )
    ax1.bar_label(bars_tp, fmt='%.1f', padding=3)
    
    ax1.set_ylabel('Total Vehicles Arrived')
    ax1.set_title('Throughput (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    bars_delay = ax2.bar(
        x - width/2, delays, width, yerr=delay_errors, capsize=4,
        label='Avg Congestion Delay', color='#1f77b4'
    )
    ax2.bar_label(bars_delay, fmt='%.2f', padding=3)
    
    bars_ev = ax2.bar(
        x + width/2, ev_waits, width, yerr=ev_wait_errors, capsize=4,
        label='Avg Emergency Vehicle Delay', color='#d62728'
    )
    ax2.bar_label(bars_ev, fmt='%.2f', padding=3)
    
    ax2.set_ylabel('Delay Time (seconds)')
    ax2.set_title('Delays (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('CTDE Evaluation: Mean ± Standard Deviation (5 Seeds)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("CTDE_Evaluation_Results.png", dpi=300, bbox_inches='tight')
    print("Saved comparison chart as 'CTDE_Evaluation_Results.png'")
    plt.show()

if __name__ == "__main__":
    print("=== Phase 5: Multi-Agent Benchmark Evaluation ===")
    results = {}
    route_dir = Path("sumo_env/eval_routes")
    route_dir.mkdir(parents=True, exist_ok=True)
    route_files = {}
    for seed in EVALUATION_SEEDS:
        route_file = route_dir / f"routes_seed_{seed}.rou.xml"
        generate_routefile(
            filename=str(route_file),
            max_steps=TEST_STEPS,
            grid_size=5,
            seed=seed,
        )
        route_files[seed] = route_file
    
    print(f"\nEvaluation seeds: {list(EVALUATION_SEEDS)}")
    results["Static"] = []
    for seed in EVALUATION_SEEDS:
        print(f"\nEvaluating Webster's Fixed-Time (Static), seed {seed}...")
        results["Static"].append(
            evaluate_traditional("static", seed, route_files[seed])
        )
    
    for alg in ALGORITHMS:
        weight_path = Path(MODELS_DIR) / f"{alg}_actor.pth"
        if not weight_path.is_file():
            print(f"Weights for {alg} not found. Skipping...")
            continue
        results[alg] = []
        for seed in EVALUATION_SEEDS:
            print(f"\nEvaluating {alg}, seed {seed}...")
            results[alg].append(
                evaluate_rl_model(alg, seed, route_files[seed])
            )
            
    summary = summarize_results(results)
    raw_path, summary_path = save_results(results, summary)
    print("\nMean and sample variance across evaluation seeds:")
    print(f"{'Method':<12} | {'Metric':<22} | {'Mean':>14} | {'Variance':>14}")
    print("-" * 70)
    for name, values in summary.items():
        for metric_name, mean, variance in zip(
            METRIC_NAMES, values["mean"], values["variance"]
        ):
            print(f"{name:<12} | {metric_name:<22} | {mean:>14.4f} | {variance:>14.4f}")
        print("-" * 70)
    print(f"Per-seed results saved to {raw_path}")
    print(f"Mean/variance summary saved to {summary_path}")
    
    if len(results) > 0:
        plot_results(summary)