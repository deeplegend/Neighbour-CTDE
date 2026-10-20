import os
import time
import subprocess
import traceback
import csv
from pathlib import Path
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor
from sumo_env.generate_traffic import generate_routefile

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
EVALUATION_SEEDS = (1001, 1002, 1003, 1004, 1005)
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

def evaluate_traditional(controller_type="static", seed=42, route_file=None):
    """Run the static controller on one fixed seeded traffic scenario."""
    recompile_network(controller_type)
    
    binary = checkBinary('sumo')
    cmd = [binary, "-c", "sumo_env/square_grid.sumocfg", "--no-step-log", "true"]
    cmd.extend(["--seed", str(seed)])
    if route_file is not None:
        cmd.extend(["--route-files", os.fspath(route_file)])
    
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

def evaluate_rl_model(algorithm, seed, route_file):
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
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        obs_dict, _ = env.reset(seed=seed, route_file=route_file)
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
    metric_names = ("Throughput", "Avg Delay", "Avg EV Wait", "Inference Time")

    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("Method", "Seed", *metric_names))
        for method, runs in results.items():
            for seed, metrics in zip(EVALUATION_SEEDS, runs):
                writer.writerow((method, seed, *metrics))

    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        headings = ["Method", "Seed Count"]
        for metric in metric_names:
            headings.extend((f"{metric} Mean", f"{metric} Variance"))
        writer.writerow(headings)
        for method, values in summary.items():
            row = [method, len(results[method])]
            for mean, variance in zip(values["mean"], values["variance"]):
                row.extend((mean, variance))
            writer.writerow(row)
    return raw_path, summary_path


def plot_results(summary):
    labels = list(summary)
    means = [summary[name]["mean"] for name in labels]
    standard_deviations = [
        np.sqrt(summary[name]["variance"]) for name in labels
    ]
    throughputs = [res[0] for res in means]
    delays = [res[1] for res in means]
    ev_waits = [res[2] for res in means]
    throughput_errors = [res[0] for res in standard_deviations]
    delay_errors = [res[1] for res in standard_deviations]
    ev_wait_errors = [res[2] for res in standard_deviations]
    
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars_tp = ax.bar(
        x - width, throughputs, width, yerr=throughput_errors, capsize=4,
        label='System Throughput (Vehicles)', color='#2ca02c'
    )
    ax.bar_label(bars_tp, fmt='%.1f', padding=3)
    
    bars_delay = ax.bar(
        x, delays, width, yerr=delay_errors, capsize=4,
        label='Avg Congestion Delay (s)', color='#1f77b4'
    )
    ax.bar_label(bars_delay, fmt='%.2f', padding=3)
    
    bars_ev = ax.bar(
        x + width, ev_waits, width, yerr=ev_wait_errors, capsize=4,
        label='Avg EV Delay (s)', color='#d62728'
    )
    ax.bar_label(bars_ev, fmt='%.2f', padding=3)

    ax.set_ylabel('Performance Metric Values')
    ax.set_title(
        'Comparative Benchmark: Mean ± SD across Five Test Seeds',
        fontweight='bold'
    )
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
    route_dir = Path("sumo_env/eval_routes")
    route_dir.mkdir(parents=True, exist_ok=True)
    route_files = {}
    for seed in EVALUATION_SEEDS:
        route_file = route_dir / f"routes_seed_{seed}.rou.xml"
        generate_routefile(
            filename=str(route_file), max_steps=TEST_STEPS, seed=seed
        )
        route_files[seed] = route_file

    results["Static"] = []
    for seed in EVALUATION_SEEDS:
        print(f"\nEvaluating Static on seed {seed}...")
        results["Static"].append(
            evaluate_traditional("static", seed, route_files[seed])
        )
    
    tested_names = set()
    for alg in ALGORITHMS:
        if alg in tested_names:
            continue
        print(f"\nEvaluating {alg}...")
        weight_path = find_weight_file(alg)
        if not weight_path:
            print(f"  [!] Weights for {alg} not found in '{MODELS_DIR}/'. Skipping...")
            continue
        results[alg] = []
        for seed in EVALUATION_SEEDS:
            print(f"\nEvaluating {alg} on seed {seed}...")
            res = evaluate_rl_model(alg, seed, route_files[seed])
            if res[0] == 0 and res[1] == 0:
                raise RuntimeError(f"Evaluation failed for {alg} on seed {seed}")
            results[alg].append(res)
            tested_names.add(alg)

    summary = summarize_results(results)
    raw_path, summary_path = save_results(results, summary)
    print("\nMean and sample variance across five test seeds:")
    for name, values in summary.items():
        print(f"\n{name}")
        for metric, mean, variance in zip(
            ("Throughput", "Avg Delay", "Avg EV Wait", "Inference Time"),
            values["mean"], values["variance"]
        ):
            print(f"  {metric:<18}: {mean:.4f} | variance={variance:.4f}")
    print(f"\nPer-seed results: {raw_path}")
    print(f"Summary statistics: {summary_path}")
    
    if len(results) > 1:
        plot_results(summary)