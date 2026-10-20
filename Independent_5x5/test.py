"""Multi-scenario evaluation for the 5x5 Independent traffic-control project.

This script DOES NOT train or update model weights. It builds one fixed set of
traffic route files, evaluates the static baseline and each existing checkpoint
on exactly the same route file and SUMO seed, and writes per-scenario and
aggregate CSV results.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import traci
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor
from sumo_env.generate_traffic import generate_routefile


PROJECT_DIR = Path(__file__).resolve().parent
SUMO_DIR = PROJECT_DIR / "sumo_env"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models_5x5"
DEFAULT_SEEDS = [1001, 1002, 1003, 1004, 1005]
ALGORITHMS = ["I-DQN", "I-D2QN", "I-D3QN", "IPPO"]
METRICS = [
    "throughput",
    "avg_delay",
    "avg_ev_wait",
    "total_departed",
    "vehicles_stuck_in_network",
]


def get_device():
    """Prefer the project's existing DirectML option, then CUDA, then CPU."""
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML device: {torch_directml.device_name(0)}")
        return device
    except Exception as exc:
        print(f"DirectML unavailable ({exc}); trying CUDA/CPU.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    print("Using CPU for model inference")
    return torch.device("cpu")


def recompile_network(tls_type: str = "static") -> None:
    """Compile the generated 5x5 grid using the requested signal type."""
    print(f"Compiling SUMO network (TLS type: {tls_type})...")
    cmd = [
        checkBinary("netconvert"),
        "--node-files", str(SUMO_DIR / "square_grid.nod.xml"),
        "--edge-files", str(SUMO_DIR / "square_grid.edg.xml"),
        "--connection-files", str(SUMO_DIR / "square_grid.con.xml"),
        "--output-file", str(SUMO_DIR / "square_grid.net.xml"),
        "--tls.default-type", tls_type,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "netconvert failed.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.stderr.strip():
        print(result.stderr.strip())


def actual_simulation_horizon(requested_steps: int, yellow_duration: int = 3,
                              green_duration: int = 10) -> int:
    """Round horizon up to full agent control intervals for fair comparison."""
    macro_step = yellow_duration + green_duration
    return ((requested_steps + macro_step - 1) // macro_step) * macro_step


def generate_test_scenarios(seeds, route_dir: Path, demand_steps: int,
                             grid_size: int, ev_probability: float,
                             spawn_probability: float):
    """Generate a reproducible route file for each evaluation seed exactly once."""
    route_dir.mkdir(parents=True, exist_ok=True)
    routes = {}
    for seed in seeds:
        route_path = (route_dir / f"routes_seed_{int(seed)}.rou.xml").resolve()
        # Deterministically regenerate once at the start of each evaluation
        # invocation. This avoids accidentally reusing a stale scenario file
        # when --steps or traffic-demand parameters have changed.
        generate_routefile(
            filename=str(route_path),
            max_steps=demand_steps,
            ev_probability=ev_probability,
            grid_size=grid_size,
            spawn_probability=spawn_probability,
            seed=int(seed),
        )
        routes[int(seed)] = route_path
    return routes


def evaluate_static(seed: int, route_file: Path, simulation_steps: int):
    """Evaluate the static SUMO signal controller on one fixed scenario."""
    binary = checkBinary("sumo")
    cmd = [
        binary,
        "-c", str(SUMO_DIR / "square_grid.sumocfg"),
        "--route-files", str(route_file.resolve()),
        "--seed", str(int(seed)),
        "--end", str(int(simulation_steps)),
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "-1",
    ]
    throughput = 0
    departed = 0
    cumulative_wait = 0.0
    ev_wait_sum = 0.0
    ev_observations = 0
    stuck = np.nan
    wall_start = time.perf_counter()
    traci_started = False
    try:
        traci.start(cmd)
        traci_started = True
        for _ in range(int(simulation_steps)):
            traci.simulationStep()
            throughput += traci.simulation.getArrivedNumber()
            departed += traci.simulation.getDepartedNumber()
            for vehicle_id in traci.vehicle.getIDList():
                wait = traci.vehicle.getWaitingTime(vehicle_id)
                cumulative_wait += wait
                if "emergency" in traci.vehicle.getTypeID(vehicle_id):
                    ev_wait_sum += wait
                    ev_observations += 1
        try:
            stuck = int(traci.vehicle.getIDCount())
        except Exception:
            stuck = np.nan
    finally:
        if traci_started:
            try:
                traci.close()
            except Exception:
                pass
    return {
        "method": "Static",
        "eval_seed": int(seed),
        "throughput": float(throughput),
        "avg_delay": float(cumulative_wait / max(1, simulation_steps)),
        "avg_ev_wait": float(ev_wait_sum / max(1, ev_observations)),
        "total_departed": float(departed),
        "vehicles_stuck_in_network": float(stuck) if not pd.isna(stuck) else np.nan,
        # Static control has no neural-policy inference latency to report.
        "decision_latency_ms": np.nan,
        "route_file": str(route_file),
    }


def find_weight_file(algorithm: str, models_dir: Path):
    """Find the existing checkpoint produced by train_independent.py."""
    candidates = [
        models_dir / f"{algorithm}_actor.pth",
        models_dir / f"{algorithm}.pth",
        models_dir / f"{algorithm.replace('I-', '')}_actor.pth",
        models_dir / "MAPPO_actor.pth" if algorithm == "IPPO" else models_dir / "__not_used__.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_actor(algorithm: str, checkpoint_path: Path, device):
    """Create the matching actor from checkpoint input shape and load weights."""
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True
    )
    if "fc1.weight" not in checkpoint:
        raise ValueError(f"Checkpoint has no fc1.weight: {checkpoint_path}")
    obs_dim = int(checkpoint["fc1.weight"].shape[1])
    if obs_dim % 14 != 0:
        raise ValueError(
            f"Unexpected input dimension {obs_dim} in {checkpoint_path}; "
            "expected a multiple of 14 features."
        )
    max_neighbors = (obs_dim // 14) - 1
    if max_neighbors < 0:
        raise ValueError(f"Invalid observation dimension {obs_dim}")

    # In the Independent approach, currently trained 5x5 checkpoints use 14
    # local features (max_neighbors=0). Infer from the checkpoint to retain
    # compatibility if the user has a model trained with another config.
    # The environment's discrete signal-action space is fixed at two actions:
    # retain the existing north-south / east-west green choice.
    action_dim = 2

    if algorithm == "I-D3QN":
        actor = DuelingDQNActor(obs_dim, action_dim).to(device)
    elif algorithm == "IPPO":
        actor = MAPPOActor(obs_dim, action_dim).to(device)
    else:
        actor = DQNActor(obs_dim, action_dim).to(device)
    actor.load_state_dict(checkpoint)
    actor.eval()
    return actor, obs_dim, max_neighbors


def evaluate_rl_model(algorithm: str, checkpoint_path: Path, seed: int,
                      route_file: Path, simulation_steps: int, device):
    """Evaluate one existing checkpoint on one seed; never updates weights."""
    actor = None
    env = None
    try:
        actor, obs_dim, max_neighbors = load_actor(
            algorithm, checkpoint_path, device
        )
        env = MultiIntersectionSUMOEnv(
            use_gui=False,
            max_steps=simulation_steps,
            grid_size=5,
            max_neighbors=max_neighbors,
        )
        obs_dict, _ = env.reset(seed=seed, route_file=str(route_file))
        done = False
        decision_latencies = []
        while not done:
            actions = {}
            start = time.perf_counter()
            with torch.no_grad():
                for agent in env.agents:
                    obs_tensor = torch.as_tensor(
                        obs_dict[agent], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    if algorithm == "IPPO":
                        # Deterministic greedy policy for fair, repeatable testing.
                        logits = actor(obs_tensor)
                        actions[agent] = int(logits.argmax(dim=1).item())
                    else:
                        q_values = actor(obs_tensor)
                        actions[agent] = int(q_values.argmax(dim=1).item())
            decision_latencies.append(time.perf_counter() - start)
            next_obs, _rewards, terms, _truncs, _infos = env.step(actions)
            done = all(terms.values())
            obs_dict = next_obs

        metrics = env.get_episode_metrics()
        return {
            "method": algorithm,
            "eval_seed": int(seed),
            "throughput": float(metrics["throughput"]),
            "avg_delay": float(metrics["avg_delay"]),
            "avg_ev_wait": float(metrics["avg_ev_wait"]),
            "total_departed": float(metrics["total_departed"]),
            "vehicles_stuck_in_network": float(metrics["vehicles_stuck_in_network"])
                if metrics["vehicles_stuck_in_network"] is not None else np.nan,
            "decision_latency_ms": float(np.mean(decision_latencies) * 1000.0)
                if decision_latencies else np.nan,
            "route_file": str(route_file),
            "checkpoint": str(checkpoint_path),
            "obs_dim": obs_dim,
        }
    finally:
        if env is not None:
            env.close()
        else:
            try:
                traci.close()
            except Exception:
                pass


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = METRICS + ["decision_latency_ms"]
    for method, group in raw.groupby("method", sort=False):
        row = {"method": method, "n_scenarios": int(group["eval_seed"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            # Sample variance and SD are undefined for fewer than two samples.
            row[f"{metric}_variance"] = float(values.var(ddof=1)) if len(values) >= 2 else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path):
    metric_info = [
        ("throughput", "Completed vehicles (higher is better)"),
        ("avg_delay", "Network-wide waiting metric (lower is better)"),
        ("avg_ev_wait", "Emergency-vehicle waiting metric (lower is better)"),
    ]
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (metric, title) in zip(axes, metric_info):
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        stds = summary[f"{metric}_std"].to_numpy(dtype=float)
        stds = np.nan_to_num(stds, nan=0.0)
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Independent 5x5 Traffic Control — Multi-scenario Evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved Independent 5x5 models on multiple fixed traffic scenarios; no training is performed."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Fixed evaluation seeds (default: 1001 through 1010).",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS,
        help="Algorithms to evaluate; missing checkpoints are skipped.",
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--route-dir", default=str(SUMO_DIR / "eval_routes"))
    parser.add_argument("--steps", type=int, default=1000,
                        help="Traffic-demand generation horizon in seconds; simulation horizon is rounded up to a full 13-second control interval.")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--ev-probability", type=float, default=0.04)
    parser.add_argument("--spawn-probability", type=float, default=0.35)
    parser.add_argument("--output-prefix", default="Independent_MultiScenario")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.grid_size != 5:
        raise ValueError("This packaged evaluator is intended for the 5x5 project; use --grid-size 5.")
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")

    models_dir = Path(args.models_dir).resolve()
    route_dir = Path(args.route_dir).resolve()
    raw_csv = PROJECT_DIR / f"{args.output_prefix}_Raw.csv"
    summary_csv = PROJECT_DIR / f"{args.output_prefix}_Summary.csv"
    chart_path = PROJECT_DIR / f"{args.output_prefix}_Results.png"

    # Environment actions advance 3 seconds of yellow + 10 seconds of green.
    simulation_steps = actual_simulation_horizon(args.steps)
    print("=" * 78)
    print("INDEPENDENT 5x5 MULTI-SCENARIO EVALUATION (NO TRAINING)")
    print(f"Seeds: {seeds}")
    print(f"Demand-generation horizon: {args.steps}s; simulated horizon: {simulation_steps}s")
    print(f"Models directory: {models_dir}")
    print("=" * 78)

    # Compile a static SUMO TLS program once. Existing trained agents explicitly
    # set/hold phases via TraCI on this same network in their own evaluations.
    recompile_network("static")
    route_files = generate_test_scenarios(
        seeds=seeds,
        route_dir=route_dir,
        demand_steps=args.steps,
        grid_size=args.grid_size,
        ev_probability=args.ev_probability,
        spawn_probability=args.spawn_probability,
    )

    device = get_device()
    rows = []
    for seed in seeds:
        route = route_files[seed]
        print(f"\n[Scenario seed {seed}] Static baseline...")
        row = evaluate_static(seed, route, simulation_steps)
        rows.append(row)
        print(
            f"  throughput={row['throughput']:.0f}, delay={row['avg_delay']:.3f}, "
            f"EV wait={row['avg_ev_wait']:.3f}"
        )

    found_models = {}
    for algorithm in args.algorithms:
        checkpoint = find_weight_file(algorithm, models_dir)
        if checkpoint is None:
            print(f"\n[SKIP] No checkpoint found for {algorithm} under {models_dir}")
        else:
            found_models[algorithm] = checkpoint
            print(f"\nCheckpoint for {algorithm}: {checkpoint}")

    for algorithm, checkpoint in found_models.items():
        for seed in seeds:
            route = route_files[seed]
            print(f"\n[{algorithm} | evaluation seed {seed}]")
            row = evaluate_rl_model(
                algorithm=algorithm,
                checkpoint_path=checkpoint,
                seed=seed,
                route_file=route,
                simulation_steps=simulation_steps,
                device=device,
            )
            rows.append(row)
            print(
                f"  throughput={row['throughput']:.0f}, delay={row['avg_delay']:.3f}, "
                f"EV wait={row['avg_ev_wait']:.3f}, "
                f"decision latency={row['decision_latency_ms']:.3f} ms"
            )

    if not rows:
        raise RuntimeError("No evaluations produced results")
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_csv, index=False)
    summary = summarize_results(raw)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, chart_path)

    print("\n" + "=" * 110)
    print("SUMMARY (mean, sample variance, and SD across evaluation scenarios)")
    for _, row in summary.iterrows():
        print(f"\n{row['method']}  (n={int(row['n_scenarios'])})")
        for metric in ["throughput", "avg_delay", "avg_ev_wait", "total_departed", "vehicles_stuck_in_network", "decision_latency_ms"]:
            mean = row.get(f"{metric}_mean", np.nan)
            variance = row.get(f"{metric}_variance", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            if pd.isna(mean):
                continue
            metric_label = metric.replace("_", " ")
            if pd.isna(std):
                print(f"  {metric_label:<30}: mean={mean:.4f} (variance/SD undefined; one valid sample)")
            else:
                print(
                    f"  {metric_label:<30}: mean={mean:.4f}, "
                    f"variance={variance:.4f}, SD={std:.4f}"
                )
    print("\nSaved files:")
    print(f"  Raw scenario results : {raw_csv}")
    print(f"  Summary statistics   : {summary_csv}")
    print(f"  Comparison chart     : {chart_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
