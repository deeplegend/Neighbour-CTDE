import os
import subprocess
import torch
import numpy as np
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import MultiIntersectionSUMOEnv
from networks_centralized import (
    CentralizedDQNActor,
    CentralizedDuelingDQNActor,
    CentralizedPPOActor,
    decode_action
)


def recompile_network(tls_type="static"):
    """Recompiles the SUMO network to switch between Static and Actuated
    controllers. Copied from the independent/neighbor-CTDE test scripts --
    this file was missing it entirely, so the net.xml's tlLogic type was
    left to whatever an earlier, unrelated script run had last set it to."""
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

# =====================================================================
# HARDWARE ACCELERATOR SETUP (AMD GPU / DIRECTML / CUDA / CPU)
# =====================================================================
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Testing on AMD GPU (DirectML): {torch_directml.device_name(0)}")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on:", torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")

# Testing Hyperparameters
# NOTE: TEST_STEPS is in SIMULATED SECONDS (matches env.max_steps), not
# env.step() macro-call count. The previous version of this script ran a
# `for step in range(TEST_STEPS)` loop that called env.step() (each of which
# advances ~13 simulated seconds) exactly TEST_STEPS times without ever
# checking env's own done/termination signal -- driving the simulation for
# roughly 13x longer (~13,000s) than intended, and roughly 13x longer than
# the Neighbor-CTDE / Independent evaluation scripts use. That alone made
# this script's numbers non-comparable to the others.
TEST_STEPS = 1000
GUI_ENABLED = False  # Set to True to watch the benchmark visually


class TrafficBenchmarker:
    def __init__(self, env):
        self.env = env
        self.agents = env.agents
        self.results = {}  # Stores metrics for the final table & plot

    def run_fixed_time(self):
        """Baseline: the REAL netconvert-generated static fixed-time program
        (same 'Static' controller used in the Independent / Neighbor-CTDE
        scripts), run directly through raw TraCI -- NOT through
        self.env.step(). env.step() always drives lights through its own
        2-action RL phase_map and immediately re-locks whichever phase it
        picks via setPhaseDuration(), which would override the static
        program's own natural 8-phase cycling. This mirrors
        evaluate_traditional('static') in the other two scripts exactly, so
        the numbers are directly comparable -- unlike the previous version,
        which toggled NS/EW every 6 macro-steps through the RL environment
        and was not equivalent to 'Static' at all.
        """
        print("\n--- Running Baseline: Static (Fixed-Time) ---")
        recompile_network("static")

        binary = checkBinary('sumo')
        cmd = [binary, "-c", "sumo_env/square_grid.sumocfg", "--no-step-log", "true"]
        traci.start(cmd)

        total_arrived = 0
        syswide_total_wait = 0.0
        ev_total_wait = 0.0
        ev_observations = 0

        for _ in range(TEST_STEPS):
            traci.simulationStep()
            total_arrived += traci.simulation.getArrivedNumber()
            for veh_id in traci.vehicle.getIDList():
                w = traci.vehicle.getWaitingTime(veh_id)
                syswide_total_wait += w
                if "emergency" in traci.vehicle.getTypeID(veh_id):
                    ev_total_wait += w
                    ev_observations += 1

        traci.close()

        metrics = {
            "throughput": float(total_arrived),
            "avg_delay": syswide_total_wait / TEST_STEPS,
            "avg_ev_wait": ev_total_wait / max(1, ev_observations),
        }
        self._save_and_print_results("Static (Fixed-Time)", metrics)

    def run_centralized_model(self, algorithm):
        """Evaluates trained Centralized models."""
        model_path = f"models/Centralized_{algorithm}.pth"
        if algorithm == "MAPPO":
            model_path = "models/Centralized_PPO.pth"

        if not os.path.exists(model_path):
            print(f"Skipping Centralized {algorithm}: Model file not found at {model_path}")
            return

        print(f"\n--- Running Centralized Model: {algorithm} ---")
        # ADDED: ensure the net.xml is deterministically "static" type before
        # every RL run, matching the independent/neighbor-CTDE scripts --
        # previously this depended on whatever an earlier script run had
        # last left the net.xml as.
        recompile_network("static")
        self.env.reset()

        global_obs_dim = self.env.global_state_dim
        joint_action_dim = 16

        if algorithm == "D3QN":
            net = CentralizedDuelingDQNActor(global_obs_dim, joint_action_dim).to(device)
        elif algorithm == "MAPPO":
            net = CentralizedPPOActor(global_obs_dim, joint_action_dim).to(device)
        else:
            net = CentralizedDQNActor(global_obs_dim, joint_action_dim).to(device)

        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        global_state = self.env.get_global_state()
        done = False

        # FIX: same as run_fixed_time -- loop until env signals done rather
        # than a fixed TEST_STEPS macro-iteration count.
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                if algorithm == "MAPPO":
                    action, _, _ = net.get_action(state_tensor)
                    joint_action = action.item()
                else:
                    joint_action = net(state_tensor).argmax(dim=1).item()

            actions_dict = decode_action(joint_action, self.agents)
            _, _, terms, _, _ = self.env.step(actions_dict)
            done = all(terms.values())
            global_state = self.env.get_global_state()

        metrics = self.env.get_episode_metrics()
        self._save_and_print_results(f"Centralized {algorithm}", metrics)

    def _save_and_print_results(self, name, metrics):
        """Stores and prints the metrics returned by env.get_episode_metrics().

        FIX: previously this method computed its own avg_wait as
        total_wait_time / throughput (person-seconds of queueing per arrived
        vehicle) -- a different statistic from the other evaluation scripts,
        which divide by elapsed TIME instead. avg_delay_sec here now comes
        straight from env.py, using the same time-based definition every
        other script uses, so the numbers are directly comparable.

        FIX: EV delay now comes from env.py's avg_ev_delay_sec (waiting time
        sampled while an EV is present), matching the "Avg EV Wait" columns
        in the other tables -- not total trip duration measured in outer-loop
        steps, which is what this method used to report as "Avg EV Time".
        """
        self.results[name] = {
            "throughput": metrics["throughput"],
            "avg_wait": metrics["avg_delay"],
            "avg_ev_time": metrics["avg_ev_wait"],
        }

        print(f"[{name} Results]")
        print(f"  - Total Throughput:     {metrics['throughput']:.0f} vehicles")
        print(f"  - Avg Wait Time/Veh:    {metrics['avg_delay']:.2f} seconds")
        print(f"  - Avg EV Wait:          {metrics['avg_ev_wait']:.2f} seconds")
        print("-" * 50)

    def print_summary_table(self):
        """Displays a structured table comparing throughput, average wait time, and EV wait time."""
        if not self.results:
            print("No evaluation data available to display table.")
            return

        table_width = 85
        print("\n" + "=" * table_width)
        print(f"{'BENCHMARK SUMMARY COMPARISON TABLE':^{table_width}}")
        print("=" * table_width)
        print(f"{'Method / Controller':<25} | {'Throughput (Vehicles)':<21} | {'Avg Wait Time (s)':<17} | {'Avg EV Wait (s)':<15}")
        print("-" * table_width)

        for method, data in self.results.items():
            print(f"{method:<25} | {data['throughput']:<21.0f} | {data['avg_wait']:<17.2f} | {data['avg_ev_time']:<15.2f}")

        print("=" * table_width + "\n")

    def plot_comparison_graphs(self):
        if not self.results:
            print("No models were successfully tested. Cannot generate graph.")
            return

        labels = list(self.results.keys())
        throughput = [self.results[m]["throughput"] for m in labels]
        wait_times = [self.results[m]["avg_wait"] for m in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'royalblue'
        bars1 = ax1.bar(x - width / 2, throughput, width, label='Total Throughput', color=color1, edgecolor='black', alpha=0.85)
        ax1.bar_label(bars1, fmt='%.1f', padding=3)

        ax1.set_ylabel('Total Throughput (Vehicles) [\u2191 Higher is better]', color=color1, fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=10, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        color2 = 'crimson'
        bars2 = ax2.bar(x + width / 2, wait_times, width, label='Avg Waiting Time', color=color2, edgecolor='black', alpha=0.85)
        ax2.bar_label(bars2, fmt='%.2f', padding=3)

        ax2.set_ylabel('Avg Waiting Time (Seconds) [\u2193 Lower is better]', color=color2, fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title('Traffic Control Benchmarking: Throughput vs. Average Delay', fontsize=13, fontweight='bold', pad=15)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

        fig.tight_layout()
        plt.savefig("benchmark_comparison_single.png", dpi=300)
        print("=> Single combined benchmark graph saved to 'benchmark_comparison_single.png'.")
        plt.show()


if __name__ == "__main__":
    print("==============================================")
    print("  MULTI-AGENT TRAFFIC CONTROL BENCHMARKING    ")
    print("==============================================")

    eval_env = MultiIntersectionSUMOEnv(use_gui=GUI_ENABLED, max_steps=TEST_STEPS)
    benchmarker = TrafficBenchmarker(eval_env)

    # 1. Run Baseline
    benchmarker.run_fixed_time()

    # 3. Run Centralized Models
    benchmarker.run_centralized_model("DQN")
    benchmarker.run_centralized_model("D2QN")
    benchmarker.run_centralized_model("D3QN")
    benchmarker.run_centralized_model("MAPPO")

    eval_env.close()

    # 4. Display Formatted Table
    benchmarker.print_summary_table()

    # 5. Generate and Display Single Combined Graph
    benchmarker.plot_comparison_graphs()

    print("\nBenchmarking Complete!")