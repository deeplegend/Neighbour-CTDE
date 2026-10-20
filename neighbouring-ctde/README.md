# Neighbor-CTDE Traffic Signal Control — 5×5 SUMO Grid

This package scales the original 2×2 Neighbor-CTDE traffic-control project to a 5×5 grid (25 signalized intersections). The training algorithms and core network architectures are retained; the updates make the road network, neighbor observations, global state, paths, route generation, and evaluation code support 25 intersections.

## What stays the same

- The same four selectable algorithms: DQN, D2QN, D3QN (Dueling DQN), and MAPPO.
- Shared actor/Q-network parameters across intersections; this is **not** changed into 25 separately trained neural networks.
- DQN-family training still uses the VDN mixer to sum per-intersection Q-values during centralized training.
- MAPPO still uses a shared decentralized actor and a centralized critic that consumes the concatenated global state.
- Two signal actions remain: north–south green or east–west green. Yellow clearance, minimum/maximum green time, queue/wait/emergency penalties, and throughput reward structure are retained.
- Vehicle settings, 0.35 per-second spawn probability, 0.04 emergency-vehicle probability, episode budgets, hidden layer sizes, optimizers, and main learning hyperparameters are unchanged.

## What scales for 5×5

- 25 traffic-light IDs: `TL_0` through `TL_24`.
- The 5×5 grid has 45 nodes (25 signalized intersections and 20 perimeter nodes), 80 directed internal edges, 40 directed boundary edges, and 300 signalized turning connections.
- Local node features remain 14 values. The four queue, waiting-time, and speed groups now aggregate all lanes for the N/E/S/W approaches instead of selecting the first four sorted lane IDs.
- Each local actor input is 70 values: its own 14 features followed by four stable neighbor slots (N, E, S, W), each 14 values. Missing neighbors along the grid boundary are zero-padded.
- The MAPPO global critic input is 350 values (25 intersections × 14 features).
- New checkpoints are saved under `models_5x5/` and new TensorBoard logs under `runs/CTDE_5x5_*`. Old 2×2 checkpoints have incompatible input dimensions and must not be reused.

## 1. Prerequisites

Install SUMO for Windows (including `sumo`, `sumo-gui`, and `netconvert`) and ensure the SUMO `bin` directory is on `PATH`.

Check in PowerShell:

```powershell
sumo --version
sumo-gui --version
netconvert --version
```

The Python process also needs the `traci` and `sumolib` modules that match your SUMO installation. If they are not importable, set `SUMO_HOME` to your SUMO installation directory and add its `tools` folder to `PYTHONPATH` in the current PowerShell session. Adjust the path for your installation:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
python -c "import traci, sumolib; print('SUMO Python tools OK')"
```

If SUMO is installed at a different location, use that location instead. These environment-variable commands apply to the current terminal only; add them to your Windows environment variables if you want them to persist.

## 2. Create/activate the Python environment

Open PowerShell in the extracted project root (the folder containing `env.py`, `train.py`, and `networks.py`).

For a new environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a virtual environment for the project, activate it and install the requirements instead. Since the project auto-selects CUDA when PyTorch detects it, verify GPU access if you intend to use the NVIDIA GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

If CUDA is unavailable, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver from the official PyTorch installation selector: <https://pytorch.org/get-started/locally/>. SUMO's vehicle simulation itself runs on the CPU; CUDA accelerates the neural-network operations.

## 3. Generate and compile the 5×5 SUMO network

From the project root, run:

```powershell
python sumo_env/build_network.py --grid-size 5
```

The script generates these files under `sumo_env/` and compiles the network via `netconvert`:

- `square_grid.nod.xml`
- `square_grid.edg.xml`
- `square_grid.con.xml`
- `square_grid.rou.xml`
- `square_grid.net.xml`

The route generator keeps the original spawn and emergency-vehicle probabilities, but builds valid routes across the larger grid. Run this build step again whenever you change the grid size or network inputs. Do **not** run the simulation or training until `square_grid.net.xml` has been generated successfully.

Optional: create the input XML files without calling `netconvert` (for example, on a machine without SUMO installed):

```powershell
python sumo_env/build_network.py --grid-size 5 --generate-only
```

This option does not compile the `.net.xml` network.

## 4. Inspect the network in SUMO-GUI

```powershell
sumo-gui -c sumo_env/square_grid.sumocfg
```

Check that the scene contains 25 signalized intersections in five rows and five columns, with access roads on the perimeter. Confirm the generated network is the expected 5×5 grid before training.

## 5. Run the checks

First test neural-network tensor dimensions (does not start SUMO):

```powershell
python test_networks.py
```

It should report 25 agents, a 70-value local observation, and a 350-value global state.

Then test the environment and one action step (requires SUMO and the compiled network):

```powershell
python test_env.py
```

The environment test should report 25 agents and a successful step. The reset routine checks that every expected traffic-light program exists and that the phase indices used by the controller are available.

## 6. Train one algorithm

```powershell
python train.py
```

Choose one option at the prompt:

| Choice | Algorithm |
|---|---|
| `1` | DQN |
| `2` | D2QN |
| `3` | D3QN |
| `4` | MAPPO |

The script trains only the selected algorithm and saves its weights to `models_5x5/`. Training logs are written to `runs/CTDE_5x5_DQN/`, `runs/CTDE_5x5_D2QN/`, `runs/CTDE_5x5_D3QN/`, or `runs/CTDE_5x5_MAPPO/`.

The original episode counts and learning settings are retained. A 5×5 SUMO episode processes more junctions and lane measurements than a 2×2 episode, so training may take longer. For a smoke test, temporarily reduce `VALUE_MAX_EPISODES` or `PPO_MAX_EPISODES` near the top of `train.py`, then restore the intended budgets for final experiments.

## 7. Evaluate trained models

After training one or more algorithms, run:

```powershell
python test.py
```

It evaluates the traditional static baseline and any available models in `models_5x5/` on the same five fixed traffic seeds (1001–1005). Training remains unchanged; these are five evaluation scenarios for each saved checkpoint, not five separately trained models. Per-seed measurements are saved to `CTDE_Evaluation_Raw.csv`; means and sample variances are saved to `CTDE_Evaluation_Summary.csv`. The comparison chart, `CTDE_Evaluation_Results.png`, plots the mean with standard-deviation error bars. Missing models are skipped. The evaluation script recompiles the network for static control, so run it from the project root with `netconvert` available on `PATH`.

To plot training reward/loss curves after multiple algorithms have been trained:

```powershell
python plot_graph.py
```

## Troubleshooting

- **`netconvert` not found:** add the SUMO `bin` directory to `PATH`, restart PowerShell, and retry the network build.
- **`No module named traci` / `sumolib`:** set `SUMO_HOME` and add `%SUMO_HOME%\tools` to `PYTHONPATH`, then verify the imports from the same activated `.venv` used to run the project.
- **`square_grid.net.xml` missing:** run `python sumo_env/build_network.py --grid-size 5` before `sumo-gui`, `test_env.py`, or training.
- **Unexpected traffic-light phase error:** rebuild using the provided `build_network.py`. Do not reuse the original 2×2 `.net.xml`.
- **Checkpoint dimension mismatch:** only use checkpoints created by this 5×5 project in `models_5x5/`. Original 2×2 actor weights expect 42-value inputs, and the old MAPPO critic expects a 56-value global state; the new model expects 70 and 350 respectively.
- **GPU is not used:** verify `torch.cuda.is_available()` in the same active `.venv`; otherwise PyTorch falls back to CPU.

## Validation performed while preparing this package

- Python syntax compilation passed for the updated scripts.
- Neural-network shape test passed for 25 agents, 70-value local observations, and a 350-value global state.
- Generated XML parsed successfully and route validation confirmed 25 traffic lights, 45 nodes, 120 directed edges, 300 connections, and 120 valid route definitions. The sample generated traffic file contains approximately 1,250 vehicles for a 3,600-second horizon.
- This preparation environment did not have SUMO (`sumo`, `netconvert`, `traci`, `sumolib`) installed, so the actual SUMO network compilation and simulation must be verified on your computer by following the steps above.
