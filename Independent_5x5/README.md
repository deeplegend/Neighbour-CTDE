# SUMO + DRL Traffic Control: 5x5 Grid

This version scales the supplied 2x2 project to a **5x5 grid (25 signalized intersections)**. It deliberately preserves the existing learning design:

- There is still **one shared DQN/Dueling-DQN model** across all intersections for I-DQN, I-D2QN, and I-D3QN.
- There is still **one shared actor and one shared local critic** for IPPO.
- Each intersection's action is chosen from its own observation; the training code stores transitions from all intersections in the same replay buffer / rollout, as before.
- The 14-value observation is local-only when `max_neighbors=0`, as used by the training code.
- The training algorithms and their existing hyperparameters are not redesigned.

The 14 observation values now have a stable and meaningful directional layout (North, East, South, West): normalized queue, waiting time, and speed for each approach, followed by emergency-vehicle presence and the current green direction. Because this corrects the previous arbitrary selection of the first four alphabetically sorted lanes, **retrain for 5x5; do not treat a 2x2 checkpoint as a trained 5x5 model**, even though the neural-network input dimension is still 14.

## 1. Prerequisites

1. Install SUMO and ensure `sumo`, `sumo-gui`, and `netconvert` are discoverable on your system `PATH`. Set `SUMO_HOME` to your SUMO installation directory if needed.
2. Use a Python version supported by your installed PyTorch.
3. From the `Independent` project directory, create a virtual environment and install dependencies.

Windows PowerShell example:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the activation instructions for your shell, or invoke `.venv\Scripts\python.exe` directly.

Quick checks:

```powershell
sumo --version
netconvert --version
python -c "import torch, traci, sumolib; print('PyTorch/TraCI/SUMOLib imports OK')"
```

If the Python TraCI/SUMOLib package version does not match your local SUMO installation, prefer the matching `tools` directory shipped with that SUMO installation. The official SUMO Python interface documentation describes using `SUMO_HOME/tools` for these imports: https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html

## 2. Generate the 5x5 network and traffic

Run these commands **from the `Independent` project root**, where `env.py` and `train_independent.py` are located:

```powershell
python sumo_env/build_network.py --grid-size 5 --max-steps 3600
```

The script regenerates all of the following files for 5x5 and compiles the network with `netconvert`:

- `sumo_env/square_grid.nod.xml`
- `sumo_env/square_grid.edg.xml`
- `sumo_env/square_grid.con.xml`
- `sumo_env/square_grid.net.xml`
- `sumo_env/square_grid.rou.xml`

The script validates that all 25 traffic lights were generated and that phases used by the existing environment (`0/1` and `4/5`) are present. If this validation fails, do not begin training until the signal program has been checked.

Open the network in the SUMO GUI before training if you want to inspect it visually:

```powershell
sumo-gui -c sumo_env/square_grid.sumocfg
```

Check that there are 25 signalized intersections, vehicles can enter and leave from boundary roads, and vehicles can traverse both horizontal and vertical roads.

## 3. Train the existing shared-model implementation

```powershell
python train_independent.py
```

Choose one option from the menu:

1. `I-DQN`
2. `I-D2QN`
3. `I-D3QN`
4. `IPPO`

The scripts use one shared policy per algorithm, not 25 separate models. Training creates `models_5x5/` and writes the new 5x5 model there. Existing 2x2 models in `models/` are not overwritten. Training logs go to `runs/5x5_INDEPENDENT_*`.

The supplied training budget remains the original budget (200 episodes for value-based training and 1000 for IPPO, with the original episode/step settings). A 5x5 run may take longer per episode than the 2x2 run. For an initial smoke test, you may temporarily reduce `VALUE_MAX_EPISODES` or `PPO_MAX_EPISODES` in `train_independent.py`; restore them for the full experiment.

## 4. Evaluate a trained model

After training at least one algorithm:

```powershell
python test.py
```

This evaluates the static baseline and any 5x5 model checkpoints found under `models_5x5/` on the same five fixed traffic scenarios (seeds 1001–1005). It saves per-scenario results and aggregate means, sample variances, and standard deviations as CSV files and plots means with standard-deviation error bars. It recompiles the same 5x5 node/edge/connection files before running the baseline. It does not automatically evaluate the old checkpoints in `models/`.

## 5. What changed

### Full file replacements

- `env.py`: dynamically creates IDs `TL_0` through `TL_24`, derives a grid neighbor map, and builds a fixed local 14-value observation by aggregating lanes into N/E/S/W approaches. The shared policy/training scheme is unchanged.
- `sumo_env/build_network.py`: generates the node, edge, and connection XML for an N x N grid, runs `netconvert`, validates signal phases, and generates routes.
- `sumo_env/generate_traffic.py`: creates routes across the grid rather than referring to the old `edge_0_to_1` / `in_TL0_*` 2x2 IDs.
- `test.py`: recompiles the generated connection file and evaluates only the separate 5x5 model directory.

### Small edits already applied in `train_independent.py`

- Both training paths instantiate the environment with `grid_size=5` and `max_neighbors=0`.
- Newly trained weights are written to `models_5x5/`, preserving the existing `models/` checkpoints.
- TensorBoard logs use `runs/5x5_INDEPENDENT_*` so they do not mix with the old experiment.

### Unchanged files

- `networks.py`: model architectures remain unchanged.
- `plot_results.py`: plotting utility remains unchanged.
- `sumo_env/square_grid.sumocfg`: the configuration continues to point at `square_grid.net.xml` and `square_grid.rou.xml`; build the network first.

## 6. Important experimental notes

- The new 5x5 features use explicit direction order N/E/S/W and aggregate the lanes for each approach. Retraining is required for meaningful evaluation; an old 2x2 model can have the right tensor shape but was trained on different lane-feature semantics.
- The reward calculation and shared throughput component have been retained from the supplied code. Each agent's observation remains local, but the throughput part of its reward is still based on network arrivals, as in the original implementation.
- For fair comparison, keep the vehicle-generation seed, arrival probability, emergency-vehicle probability, training budget, and reward weights consistent across runs.
- This package was checked for Python syntax and XML/route connectivity. Run `build_network.py` on a machine with SUMO installed to compile the SUMO network and perform the simulation test.
## Multi-scenario evaluation without retraining

Run `python test.py` to evaluate the existing `models_5x5` checkpoints on five fixed route scenarios (seeds 1001–1005). This does not train or update model weights.
