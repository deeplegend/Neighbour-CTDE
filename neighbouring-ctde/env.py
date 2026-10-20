import os
import sys
import numpy as np
import traci
from sumolib import checkBinary

class MultiIntersectionSUMOEnv:
    """
    Multi-Agent Environment for an NxN signalized grid.
    Supports neighbor-observation CTDE with a centralized critic / VDN training,
    Emergency Vehicle (EV) prioritization, and Hardware Failsafe overrides.
    """
    def __init__(
        self,
        sumocfg_path="sumo_env/square_grid.sumocfg",
        use_gui=False,
        max_steps=3600,
        yellow_duration=3,
        step_duration=10,         # INCREASED from 5 to 10 for queue discharge
        min_green_time=15,        # ADDED: Prevents phase fluttering
        max_green_time=60,
        max_neighbors=4,
        w_queue=0.1,
        w_wait=0.05,
        w_ev=3.0,
        w_switch=1.5,
        w_throughput=1.5,         # ADDED: Positive reward for completed trips
        spillback_occupancy_threshold=0.85,  # ADDED: for spillback metric
        grid_size=5
    ):
        self.grid_size = int(grid_size)
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        # Resolve the config relative to the current working directory first,
        # then relative to this project directory.
        if os.path.isabs(sumocfg_path):
            resolved_cfg = sumocfg_path
        elif os.path.exists(sumocfg_path):
            resolved_cfg = os.path.abspath(sumocfg_path)
        else:
            resolved_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), sumocfg_path)
        self.sumocfg_path = resolved_cfg
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.step_duration = step_duration
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.max_neighbors = max_neighbors

        self.w_queue = w_queue
        self.w_wait = w_wait
        self.w_ev = w_ev
        self.w_switch = w_switch
        self.w_throughput = w_throughput

        self.spillback_occupancy_threshold = spillback_occupancy_threshold

        # FIX: SUMO's own tlLogic program will auto-advance a traffic light once
        # the *default* duration of its currently active phase elapses, even if
        # traci.trafficlight.setPhase() was called to force that phase active.
        # setPhase() alone does NOT give the RL agent exclusive authority over
        # timing. We therefore pin every phase we set with a very large
        # setPhaseDuration() call so SUMO's internal program can never silently
        # override our control loop. Without this, self.current_phases /
        # self.time_in_green can desync from what the simulator is actually
        # doing, corrupting both the reward signal and the next observation.
        self.HOLD_DURATION = 9999

        self.agents = [f"TL_{i}" for i in range(self.grid_size * self.grid_size)]
        self.possible_agents = self.agents[:]
        self.agent_positions = {
            f"TL_{row * self.grid_size + col}": (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
        }
        self.directional_neighbors = self._build_directional_neighbors()
        # Compatibility/readability view: only actual neighboring agents, in N/E/S/W order.
        self.neighbor_map = {
            agent: [neighbor for neighbor in self.directional_neighbors[agent].values() if neighbor is not None]
            for agent in self.agents
        }

        self.node_feature_dim = 14
        if not 0 <= self.max_neighbors <= 4:
            raise ValueError("max_neighbors must be between 0 and 4")
        self.obs_dim = (1 + self.max_neighbors) * self.node_feature_dim
        self.global_state_dim = len(self.agents) * self.node_feature_dim
        self.action_dim = 2

        # FIX: this network's tlLogic has 8 phases per junction, not 4 --
        # netconvert inserted dedicated protected-left-turn phases (indices
        # 2/3 and 6/7) alongside the two main through-phases (0/1 and 4/5).
        # The RL action space here only models the two main directions
        # (permitted/yielding left turns during the main green, no protected
        # phase), so action 1 must map to phase index 4 (the real EW-direction
        # green) and index 5 (its yellow) -- NOT index 2, which is the NS
        # protected-left-only phase where every through movement, including
        # the entire EW direction, sits on red. Mapping action 1 to index 2
        # meant every "give the cross street a turn" decision actually
        # gridlocked most of the intersection.
        self.phase_map = {
            0: {"green": 0, "yellow": 1},
            1: {"green": 4, "yellow": 5}
        }

        self.current_phases = {agent: 0 for agent in self.agents}
        self.time_in_green = {agent: 0 for agent in self.agents}
        self.is_yellow = {agent: False for agent in self.agents}
        self.step_count = 0
        self.tls_lanes = {}
        self.approach_lanes = {}

        self.metrics = {}

    def _build_directional_neighbors(self):
        """Build the fixed N/E/S/W neighbor slots for every signal."""
        neighbors = {}
        for agent, (row, col) in self.agent_positions.items():
            directional = {
                "N": f"TL_{(row - 1) * self.grid_size + col}" if row > 0 else None,
                "E": f"TL_{row * self.grid_size + (col + 1)}" if col < self.grid_size - 1 else None,
                "S": f"TL_{(row + 1) * self.grid_size + col}" if row < self.grid_size - 1 else None,
                "W": f"TL_{row * self.grid_size + (col - 1)}" if col > 0 else None,
            }
            neighbors[agent] = directional
        return neighbors

    def _incoming_edge_id(self, agent, direction):
        """Return the edge that enters an intersection from a compass direction."""
        index = int(agent.split("_")[1])
        row, col = self.agent_positions[agent]
        if direction == "N" and row > 0:
            upstream = index - self.grid_size
            return f"edge_{upstream}_to_{index}"
        if direction == "E" and col < self.grid_size - 1:
            upstream = index + 1
            return f"edge_{upstream}_to_{index}"
        if direction == "S" and row < self.grid_size - 1:
            upstream = index + self.grid_size
            return f"edge_{upstream}_to_{index}"
        if direction == "W" and col > 0:
            upstream = index - 1
            return f"edge_{upstream}_to_{index}"
        return f"in_TL_{index}_{direction}"

    def _get_controlled_lanes(self):
        """Map all incoming lanes into stable N/E/S/W approach groups.

        The old 2x2 code took the first four alphabetically sorted controlled
        lane IDs. On a 5x5 network each approach has multiple lanes, so that
        could omit entire directions. This mapping keeps the 14-feature schema
        but aggregates all lanes per compass approach.
        """
        self.tls_lanes = {}
        self.approach_lanes = {}
        for agent in self.agents:
            per_direction = {}
            all_lanes = []
            for direction in ("N", "E", "S", "W"):
                edge_id = self._incoming_edge_id(agent, direction)
                lane_count = traci.edge.getLaneNumber(edge_id)
                lane_ids = [f"{edge_id}_{lane_index}" for lane_index in range(lane_count)]
                per_direction[direction] = lane_ids
                all_lanes.extend(lane_ids)
            self.approach_lanes[agent] = per_direction
            self.tls_lanes[agent] = all_lanes

    def reset(self, seed: int | None = None, route_file: str | None = None):
        """Resets the SUMO simulation and returns initial observations."""
        try:
            traci.close()
        except Exception:
            pass

        binary = checkBinary('sumo-gui' if self.use_gui else 'sumo')
        sumo_cmd = [
            binary,
            "-c", self.sumocfg_path,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "-W"
        ]
        if seed is not None:
            sumo_cmd.extend(["--seed", str(seed)])
        if route_file is not None:
            sumo_cmd.extend(["--route-files", route_file])
        traci.start(sumo_cmd)

        actual_agents = set(traci.trafficlight.getIDList())
        expected_agents = set(self.agents)
        missing_agents = sorted(expected_agents - actual_agents)
        if missing_agents:
            traci.close()
            raise RuntimeError(
                f"SUMO network does not contain all expected {self.grid_size}x{self.grid_size} "
                f"traffic lights. Missing: {missing_agents[:10]}"
            )
        # The controller below uses phases 0/1 for NS green/yellow and 4/5 for
        # EW green/yellow, matching the 8-phase signal programs produced by
        # the generated 4-approach connection layout. Fail clearly otherwise.
        for agent in self.agents:
            logics = traci.trafficlight.getAllProgramLogics(agent)
            active_program = traci.trafficlight.getProgram(agent)
            active_logic = next((logic for logic in logics if logic.programID == active_program), logics[0] if logics else None)
            if active_logic is None or len(active_logic.phases) < 6:
                traci.close()
                raise RuntimeError(
                    f"Traffic light {agent} has an unexpected signal program. "
                    "Expected at least 6 phases so phase indices 0/1 and 4/5 are available. "
                    "Rebuild the network with sumo_env/build_network.py."
                )

        self.step_count = 0
        self.current_phases = {agent: 0 for agent in self.agents}
        self.time_in_green = {agent: 0 for agent in self.agents}
        self.is_yellow = {agent: False for agent in self.agents}

        self.metrics = {
            "total_wait": 0.0,
            "total_queue": 0.0,
            "queue_samples": 0,
            "total_arrived": 0,
            "total_departed": 0,   # ADDED
            "spillback_events": 0,
            "ev_total_wait": 0.0,  # ADDED (controlled-lane scope, diagnostic)
            "ev_wait_samples": 0,  # ADDED (controlled-lane scope, diagnostic)
            "syswide_total_wait": 0.0,      # ADDED: network-wide, every vehicle
            "elementary_ticks": 0,          # ADDED: sampled every simulated second
            "ev_syswide_total_wait": 0.0,   # ADDED
            "ev_syswide_observations": 0,   # ADDED
        }

        self._get_controlled_lanes()

        # Set initial green phases AND lock their duration (see HOLD_DURATION note above)
        for agent in self.agents:
            traci.trafficlight.setPhase(agent, self.phase_map[0]["green"])
            traci.trafficlight.setPhaseDuration(agent, self.HOLD_DURATION)

        obs = self._get_all_observations()
        info = {agent: {} for agent in self.agents}
        return obs, info

    def _extract_local_features(self, agent):
        """Return the fixed 14-dimensional feature vector for one intersection.

        Layout is retained from the original implementation:
        [0:4] queue features for N/E/S/W approaches,
        [4:8] waiting-time features for N/E/S/W,
        [8:12] mean speeds for N/E/S/W,
        [12] emergency-vehicle-present flag, [13] active green direction.
        Every lane belonging to an approach is aggregated into that direction's
        feature, so lane ordering does not change the meaning of an input.
        """
        queue_lengths = []
        wait_times = []
        mean_speeds = []
        ev_present = 0.0
        per_direction = self.approach_lanes.get(agent, {})

        for direction in ("N", "E", "S", "W"):
            lanes = per_direction.get(direction, [])
            queue = 0
            waiting = 0.0
            speeds = []
            for lane in lanes:
                queue += traci.lane.getLastStepHaltingNumber(lane)
                waiting += traci.lane.getWaitingTime(lane)
                speeds.append(traci.lane.getLastStepMeanSpeed(lane))
                for vehicle_id in traci.lane.getLastStepVehicleIDs(lane):
                    if (traci.vehicle.getTypeID(vehicle_id) == "emergency_car"
                            or traci.vehicle.getVehicleClass(vehicle_id) == "emergency"):
                        ev_present = 1.0

            queue_lengths.append(min(queue / 20.0, 1.0))
            wait_times.append(min(waiting / 120.0, 1.0))
            mean_speed = float(np.mean(speeds)) if speeds else 0.0
            mean_speeds.append(min(mean_speed / 13.89, 1.0))

        phase_feat = [float(self.current_phases[agent])]
        features = queue_lengths + wait_times + mean_speeds + [ev_present] + phase_feat
        return np.asarray(features, dtype=np.float32)

    def _get_all_observations(self):
        """Build each local observation from ego features + N/E/S/W neighbor slots.

        Missing neighbors at the grid boundary are represented by all-zero
        feature vectors, preserving the same input dimension for every actor.
        """
        node_features = {agent: self._extract_local_features(agent) for agent in self.agents}
        obs = {}
        directions = ("N", "E", "S", "W")[:self.max_neighbors]

        for agent in self.agents:
            pieces = [node_features[agent]]
            directional = self.directional_neighbors[agent]
            for direction in directions:
                neighbor_id = directional[direction]
                if neighbor_id is None:
                    pieces.append(np.zeros(self.node_feature_dim, dtype=np.float32))
                else:
                    pieces.append(node_features[neighbor_id])
            obs[agent] = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        return obs

    def get_global_state(self):
        """Return concatenated 14-feature states for all 25 intersections (350 values for 5x5)."""
        features = [self._extract_local_features(agent) for agent in self.agents]
        return np.concatenate(features, axis=0).astype(np.float32, copy=False)

    def _calculate_reward(self, agent, action_switched, arrived_vehicles):
        lanes = self.tls_lanes.get(agent, [])
        total_queue = 0
        total_wait = 0
        ev_delay = 0

        for lane in lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            total_wait += min(traci.lane.getWaitingTime(lane), 300.0)

            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vid in veh_ids:
                if traci.vehicle.getTypeID(vid) == "emergency_car" or traci.vehicle.getVehicleClass(vid) == "emergency":
                    if traci.vehicle.getSpeed(vid) < 1.0:
                        ev_delay += 1.0

        switch_penalty = 1.0 if action_switched else 0.0

        # Reward = Throughput Bonus - Penalties
        reward = (
            (self.w_throughput * (arrived_vehicles / len(self.agents))) -
            (self.w_queue * total_queue +
             self.w_wait * (total_wait / 60.0) +
             self.w_ev * ev_delay +
             self.w_switch * switch_penalty)
        )
        return float(reward)

    def _update_episode_metrics(self):
        """ADDED: accumulate raw traffic metrics (avg delay, avg queue, spillback
        events, EV delay) each macro-step so training/eval code can build a fair,
        direct comparison against a static baseline using the same units your
        methodology describes (throughput, delay, spillbacks, EV wait). Using
        this single method from every evaluation script guarantees identical
        definitions/sampling frequency across Neighbor-CTDE, Centralized, and
        Independent comparisons -- rather than each test script re-deriving its
        own (differently-defined) statistics."""
        for agent in self.agents:
            for lane in self.tls_lanes.get(agent, []):
                w = traci.lane.getWaitingTime(lane)
                q = traci.lane.getLastStepHaltingNumber(lane)
                occ = traci.lane.getLastStepOccupancy(lane)

                self.metrics["total_wait"] += w
                self.metrics["total_queue"] += q
                self.metrics["queue_samples"] += 1

                if occ >= self.spillback_occupancy_threshold:
                    self.metrics["spillback_events"] += 1

                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getTypeID(vid) == "emergency_car" or traci.vehicle.getVehicleClass(vid) == "emergency":
                        self.metrics["ev_total_wait"] += traci.vehicle.getWaitingTime(vid)
                        self.metrics["ev_wait_samples"] += 1

    def get_episode_metrics(self):
        """ADDED: returns aggregated metrics for the episode so far.

        throughput_vehicles: completed trips (arrived).
        total_departed: vehicles that actually entered the network.
        vehicles_stuck_in_network: departed but not yet arrived when this is
            called -- call at episode end (before close()) to see how many
            vehicles never finished their trip within the episode. A large
            gap between total_departed and throughput_vehicles means
            vehicles are getting in but not getting through (a congestion /
            starvation problem downstream). A large gap between the demand's
            expected vehicle count and total_departed instead points to
            vehicles never even being able to enter (jammed entry edges).
        """
        samples = max(1, self.metrics["queue_samples"])
        ev_samples = max(1, self.metrics["ev_wait_samples"])
        ticks = max(1, self.metrics["elementary_ticks"])
        ev_obs = max(1, self.metrics["ev_syswide_observations"])
        try:
            vehicles_stuck = traci.vehicle.getIDCount()
        except Exception:
            vehicles_stuck = None
        return {
            # PRIMARY comparison metrics: network-wide (every vehicle,
            # every edge), sampled every elementary simulated second.
            # Key names match what evaluate_rl_model() in the
            # independent / neighbor-CTDE / centralized test scripts
            # already expect.
            "throughput": float(self.metrics["total_arrived"]),
            "avg_delay": self.metrics["syswide_total_wait"] / ticks,
            "avg_ev_wait": self.metrics["ev_syswide_total_wait"] / ev_obs,
            "total_departed": float(self.metrics["total_departed"]),
            "vehicles_stuck_in_network": vehicles_stuck,
            # DIAGNOSTIC-only metrics: scoped to just the ~16 immediate
            # traffic-light approach lanes, sampled once per macro-step.
            # Useful for spillback/switch-avoidance debugging, but NOT
            # comparable to avg_delay above -- keep these out of any
            # cross-method leaderboard table.
            "avg_controlled_lane_delay_sec": self.metrics["total_wait"] / samples,
            "avg_controlled_lane_queue_len": self.metrics["total_queue"] / samples,
            "avg_controlled_lane_ev_delay_sec": self.metrics["ev_total_wait"] / ev_samples,
            "spillback_events": float(self.metrics["spillback_events"]),
        }

    def _advance_ticks(self, n_ticks):
        """ADDED: advance the SUMO simulation by n_ticks elementary seconds,
        sampling throughput and NETWORK-WIDE delay/EV metrics on EVERY tick
        (not once per ~13-second macro-step). This mirrors exactly how a
        naive per-second baseline evaluation loop measures things (sum
        traci.vehicle.getWaitingTime() over every vehicle in the network,
        every simulated second), so avg_delay_sec / avg_ev_delay_sec below
        are directly comparable across every evaluation script that uses
        this same per-second definition -- rather than the narrower,
        controlled-lanes-only, once-per-macro-step figures previously used,
        which measure a genuinely different (smaller, sparser) quantity.
        """
        arrived = 0
        departed = 0
        for _ in range(n_ticks):
            traci.simulationStep()
            self.step_count += 1
            arrived += traci.simulation.getArrivedNumber()
            departed += traci.simulation.getDepartedNumber()

            for veh_id in traci.vehicle.getIDList():
                w = traci.vehicle.getWaitingTime(veh_id)
                self.metrics["syswide_total_wait"] += w
                # NOTE: matches the "emergency" in getTypeID(...) substring
                # check already used by evaluate_traditional()/evaluate_rl_model()
                # in the independent and neighbor-CTDE test scripts, so EV
                # figures are computed identically everywhere. This is
                # metrics-only bookkeeping -- it does not affect the EV
                # detection used in observations/reward (_extract_local_features,
                # _calculate_reward), which stays untouched to avoid invalidating
                # already-trained models.
                if "emergency" in traci.vehicle.getTypeID(veh_id):
                    self.metrics["ev_syswide_total_wait"] += w
                    self.metrics["ev_syswide_observations"] += 1
            self.metrics["elementary_ticks"] += 1
        return arrived, departed

    def step(self, actions):
        switched = {agent: False for agent in self.agents}

        # Step 1: Enforce MIN_GREEN and MAX_GREEN constraints
        for agent, target_phase in actions.items():
            current_p = self.current_phases[agent]

            # If agent wants to switch but hasn't satisfied min_green_time, override action
            if target_phase != current_p and self.time_in_green[agent] < self.min_green_time:
                target_phase = current_p

            # Hardware Failsafe: Force phase change if max_green_time exceeded
            if self.time_in_green[agent] >= self.max_green_time and target_phase == current_p:
                target_phase = 1 - current_p

            if target_phase != current_p:
                traci.trafficlight.setPhase(agent, self.phase_map[current_p]["yellow"])
                traci.trafficlight.setPhaseDuration(agent, self.HOLD_DURATION)  # FIX: lock duration
                self.is_yellow[agent] = True
                self.current_phases[agent] = target_phase
                # FIX: credit this macro-step's green time (after yellow clears)
                # rather than zeroing it out entirely, so max_green_time
                # enforcement doesn't silently drift.
                self.time_in_green[agent] = self.step_duration
                switched[agent] = True
            else:
                # FIX: previously this branch made NO TraCI call at all, which
                # meant SUMO's own tlLogic program was free to auto-advance the
                # signal once its default phase duration elapsed -- completely
                # bypassing the RL agent's decision. Re-assert and re-lock the
                # phase every macro-step so the RL agent has full authority.
                traci.trafficlight.setPhase(agent, self.phase_map[current_p]["green"])
                traci.trafficlight.setPhaseDuration(agent, self.HOLD_DURATION)
                self.time_in_green[agent] += self.step_duration

        # Step 2: Yellow Clearance Phase
        arrived_this_step, departed_this_step = self._advance_ticks(self.yellow_duration)

        # Step 3: Switch to Target Green Phase
        for agent in self.agents:
            if self.is_yellow[agent]:
                p = self.current_phases[agent]
                traci.trafficlight.setPhase(agent, self.phase_map[p]["green"])
                traci.trafficlight.setPhaseDuration(agent, self.HOLD_DURATION)  # FIX: lock duration
                self.is_yellow[agent] = False

        # Step 4: Green Step
        arrived_more, departed_more = self._advance_ticks(self.step_duration)
        arrived_this_step += arrived_more
        departed_this_step += departed_more

        self.metrics["total_arrived"] += arrived_this_step
        # ADDED: total_departed tells you how many vehicles actually entered
        # the network (vs. were still waiting to be inserted because the
        # entry edge itself was jammed). Comparing departed vs. arrived vs.
        # what's still on the network at episode end pinpoints whether low
        # throughput is a "never got in" problem or a "got in, got stuck"
        # problem.
        self.metrics["total_departed"] += departed_this_step
        self._update_episode_metrics()

        # Step 5: Observations & Rewards
        next_obs = self._get_all_observations()
        rewards = {agent: self._calculate_reward(agent, switched[agent], arrived_this_step) for agent in self.agents}

        done = self.step_count >= self.max_steps
        terms = {agent: done for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        infos = {agent: {"switched": switched[agent]} for agent in self.agents}

        return next_obs, rewards, terms, truncs, infos

    def close(self):
        """Safely terminates the TraCI instance."""
        try:
            traci.close()
        except Exception:
            pass