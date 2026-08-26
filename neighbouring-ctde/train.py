import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, VDNMixer, MAPPOActor, MAPPOCritic

# =====================================================================
# HARDWARE CONFIGURATION (NVIDIA / AMD / APPLE SILICON)
# =====================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Check if PyTorch is compiled with ROCm (AMD) or CUDA (NVIDIA)
    if getattr(torch.version, 'hip', None) is not None:
        device_name = f"AMD GPU (ROCm) - {torch.cuda.get_device_name(0)}"
    else:
        device_name = f"NVIDIA GPU - {torch.cuda.get_device_name(0)}"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # Supports AMD GPUs running on Mac via Metal Performance Shaders
    device = torch.device("mps")
    device_name = "Apple/AMD GPU (Metal/MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"\n[Hardware Initialized: {device_name}]")

# =====================================================================
# GLOBAL HYPERPARAMETERS
# =====================================================================
BATCH_SIZE = 64
GAMMA = 0.99
LR_VALUE = 5e-4
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
WARMUP_STEPS = 1500          # Buffer fills before training starts
TAU = 0.005                  # Soft target network update rate (Polyak averaging)
MAX_GRAD_NORM = 5.0          # Gradient clipping threshold
BUFFER_SIZE = 50000

# Equalized Environment Step Budgets (Total: 200,000 steps each)
VALUE_MAX_EPISODES = 200     # 200 episodes * 1000 steps = 200,000 steps
PPO_MAX_EPISODES = 1000      # 1000 episodes * 200 steps = 200,000 steps

# ADDED: how many greedy episodes to run for a fair post-training comparison
# against a static baseline (no exploration noise).
EVAL_EPISODES = 5


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        self.buffer.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


def compute_epsilon(global_step, total_anneal_steps, eps_start=1.0, eps_min=0.05):
    """ADDED: linear epsilon anneal driven by macro-decision steps rather than
    episodes. The original code decayed epsilon once per *episode*
    (epsilon *= 0.995), which over only ~200 episodes never actually reaches
    epsilon_min -- the trained policy was still acting close to 50% randomly
    late in training. Annealing over the real step budget guarantees the
    policy is (near-)fully greedy by the time training ends, which matters a
    lot when you're about to compare it against a deterministic static plan.
    """
    frac = min(1.0, global_step / max(1, total_anneal_steps))
    return eps_start + frac * (eps_min - eps_start)


def evaluate_value_policy(env, q_net, episodes=EVAL_EPISODES):
    """ADDED: runs the trained Q-network fully greedily (epsilon=0, no
    exploration) and reports the same metrics (delay, queue, throughput,
    spillback) your methodology uses, so the RL-vs-static comparison is
    apples-to-apples instead of comparing a still-exploring policy."""
    q_net.eval()
    agents = env.agents
    results = []
    with torch.no_grad():
        for _ in range(episodes):
            obs_dict, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                actions = {}
                for a in agents:
                    obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                    q_vals = q_net(obs_tensor)
                    actions[a] = q_vals.argmax(dim=1).item()
                obs_dict, rewards_dict, terms, _, _ = env.step(actions)
                ep_reward += sum(rewards_dict.values())
                done = all(terms.values())
            m = env.get_episode_metrics()
            m["episode_reward"] = ep_reward
            results.append(m)
    q_net.train()
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def evaluate_mappo_policy(env, actor, episodes=EVAL_EPISODES):
    """ADDED: evaluation pass for MAPPO. NOTE: this still samples from the
    stochastic policy because MAPPOActor.get_action() (defined in networks.py,
    which wasn't shared) doesn't currently expose a deterministic/greedy mode.
    For a fully fair static-vs-RL comparison, consider adding a
    `deterministic=True` argument to get_action() that returns argmax(probs)
    instead of sampling."""
    agents = env.agents
    results = []
    with torch.no_grad():
        for _ in range(episodes):
            obs_dict, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                actions = {}
                for a in agents:
                    obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                    action, _, _ = actor.get_action(obs_tensor)
                    actions[a] = action.item()
                obs_dict, rewards_dict, terms, _, _ = env.step(actions)
                ep_reward += sum(rewards_dict.values())
                done = all(terms.values())
            m = env.get_episode_metrics()
            m["episode_reward"] = ep_reward
            results.append(m)
    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


# =====================================================================
# VALUE-BASED CTDE TRAINING (DQN, D2QN, D3QN via VDN)
# =====================================================================
def train_value_based(algorithm="DQN"):
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000)
    writer = SummaryWriter(f"runs/CTDE_{algorithm}")

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agents = env.agents

    # Initialize networks and push to GPU
    if algorithm == "D3QN":
        q_net = DuelingDQNActor(obs_dim, action_dim).to(device)
        target_net = DuelingDQNActor(obs_dim, action_dim).to(device)
    else:
        q_net = DQNActor(obs_dim, action_dim).to(device)
        target_net = DQNActor(obs_dim, action_dim).to(device)

    target_net.load_state_dict(q_net.state_dict())

    mixer = VDNMixer().to(device)
    target_mixer = VDNMixer().to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    optimizer = optim.Adam(list(q_net.parameters()) + list(mixer.parameters()), lr=LR_VALUE)
    buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon_min = 0.05
    global_step = 0

    # ADDED: estimate the real macro-decision-step budget so epsilon can be
    # annealed against actual training length instead of episode count.
    macro_steps_per_ep = max(1, env.max_steps // (env.yellow_duration + env.step_duration))
    total_macro_steps = VALUE_MAX_EPISODES * macro_steps_per_ep
    ANNEAL_FRACTION = 0.7
    total_anneal_steps = max(1, int(total_macro_steps * ANNEAL_FRACTION))

    print(f"\n========================================================")
    print(f" Starting {algorithm} (CTDE-VDN) Training on {device_name}")
    print(f" Target Budget: {VALUE_MAX_EPISODES} Episodes (200,000 Environment Steps)")
    print(f" Replay Warmup: {WARMUP_STEPS} Steps | Loss: Smooth L1 (Huber)")
    print(f" Epsilon anneal horizon: {total_anneal_steps} macro-steps ({ANNEAL_FRACTION*100:.0f}% of budget)")
    print(f"========================================================\n")

    for episode in range(VALUE_MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_reward = 0
        episode_switches = 0  # ADDED: diagnostic for switch-avoidance behavior
        done = False

        while not done:
            epsilon = compute_epsilon(global_step, total_anneal_steps, eps_start=1.0, eps_min=epsilon_min)

            actions = {}
            if random.random() < epsilon:
                actions = {a: random.randint(0, action_dim - 1) for a in agents}
            else:
                with torch.no_grad():
                    for a in agents:
                        obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                        q_vals = q_net(obs_tensor)
                        actions[a] = q_vals.argmax(dim=1).item()

            next_obs_dict, rewards_dict, terms, _, infos = env.step(actions)
            team_reward = sum(rewards_dict.values())
            episode_reward += team_reward
            episode_switches += sum(1 for a in agents if infos[a]["switched"])
            done = all(terms.values())

            # Store transition in RAM buffer as CPU numpy arrays
            buffer.push(
                np.array([obs_dict[a] for a in agents]),
                np.array([actions[a] for a in agents]),
                team_reward,
                np.array([next_obs_dict[a] for a in agents]),
                done
            )

            obs_dict = next_obs_dict
            global_step += 1

            # Start optimization only after warmup steps are collected
            if len(buffer) >= WARMUP_STEPS:
                for _ in range(2):
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(BATCH_SIZE)

                    b_obs = torch.FloatTensor(np.array(b_obs)).to(device)
                    b_actions = torch.LongTensor(np.array(b_actions)).to(device)
                    b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                    b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                    b_next_obs = torch.FloatTensor(np.array(b_next_obs)).to(device)

                    curr_q = []
                    for i in range(len(agents)):
                        q_a = q_net(b_obs[:, i, :])
                        act_a = b_actions[:, i].unsqueeze(1)
                        curr_q.append(q_a.gather(1, act_a))
                    curr_q = torch.cat(curr_q, dim=1)
                    q_tot = mixer(curr_q)

                    with torch.no_grad():
                        target_q = []
                        for i in range(len(agents)):
                            if algorithm in ["D2QN", "D3QN"]:
                                argmax_a = q_net(b_next_obs[:, i, :]).argmax(dim=1, keepdim=True)
                                max_q_a = target_net(b_next_obs[:, i, :]).gather(1, argmax_a)
                            else:
                                max_q_a = target_net(b_next_obs[:, i, :]).max(dim=1, keepdim=True)[0]
                            target_q.append(max_q_a)
                        target_q = torch.cat(target_q, dim=1)
                        target_q_tot = target_mixer(target_q)

                    expected_q_tot = b_rewards + (GAMMA * target_q_tot * (1.0 - b_dones))
                    loss = nn.SmoothL1Loss()(q_tot, expected_q_tot)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(q_net.parameters()) + list(mixer.parameters()),
                        max_norm=MAX_GRAD_NORM
                    )
                    optimizer.step()

                    # Soft Target Update
                    for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                    for target_param, param in zip(target_mixer.parameters(), mixer.parameters()):
                        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

                writer.add_scalar("Loss/Q_Tot_Huber_Loss", loss.item(), global_step)

        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        writer.add_scalar("Hyperparameters/Epsilon", epsilon, episode)
        # ADDED: track how often agents actually switch phase. If this trends
        # toward ~0, the agent has likely learned to avoid the w_switch
        # penalty entirely, degenerating into a rigid near-static policy.
        writer.add_scalar("Diagnostics/Total_Switches", episode_switches, episode)
        print(f"Episode: {episode+1:03d}/{VALUE_MAX_EPISODES} | Team Reward: {episode_reward:8.2f} | "
              f"Epsilon: {epsilon:.3f} | Switches: {episode_switches} | Buffer: {len(buffer)}/{BUFFER_SIZE}")

    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), f"models/{algorithm}_actor.pth")
    print(f"\nModel saved successfully at models/{algorithm}_actor.pth")

    # ADDED: fair, exploration-free comparison metrics against the static baseline
    print("\nRunning greedy (epsilon=0) evaluation for a fair comparison against the static baseline...")
    eval_metrics = evaluate_value_policy(env, q_net, episodes=EVAL_EPISODES)
    print(f"[{algorithm} Eval | {EVAL_EPISODES} episodes] "
          f"Avg Delay: {eval_metrics['avg_delay']:.2f}s | "
          f"Avg Queue: {eval_metrics['avg_queue_len']:.2f} veh | "
          f"Throughput: {eval_metrics['throughput']:.1f} veh/ep | "
          f"Spillback Events: {eval_metrics['spillback_events']:.1f} | "
          f"Reward: {eval_metrics['episode_reward']:.2f}")

    env.close()
    writer.close()


# =====================================================================
# POLICY-GRADIENT CTDE TRAINING (MAPPO)
# =====================================================================
def train_mappo():
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000)
    writer = SummaryWriter("runs/CTDE_MAPPO")

    obs_dim = env.obs_dim
    global_state_dim = env.global_state_dim
    action_dim = env.action_dim
    agents = env.agents

    # Initialize MAPPO Actor and Centralized Critic
    actor = MAPPOActor(obs_dim, action_dim).to(device)
    critic = MAPPOCritic(global_state_dim).to(device)

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    PPO_EPOCHS = 4
    CLIP_EPS = 0.2
    GAE_LAMBDA = 0.95
    ROLLOUT_STEPS = 200

    print(f"\n========================================================")
    print(f" Starting MAPPO (CTDE) Training on {device_name}")
    print(f" Target Budget: {PPO_MAX_EPISODES} Episodes (200,000 Environment Steps)")
    print(f" Rollout Horizon: {ROLLOUT_STEPS} Steps | Policy Epochs: {PPO_EPOCHS}")
    print(f"========================================================\n")

    for episode in range(PPO_MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_reward = 0
        episode_switches = 0  # ADDED: diagnostic

        batch_obs, batch_actions, batch_logprobs, batch_rewards = [], [], [], []
        batch_global_states, batch_values, batch_dones = [], [], []

        done = False
        step_count = 0

        # 1. Trajectory Rollout Collection
        while not done and step_count < ROLLOUT_STEPS:
            global_state = torch.FloatTensor(env.get_global_state()).unsqueeze(0).to(device)

            actions = {}
            log_probs = {}
            with torch.no_grad():
                val = critic(global_state)

                for a in agents:
                    obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                    action, log_prob, _ = actor.get_action(obs_tensor)
                    actions[a] = action.item()
                    log_probs[a] = log_prob.item()

            next_obs_dict, rewards_dict, terms, _, infos = env.step(actions)
            team_reward = sum(rewards_dict.values())
            episode_reward += team_reward
            episode_switches += sum(1 for a in agents if infos[a]["switched"])
            done = all(terms.values())

            batch_global_states.append(global_state)
            batch_values.append(val)
            batch_rewards.append(team_reward)
            batch_dones.append(done)

            for a in agents:
                batch_obs.append(obs_dict[a])
                batch_actions.append(actions[a])
                batch_logprobs.append(log_probs[a])

            obs_dict = next_obs_dict
            step_count += 1

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_global_state = torch.FloatTensor(env.get_global_state()).unsqueeze(0).to(device)
            next_value = critic(next_global_state)

        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(batch_rewards))):
            if i == len(batch_rewards) - 1:
                next_val = next_value.item()
                mask = 1.0 - float(batch_dones[i])
            else:
                next_val = batch_values[i + 1].item()
                mask = 1.0 - float(batch_dones[i])

            delta = batch_rewards[i] + GAMMA * next_val * mask - batch_values[i].item()
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + batch_values[i].item())

        # Assemble GPU Tensors
        b_obs = torch.FloatTensor(np.array(batch_obs)).to(device)
        b_actions = torch.LongTensor(np.array(batch_actions)).to(device)
        b_logprobs = torch.FloatTensor(np.array(batch_logprobs)).to(device)

        # Broadcast global returns and advantages across all agents
        b_returns = torch.FloatTensor(returns).repeat_interleave(len(agents)).unsqueeze(1).to(device)
        b_advantages = torch.FloatTensor(advantages).repeat_interleave(len(agents)).unsqueeze(1).to(device)
        b_global_states = torch.cat(batch_global_states).repeat_interleave(len(agents), dim=0)

        # Standardize Advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 3. PPO Optimization Epochs
        actor_loss_val = 0.0
        critic_loss_val = 0.0
        for _ in range(PPO_EPOCHS):
            _, curr_logprobs, entropy = actor.get_action(b_obs, action=b_actions)
            curr_vals = critic(b_global_states)

            # Clipped Surrogate Objective
            ratios = torch.exp(curr_logprobs - b_logprobs)
            surr1 = ratios.unsqueeze(1) * b_advantages
            surr2 = torch.clamp(ratios.unsqueeze(1), 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            # Critic Huber/Smooth L1 Loss
            critic_loss = nn.SmoothL1Loss()(curr_vals, b_returns)

            # Backpropagation with Gradient Clipping
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer_critic.step()

            actor_loss_val = actor_loss.item()
            critic_loss_val = critic_loss.item()

        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        writer.add_scalar("Loss/Actor_Loss", actor_loss_val, episode)
        writer.add_scalar("Loss/Critic_Loss", critic_loss_val, episode)
        writer.add_scalar("Diagnostics/Total_Switches", episode_switches, episode)  # ADDED
        print(f"Episode: {episode+1:04d}/{PPO_MAX_EPISODES} | Team Reward: {episode_reward:8.2f} | "
              f"Actor Loss: {actor_loss_val:6.3f} | Critic Loss: {critic_loss_val:6.3f} | Switches: {episode_switches}")

    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), "models/MAPPO_actor.pth")
    print(f"\nModel saved successfully at models/MAPPO_actor.pth")

    # ADDED: comparison metrics (see evaluate_mappo_policy docstring for the
    # determinism caveat re: networks.py)
    print("\nRunning evaluation for a comparison against the static baseline...")
    eval_metrics = evaluate_mappo_policy(env, actor, episodes=EVAL_EPISODES)
    print(f"[MAPPO Eval | {EVAL_EPISODES} episodes] "
          f"Avg Delay: {eval_metrics['avg_delay']:.2f}s | "
          f"Avg Queue: {eval_metrics['avg_queue_len']:.2f} veh | "
          f"Throughput: {eval_metrics['throughput']:.1f} veh/ep | "
          f"Spillback Events: {eval_metrics['spillback_events']:.1f} | "
          f"Reward: {eval_metrics['episode_reward']:.2f}")

    env.close()
    writer.close()


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("\nSelect algorithm to train:")
    print(" [1] DQN")
    print(" [2] D2QN")
    print(" [3] D3QN")
    print(" [4] PPO (MAPPO)")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        train_value_based("DQN")
    elif choice == "2":
        train_value_based("D2QN")
    elif choice == "3":
        train_value_based("D3QN")
    elif choice == "4":
        train_mappo()
    else:
        print("Invalid choice. Defaulting to DQN.")
        train_value_based("DQN")