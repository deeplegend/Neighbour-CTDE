import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from env import MultiIntersectionSUMOEnv
from networks_centralized import (
    CentralizedDQNActor,
    CentralizedDuelingDQNActor,
    CentralizedPPOActor,
    CentralizedPPOCritic,
    decode_action
)

# =====================================================================
# HARDWARE ACCELERATOR SETUP (AMD GPU / DIRECTML)
# =====================================================================
try:
    import torch_directml
    device = torch_directml.device()
    hardware_name = f"AMD GPU (DirectML): {torch_directml.device_name(0)}"
except ImportError:
    # Fallback to ROCm (Linux) or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"

# Global Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 100
BUFFER_SIZE = 50000

# Equalized Step Budget (Identical to CTDE: 200,000 total steps)
VALUE_MAX_EPISODES = 200      # 200 episodes * 1000 steps = 200k steps
PPO_MAX_EPISODES = 1000       # 1000 episodes * 200 steps = 200k steps


class CentralizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =====================================================================
# 1. VALUE-BASED CENTRALIZED TRAINING (DQN, D2QN, D3QN)
# =====================================================================

def train_centralized_value_based(algorithm="DQN"):
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000)
    writer = SummaryWriter(f"runs/Centralized_{algorithm}")

    global_obs_dim = env.global_state_dim  # 56
    joint_action_dim = 2 ** len(env.agents)  # 2^4 = 16
    agents = env.agents

    # Initialize Centralized Networks and move to GPU
    if algorithm == "D3QN":
        q_net = CentralizedDuelingDQNActor(global_obs_dim, joint_action_dim).to(device)
        target_net = CentralizedDuelingDQNActor(global_obs_dim, joint_action_dim).to(device)
    else:
        q_net = CentralizedDQNActor(global_obs_dim, joint_action_dim).to(device)
        target_net = CentralizedDQNActor(global_obs_dim, joint_action_dim).to(device)

    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = CentralizedReplayBuffer(BUFFER_SIZE)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    steps = 0

    print(f"--- Starting Centralized {algorithm} Training on {hardware_name} ---")

    for episode in range(VALUE_MAX_EPISODES):
        env.reset()
        global_state = env.get_global_state()
        episode_reward = 0
        done = False

        while not done:
            # Epsilon-Greedy Joint Action Selection
            if random.random() < epsilon:
                joint_action = random.randint(0, joint_action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                    q_vals = q_net(state_tensor)
                    joint_action = q_vals.argmax(dim=1).item()

            # Decode single integer (0-15) into individual intersection actions
            actions_dict = decode_action(joint_action, agents)

            # Step SUMO Environment
            _, rewards_dict, terms, _, _ = env.step(actions_dict)
            next_global_state = env.get_global_state()
            team_reward = sum(rewards_dict.values())
            episode_reward += team_reward
            done = all(terms.values())

            # Store in CPU Replay Buffer
            buffer.push(global_state, joint_action, team_reward, next_global_state, done)

            global_state = next_global_state
            steps += 1

            # Optimization Step
            if len(buffer) > BATCH_SIZE:
                b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(BATCH_SIZE)

                # Move mini-batch tensors to GPU
                b_states = torch.FloatTensor(np.array(b_states)).to(device)
                b_actions = torch.LongTensor(np.array(b_actions)).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                b_next_states = torch.FloatTensor(np.array(b_next_states)).to(device)

                # Current Q(s, a)
                curr_q = q_net(b_states).gather(1, b_actions)

                # Compute Target Q(s', a')
                with torch.no_grad():
                    if algorithm in ["D2QN", "D3QN"]:
                        # Double Q-Learning: Select with online net, evaluate with target net
                        argmax_a = q_net(b_next_states).argmax(dim=1, keepdim=True)
                        max_next_q = target_net(b_next_states).gather(1, argmax_a)
                    else:
                        # Standard DQN
                        max_next_q = target_net(b_next_states).max(dim=1, keepdim=True)[0]

                target_q = b_rewards + (GAMMA * max_next_q * (1 - b_dones))
                loss = nn.MSELoss()(curr_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/MSE_Loss", loss.item(), steps)

            # Sync Target Network
            if steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        # Decay Exploration Rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        writer.add_scalar("Hyperparameters/Epsilon", epsilon, episode)
        print(f"Episode: {episode + 1}/{VALUE_MAX_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

    # Save Centralized Model
    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), f"models/Centralized_{algorithm}.pth")
    env.close()
    writer.close()


# =====================================================================
# 2. POLICY-GRADIENT CENTRALIZED TRAINING (PPO)
# =====================================================================

def train_centralized_ppo():
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000)
    writer = SummaryWriter("runs/Centralized_PPO")

    global_obs_dim = env.global_state_dim  # 56
    joint_action_dim = 2 ** len(env.agents)  # 16
    agents = env.agents

    # Initialize Centralized PPO Networks on GPU
    actor = CentralizedPPOActor(global_obs_dim, joint_action_dim).to(device)
    critic = CentralizedPPOCritic(global_obs_dim).to(device)

    optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR)

    PPO_EPOCHS = 4
    CLIP_EPS = 0.2
    GAE_LAMBDA = 0.95
    ROLLOUT_STEPS = 200

    print(f"--- Starting Centralized PPO Training on {hardware_name} ---")
    steps = 0

    for episode in range(PPO_MAX_EPISODES):
        env.reset()
        global_state = env.get_global_state()
        episode_reward = 0

        batch_states, batch_actions, batch_logprobs, batch_rewards = [], [], [], []
        batch_values, batch_dones = [], []

        done = False
        step_count = 0

        # Rollout Collection
        while not done and step_count < ROLLOUT_STEPS:
            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)

            with torch.no_grad():
                val = critic(state_tensor)
                action, log_prob, _ = actor.get_action(state_tensor)

            joint_action = action.item()
            actions_dict = decode_action(joint_action, agents)

            _, rewards_dict, terms, _, _ = env.step(actions_dict)
            next_global_state = env.get_global_state()
            team_reward = sum(rewards_dict.values())
            episode_reward += team_reward
            done = all(terms.values())

            # Store transition records
            batch_states.append(global_state)
            batch_actions.append(joint_action)
            batch_logprobs.append(log_prob.item())
            batch_rewards.append(team_reward)
            batch_values.append(val.item())
            batch_dones.append(done)

            global_state = next_global_state
            step_count += 1
            steps += 1

        # GAE Advantage Estimation
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
            next_value = critic(next_state_tensor).item()

        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(batch_rewards))):
            if i == len(batch_rewards) - 1:
                next_val = next_value
                mask = 1.0 - batch_dones[i]
            else:
                next_val = batch_values[i + 1]
                mask = 1.0 - batch_dones[i]

            delta = batch_rewards[i] + GAMMA * next_val * mask - batch_values[i]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + batch_values[i])

        # Push batch tensors to GPU
        b_states = torch.FloatTensor(np.array(batch_states)).to(device)
        b_actions = torch.LongTensor(np.array(batch_actions)).to(device)
        b_logprobs = torch.FloatTensor(np.array(batch_logprobs)).to(device)
        b_returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        b_advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)

        # Advantage Normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO Surrogate Update Epochs
        for _ in range(PPO_EPOCHS):
            _, curr_logprobs, entropy = actor.get_action(b_states, action=b_actions)
            curr_vals = critic(b_states)

            ratios = torch.exp(curr_logprobs - b_logprobs)
            surr1 = ratios.unsqueeze(1) * b_advantages
            surr2 = torch.clamp(ratios.unsqueeze(1), 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            critic_loss = nn.MSELoss()(curr_vals, b_returns)

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        writer.add_scalar("Loss/Actor_Loss", actor_loss.item(), episode)
        writer.add_scalar("Loss/Critic_Loss", critic_loss.item(), episode)
        print(f"Episode: {episode + 1}/{PPO_MAX_EPISODES} | Reward: {episode_reward:.2f}")

    # Save Centralized PPO Model
    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), "models/Centralized_PPO.pth")
    env.close()
    writer.close()


# =====================================================================
# MAIN EXECUTION DISPATCHER
# =====================================================================

if __name__ == "__main__":
    print("==============================================")
    print("  CENTRALIZED MULTI-INTERSECTION TRAINING     ")
    print("==============================================")
    print("Select algorithm to train:")
    print("[1] Centralized DQN")
    print("[2] Centralized D2QN")
    print("[3] Centralized D3QN")
    print("[4] Centralized PPO")
    choice = input("Enter choice (1-4): ")

    if choice == "1":
        train_centralized_value_based("DQN")
    elif choice == "2":
        train_centralized_value_based("D2QN")
    elif choice == "3":
        train_centralized_value_based("D3QN")
    elif choice == "4":
        train_centralized_ppo()
    else:
        print("Invalid choice. Defaulting to Centralized DQN.")
        train_centralized_value_based("DQN")