import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env import MultiIntersectionSUMOEnv
from networks import DQNActor, DuelingDQNActor, MAPPOActor, LocalCritic

# =====================================================================
# HARDWARE & HYPERPARAMETERS
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nTraining device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
else:
    print("CUDA is unavailable. Training will use the CPU.")

BATCH_SIZE = 64
GAMMA = 0.99
LR_VALUE = 5e-4
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
WARMUP_STEPS = 1500
TAU = 0.005
MAX_GRAD_NORM = 5.0
BUFFER_SIZE = 50000
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models_5x5")  # Keep 5x5 weights separate from 2x2 checkpoints.
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

VALUE_MAX_EPISODES = 200
PPO_MAX_EPISODES = 1000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)

def compute_epsilon(global_step, total_anneal_steps, eps_start=1.0, eps_min=0.05):
    frac = min(1.0, global_step / max(1, total_anneal_steps))
    return eps_start + frac * (eps_min - eps_start)

# =====================================================================
# INDEPENDENT VALUE-BASED TRAINING (IQL, ID2QN, ID3QN)
# =====================================================================
def train_independent_value_based(algorithm="I-DQN"):
    # KEY CHANGE: max_neighbors=0 drops the observation space to 14 dimensions
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000, grid_size=5, max_neighbors=0)
    writer = SummaryWriter(os.path.join(RUNS_DIR, f"5x5_INDEPENDENT_{algorithm}"))
    
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agents = env.agents
    
    if algorithm == "I-D3QN":
        q_net = DuelingDQNActor(obs_dim, action_dim).to(device)
        target_net = DuelingDQNActor(obs_dim, action_dim).to(device)
    else:
        q_net = DQNActor(obs_dim, action_dim).to(device)
        target_net = DQNActor(obs_dim, action_dim).to(device)
        
    target_net.load_state_dict(q_net.state_dict())
    
    # KEY CHANGE: No VDN Mixer. Optimizer only handles the Q-Network.
    optimizer = optim.Adam(q_net.parameters(), lr=LR_VALUE)
    buffer = ReplayBuffer(BUFFER_SIZE)
    
    epsilon_min = 0.05
    global_step = 0
    macro_steps_per_ep = max(1, env.max_steps // (env.yellow_duration + env.step_duration))
    total_anneal_steps = int(VALUE_MAX_EPISODES * macro_steps_per_ep * 0.7)

    print(f"\n Starting {algorithm} (Fully Independent) Training")

    for episode in range(VALUE_MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_reward = 0
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
            
            next_obs_dict, rewards_dict, terms, _, _ = env.step(actions)
            episode_reward += sum(rewards_dict.values())
            done = all(terms.values())
            
            # KEY CHANGE: We push individual agent transitions, not joint transitions
            for a in agents:
                buffer.push(
                    obs_dict[a], actions[a], rewards_dict[a], next_obs_dict[a], done
                )
            
            obs_dict = next_obs_dict
            global_step += 1
            
            if len(buffer) >= WARMUP_STEPS:
                for _ in range(2):
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(BATCH_SIZE)
                    
                    b_obs = torch.FloatTensor(np.array(b_obs)).to(device)
                    b_actions = torch.LongTensor(np.array(b_actions)).unsqueeze(1).to(device)
                    b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                    b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                    b_next_obs = torch.FloatTensor(np.array(b_next_obs)).to(device)
                    
                    # Standard Single-Agent Q-Learning Equation
                    curr_q = q_net(b_obs).gather(1, b_actions)
                    
                    with torch.no_grad():
                        if algorithm in ["I-D2QN", "I-D3QN"]:
                            argmax_a = q_net(b_next_obs).argmax(dim=1, keepdim=True)
                            max_next_q = target_net(b_next_obs).gather(1, argmax_a)
                        else:
                            max_next_q = target_net(b_next_obs).max(dim=1, keepdim=True)[0]
                            
                    target_q = b_rewards + (GAMMA * max_next_q * (1.0 - b_dones))
                    loss = nn.SmoothL1Loss()(curr_q, target_q)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=MAX_GRAD_NORM)
                    optimizer.step()
                    
                    for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                        
                writer.add_scalar("Loss/Q_Loss", loss.item(), global_step)
                
        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        print(f"Episode: {episode+1:03d}/{VALUE_MAX_EPISODES} | Team Reward: {episode_reward:8.2f} | Epsilon: {epsilon:.3f}")
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/{algorithm}_actor.pth")
    env.close()
    writer.close()

# =====================================================================
# INDEPENDENT PPO TRAINING (IPPO)
# =====================================================================
def train_ippo():
    env = MultiIntersectionSUMOEnv(use_gui=False, max_steps=1000, grid_size=5, max_neighbors=0)
    writer = SummaryWriter(os.path.join(RUNS_DIR, "5x5_INDEPENDENT_IPPO"))
    
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agents = env.agents
    
    actor = MAPPOActor(obs_dim, action_dim).to(device)
    critic = LocalCritic(obs_dim).to(device) # KEY CHANGE: Local Critic instead of Global
    
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)
    
    PPO_EPOCHS = 4
    CLIP_EPS = 0.2
    GAE_LAMBDA = 0.95
    ROLLOUT_STEPS = 200

    print(f"\n Starting IPPO (Fully Independent) Training")

    for episode in range(PPO_MAX_EPISODES):
        obs_dict, _ = env.reset()
        episode_reward = 0
        
        batch_obs, batch_actions, batch_logprobs = [], [], []
        batch_rewards, batch_values, batch_dones = [], [], []
        
        done = False
        step_count = 0
        
        while not done and step_count < ROLLOUT_STEPS:
            actions = {}
            log_probs = {}
            
            with torch.no_grad():
                for a in agents:
                    obs_tensor = torch.FloatTensor(obs_dict[a]).unsqueeze(0).to(device)
                    action, log_prob, _ = actor.get_action(obs_tensor)
                    val = critic(obs_tensor) # Critic evaluates local state
                    
                    actions[a] = action.item()
                    log_probs[a] = log_prob.item()
                    
                    batch_obs.append(obs_dict[a])
                    batch_actions.append(action.item())
                    batch_logprobs.append(log_prob.item())
                    batch_values.append(val.item())
                    
            next_obs_dict, rewards_dict, terms, _, _ = env.step(actions)
            team_reward = sum(rewards_dict.values())
            episode_reward += team_reward
            done = all(terms.values())
            
            for a in agents:
                batch_rewards.append(rewards_dict[a]) # Use local reward
                batch_dones.append(done)
                
            obs_dict = next_obs_dict
            step_count += 1
            
        # Advantage Calculation (GAE) per agent transition
        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(batch_rewards))):
            mask = 1.0 - float(batch_dones[i])
            if i == len(batch_rewards) - 1 or batch_dones[i]:
                next_val = 0.0 # Simplify end of horizon for independence
            else:
                next_val = batch_values[i + len(agents)]
                
            delta = batch_rewards[i] + GAMMA * next_val * mask - batch_values[i]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + batch_values[i])
            
        b_obs = torch.FloatTensor(np.array(batch_obs)).to(device)
        b_actions = torch.LongTensor(np.array(batch_actions)).to(device)
        b_logprobs = torch.FloatTensor(np.array(batch_logprobs)).to(device)
        b_returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        b_advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Optimization
        for _ in range(PPO_EPOCHS):
            _, curr_logprobs, entropy = actor.get_action(b_obs, action=b_actions)
            curr_vals = critic(b_obs)
            
            ratios = torch.exp(curr_logprobs - b_logprobs)
            surr1 = ratios.unsqueeze(1) * b_advantages
            surr2 = torch.clamp(ratios.unsqueeze(1), 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            critic_loss = nn.SmoothL1Loss()(curr_vals, b_returns)
            
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer_actor.step()
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer_critic.step()
            
        writer.add_scalar("Reward/Team_Reward", episode_reward, episode)
        print(f"Episode: {episode+1:04d}/{PPO_MAX_EPISODES} | Team Reward: {episode_reward:8.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(actor.state_dict(), f"{MODEL_DIR}/IPPO_actor.pth")
    env.close()
    writer.close()

if __name__ == "__main__":
    print("\nSelect INDEPENDENT algorithm to train:")
    print(" [1] I-DQN")
    print(" [2] I-D2QN")
    print(" [3] I-D3QN")
    print(" [4] IPPO")
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1": train_independent_value_based("I-DQN")
    elif choice == "2": train_independent_value_based("I-D2QN")
    elif choice == "3": train_independent_value_based("I-D3QN")
    elif choice == "4": train_ippo()