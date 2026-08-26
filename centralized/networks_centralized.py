import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =====================================================================
# ACTION DECODER FOR ENV INTEGRATION
# =====================================================================

def decode_action(joint_action: int, agents: list):
    """
    Converts a single integer (0-15) into a dictionary of individual actions (0 or 1).
    Example: joint_action = 6 -> binary '0110'
    Returns: {"TL_0": 0, "TL_1": 1, "TL_2": 1, "TL_3": 0}
    """
    # format into a binary string with leading zeros matching the number of agents
    binary_str = format(joint_action, f'0{len(agents)}b')
    return {agent: int(bit) for agent, bit in zip(agents, binary_str)}


# =====================================================================
# VALUE-BASED CENTRALIZED NETWORKS (DQN, D2QN, D3QN)
# =====================================================================

class CentralizedDQNActor(nn.Module):
    """
    Centralized Deep Q-Network for DQN and D2QN.
    Takes GLOBAL observation (56 dims) -> Outputs 16 Joint Q-values.
    """
    def __init__(self, global_obs_dim, joint_action_dim):
        super(CentralizedDQNActor, self).__init__()
        # Wider layers to handle the combinatorial explosion of the action space
        self.fc1 = nn.Linear(global_obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, joint_action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class CentralizedDuelingDQNActor(nn.Module):
    """
    Centralized Dueling Deep Q-Network (D3QN).
    Separates Global State-Value V(s) from Joint Action-Advantage A(s, a).
    """
    def __init__(self, global_obs_dim, joint_action_dim):
        super(CentralizedDuelingDQNActor, self).__init__()
        # Shared feature extraction
        self.fc1 = nn.Linear(global_obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value stream V(s) -> Output 1
        self.val_fc = nn.Linear(128, 128)
        self.val_out = nn.Linear(128, 1)
        
        # Advantage stream A(s, a) -> Output 16
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128, joint_action_dim)

    def forward(self, x):
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        
        val = self.val_out(F.relu(self.val_fc(features)))
        adv = self.adv_out(F.relu(self.adv_fc(features)))
        
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        q_values = val + adv - adv.mean(dim=-1, keepdim=True)
        return q_values


# =====================================================================
# POLICY-GRADIENT CENTRALIZED NETWORKS (PPO)
# =====================================================================

class CentralizedPPOActor(nn.Module):
    """
    Centralized Actor for PPO.
    Takes GLOBAL observation (56 dims) -> Outputs Categorical distribution over 16 actions.
    """
    def __init__(self, global_obs_dim, joint_action_dim):
        super(CentralizedPPOActor, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.logits_out = nn.Linear(128, joint_action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_out(x)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

class CentralizedPPOCritic(nn.Module):
    """
    Centralized Critic for PPO.
    Takes GLOBAL observation (56 dims) -> Outputs a single Global State Value V(s).
    """
    def __init__(self, global_obs_dim):
        super(CentralizedPPOCritic, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.val_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.val_out(x)