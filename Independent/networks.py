import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =====================================================================
# ADDED: Orthogonal initialization helper.
#
# By default, nn.Linear uses Kaiming-uniform init, which for small networks
# like these can produce an arbitrarily skewed initial output distribution
# rather than a near-uniform one. For MAPPOActor in particular, this means
# the policy can start out already biased toward one action (e.g. "stay
# green"), and combined with a training budget of only ~1000 episodes plus
# the env's w_switch penalty, that early bias can lock in before the agent
# ever explores switching enough to learn it's sometimes beneficial --
# producing a policy that looks close to a rigid static timer.
#
# Orthogonal init with a small std (0.01) on the actor's final layer is the
# standard fix (used e.g. in CleanRL's PPO reference implementation): it
# starts the policy close to a uniform distribution over actions, giving the
# agent an unbiased starting point to explore from. Hidden layers use the
# common sqrt(2) gain; value/Q-value output layers use std=1.0.
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# =====================================================================
# 1. VALUE-BASED NETWORKS (DQN, D2QN, D3QN)
# =====================================================================

class DQNActor(nn.Module):
    """
    Standard Deep Q-Network. Used for both DQN and D2QN.
    Takes localized neighbor-padded observation (42 dims) -> Outputs 2 Q-values.
    """
    def __init__(self, obs_dim, action_dim):
        super(DQNActor, self).__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.out = layer_init(nn.Linear(64, action_dim), std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DuelingDQNActor(nn.Module):
    """
    Dueling Deep Q-Network (D3QN).
    Separates state-value approximation V(s) from action-advantage A(s,a).
    """
    def __init__(self, obs_dim, action_dim):
        super(DuelingDQNActor, self).__init__()
        # Shared feature extraction
        self.fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))

        # Value stream V(s)
        self.val_fc = layer_init(nn.Linear(64, 64))
        self.val_out = layer_init(nn.Linear(64, 1), std=1.0)

        # Advantage stream A(s, a)
        self.adv_fc = layer_init(nn.Linear(64, 64))
        self.adv_out = layer_init(nn.Linear(64, action_dim), std=1.0)

    def forward(self, x):
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))

        val = F.relu(self.val_fc(features))
        val = self.val_out(val)

        adv = F.relu(self.adv_fc(features))
        adv = self.adv_out(adv)

        # Q(s,a) = V(s) + A(s,a) - mean(A)
        q_values = val + adv - adv.mean(dim=-1, keepdim=True)
        return q_values

class VDNMixer(nn.Module):
    """
    Value-Decomposition Network (VDN) Mixer.
    Centralized component for Value-Based CTDE. It simply sums the selected
    Q-values of all agents to form Q_tot for global loss calculation.
    """
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_q_values):
        # agent_q_values shape: (batch_size, num_agents)
        # Sums across the agent dimension
        return torch.sum(agent_q_values, dim=-1, keepdim=True)


# =====================================================================
# 2. POLICY-GRADIENT NETWORKS (MAPPO)
# =====================================================================

class MAPPOActor(nn.Module):
    """
    Decentralized Actor for MAPPO.
    Takes localized observation (42 dims) -> Outputs Categorical action distribution.
    """
    def __init__(self, obs_dim, action_dim):
        super(MAPPOActor, self).__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        # FIX: very small std on the logits layer so the policy starts close
        # to a uniform distribution over the two phases, instead of an
        # arbitrarily biased one. This gives the agent a fair, unbiased
        # starting point from which to discover when switching helps.
        self.logits_out = layer_init(nn.Linear(64, action_dim), std=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.logits_out(x)
        return logits

    def get_action(self, x, action=None):
        """Returns sampled action, log probability, and distribution entropy."""
        logits = self.forward(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

class MAPPOCritic(nn.Module):
    """
    Centralized Critic for MAPPO.
    Takes GLOBAL state (56 dims) -> Outputs single global State Value V(s_global).
    """
    def __init__(self, global_state_dim):
        super(MAPPOCritic, self).__init__()
        self.fc1 = layer_init(nn.Linear(global_state_dim, 128))
        self.fc2 = layer_init(nn.Linear(128, 64))
        self.val_out = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        return self.val_out(x)


class LocalCritic(nn.Module):
    """
    Decentralized Critic for Independent PPO (IPPO).
    Takes only the LOCAL state (14 dims) -> Outputs single Local State Value V(s).
    """
    def __init__(self, obs_dim):
        super(LocalCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)

    def forward(self, local_state):
        x = F.relu(self.fc1(local_state))
        x = F.relu(self.fc2(x))
        return self.val_out(x)