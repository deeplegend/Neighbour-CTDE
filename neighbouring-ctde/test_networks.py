"""Tensor shape tests for the 5x5 Neighbor-CTDE networks."""
import torch
from networks import DQNActor, DuelingDQNActor, VDNMixer, MAPPOActor, MAPPOCritic


def verify_networks():
    print("Testing Neural Network Architectures for 5x5 Neighbor-CTDE...")
    batch_size = 32
    num_agents = 25
    node_feature_dim = 14
    max_neighbors = 4
    obs_dim = (1 + max_neighbors) * node_feature_dim  # 70
    global_state_dim = num_agents * node_feature_dim  # 350
    action_dim = 2

    local_obs = torch.randn(batch_size, obs_dim)
    global_state = torch.randn(batch_size, global_state_dim)
    mock_agent_q_values = torch.randn(batch_size, num_agents)

    dqn = DQNActor(obs_dim, action_dim)
    assert dqn(local_obs).shape == (batch_size, action_dim)
    print("DQN Actor: OK")

    d3qn = DuelingDQNActor(obs_dim, action_dim)
    assert d3qn(local_obs).shape == (batch_size, action_dim)
    print("D3QN Actor: OK")

    mixer = VDNMixer()
    assert mixer(mock_agent_q_values).shape == (batch_size, 1)
    print("VDN Mixer with 25 agents: OK")

    mappo_actor = MAPPOActor(obs_dim, action_dim)
    action, log_prob, entropy = mappo_actor.get_action(local_obs)
    assert action.shape == log_prob.shape == entropy.shape == (batch_size,)
    print("MAPPO Actor: OK")

    mappo_critic = MAPPOCritic(global_state_dim)
    assert mappo_critic(global_state).shape == (batch_size, 1)
    print("MAPPO Centralized Critic: OK")

    print(f"All tensor dimensions align: local_obs={obs_dim}, global_state={global_state_dim}, agents={num_agents}.")


if __name__ == "__main__":
    verify_networks()
