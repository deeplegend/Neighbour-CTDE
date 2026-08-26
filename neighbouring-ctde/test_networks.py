import torch
from networks import DQNActor, DuelingDQNActor, VDNMixer, MAPPOActor, MAPPOCritic

def verify_networks():
    print("Testing Neural Network Architectures for CTDE...")
    
    batch_size = 32
    num_agents = 4
    obs_dim = 42
    global_state_dim = 56
    action_dim = 2

    # Mock Tensors
    mock_local_obs = torch.randn(batch_size, obs_dim)
    mock_global_state = torch.randn(batch_size, global_state_dim)
    
    # 1. Test Standard DQN Actor
    dqn = DQNActor(obs_dim, action_dim)
    q_vals = dqn(mock_local_obs)
    assert q_vals.shape == (batch_size, action_dim)
    print("DQN Actor: OK \t\t| Output shape:", q_vals.shape)

    # 2. Test Dueling DQN Actor
    d3qn = DuelingDQNActor(obs_dim, action_dim)
    d3qn_vals = d3qn(mock_local_obs)
    assert d3qn_vals.shape == (batch_size, action_dim)
    print("D3QN Actor: OK \t\t| Output shape:", d3qn_vals.shape)

    # 3. Test VDN Mixer
    # Mocking that each of the 4 agents picked an action and we got their Q-values
    mock_agent_q_vals = torch.randn(batch_size, num_agents) 
    mixer = VDNMixer()
    q_tot = mixer(mock_agent_q_vals)
    assert q_tot.shape == (batch_size, 1)
    print("VDN Mixer: OK \t\t| Output shape:", q_tot.shape)

    # 4. Test MAPPO Actor
    mappo_actor = MAPPOActor(obs_dim, action_dim)
    action, log_prob, entropy = mappo_actor.get_action(mock_local_obs)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    print("MAPPO Actor: OK \t| Action shape:", action.shape)

    # 5. Test MAPPO Centralized Critic
    mappo_critic = MAPPOCritic(global_state_dim)
    global_val = mappo_critic(mock_global_state)
    assert global_val.shape == (batch_size, 1)
    print("MAPPO Critic: OK \t| Output shape:", global_val.shape)
    
    print("\nPhase 3 Verification Passed! All tensor dimensions align for CTDE.")

if __name__ == "__main__":
    verify_networks()