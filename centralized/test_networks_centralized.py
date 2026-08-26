import torch
from networks_centralized import (
    CentralizedDQNActor, CentralizedDuelingDQNActor, 
    CentralizedPPOActor, CentralizedPPOCritic, decode_action
)

def test_centralized_architecture():
    agents = ["TL_0", "TL_1", "TL_2", "TL_3"]
    global_obs_dim = 56
    joint_action_dim = 16  # 2^4
    batch_size = 32
    
    print("Testing Action Decoder...")
    action_13 = decode_action(13, agents) # 13 in binary is 1101
    assert action_13 == {"TL_0": 1, "TL_1": 1, "TL_2": 0, "TL_3": 1}
    print(f"Action 13 correctly decoded to: {action_13}\n")
    
    mock_global_state = torch.randn(batch_size, global_obs_dim)

    print("Testing Tensor Output Shapes...")
    
    dqn = CentralizedDQNActor(global_obs_dim, joint_action_dim)
    q_vals = dqn(mock_global_state)
    assert q_vals.shape == (batch_size, joint_action_dim)
    print(f"Centralized DQN Actor:\tOK (Output: {q_vals.shape})")

    d3qn = CentralizedDuelingDQNActor(global_obs_dim, joint_action_dim)
    d3qn_vals = d3qn(mock_global_state)
    assert d3qn_vals.shape == (batch_size, joint_action_dim)
    print(f"Centralized D3QN Actor:\tOK (Output: {d3qn_vals.shape})")

    ppo_actor = CentralizedPPOActor(global_obs_dim, joint_action_dim)
    action, log_prob, entropy = ppo_actor.get_action(mock_global_state)
    assert action.shape == (batch_size,)
    print(f"Centralized PPO Actor:\tOK (Output Action: {action.shape})")

    ppo_critic = CentralizedPPOCritic(global_obs_dim)
    val = ppo_critic(mock_global_state)
    assert val.shape == (batch_size, 1)
    print(f"Centralized PPO Critic:\tOK (Output: {val.shape})")
    
    print("\nPhase 1 (Centralized) Complete! All dimensions align perfectly.")

if __name__ == "__main__":
    test_centralized_architecture()