from env import MultiIntersectionSUMOEnv
import numpy as np

def verify_environment():
    print("Testing MultiIntersectionSUMOEnv...")
    env = MultiIntersectionSUMOEnv(
        sumocfg_path="sumo_env/square_grid.sumocfg",
        use_gui=False,
        max_steps=100
    )
    
    obs, info = env.reset()
    print("Environment reset successful!")
    print(f"Number of agents: {len(obs)}")
    
    for agent, o in obs.items():
        print(f"Agent {agent} Observation Shape: {o.shape} (Expected: (42,))")
        assert o.shape == (42,), f"Observation shape mismatch for {agent}"
        
    global_s = env.get_global_state()
    print(f"Global Joint State Shape: {global_s.shape} (Expected: (56,))")
    assert global_s.shape == (56,), "Global state shape mismatch"

    # Step the environment with dummy actions
    dummy_actions = {agent: np.random.choice([0, 1]) for agent in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(dummy_actions)
    
    print("\nStep completed successfully!")
    print("Rewards:", rewards)
    print("Terms:", terms)
    
    env.close()
    print("\nPhase 2 Verification Passed!")

if __name__ == "__main__":
    verify_environment()