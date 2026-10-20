"""Quick environment smoke test for the 5x5 Neighbor-CTDE grid."""
from env import MultiIntersectionSUMOEnv
import numpy as np


def verify_environment():
    print("Testing MultiIntersectionSUMOEnv on a 5x5 grid...")
    env = MultiIntersectionSUMOEnv(
        sumocfg_path="sumo_env/square_grid.sumocfg",
        use_gui=False,
        max_steps=100,
        grid_size=5,
        max_neighbors=4,
    )

    obs, _ = env.reset()
    print("Environment reset successful!")
    print(f"Number of agents: {len(obs)} (expected: 25)")
    assert len(obs) == 25, f"Expected 25 signalized agents, got {len(obs)}"

    for agent, observation in obs.items():
        assert observation.shape == (env.obs_dim,), (
            f"Observation shape mismatch for {agent}: {observation.shape}, expected {(env.obs_dim,)}"
        )
    print(f"Each local observation: ({env.obs_dim},) = ego (14) + 4 neighbor slots (56)")

    global_state = env.get_global_state()
    print(f"Global state shape: {global_state.shape} (expected: ({25 * 14},))")
    assert global_state.shape == (350,), f"Global state shape mismatch: {global_state.shape}"

    dummy_actions = {agent: int(np.random.choice([0, 1])) for agent in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(dummy_actions)
    assert len(next_obs) == 25 and len(rewards) == 25
    print("One multi-agent environment step completed successfully.")
    print(f"Sample rewards: {dict(list(rewards.items())[:3])}")

    env.close()
    print("5x5 environment smoke test passed.")


if __name__ == "__main__":
    verify_environment()
