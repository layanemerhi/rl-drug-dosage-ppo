"""Tests for evaluation functionality."""

import torch

from rl_drug_dosage.agents.continuous import ContinuousAgent
from rl_drug_dosage.envs.drug_dose import DrugDoseEnv
from rl_drug_dosage.evaluation.rollout import evaluate_agent


class _DummyEnvs:
    """Dummy envs wrapper for creating agent."""

    def __init__(self, env: DrugDoseEnv):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


def test_deterministic_action_returns_mean():
    """Test that get_deterministic_action returns the actor mean."""
    env = DrugDoseEnv()
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)
    agent.eval()

    obs = torch.tensor([[2.5]], dtype=torch.float32)

    with torch.no_grad():
        deterministic_action = agent.get_deterministic_action(obs)
        expected_mean = agent.actor_mean(obs)

    assert torch.allclose(deterministic_action, expected_mean)


def test_deterministic_evaluation_is_reproducible():
    """Test that deterministic evaluation produces identical results across runs."""
    env = DrugDoseEnv(max_steps=50, randomize_k=True, noise_std=0.01)
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)

    # Run evaluation twice with the same seed
    results1 = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=3,
        deterministic=True,
        seed=42,
    )

    results2 = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=3,
        deterministic=True,
        seed=42,
    )

    assert results1["mean_reward"] == results2["mean_reward"]
    assert results1["std_reward"] == results2["std_reward"]
    assert results1["mean_length"] == results2["mean_length"]
    assert results1["mean_in_range_pct"] == results2["mean_in_range_pct"]


def test_stochastic_evaluation_produces_variation():
    """Test that stochastic evaluation produces different results across runs."""
    env = DrugDoseEnv(max_steps=50, randomize_k=True, noise_std=0.01)
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)

    # Run stochastic evaluation multiple times
    rewards = []
    for _ in range(5):
        results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=1,
            deterministic=False,
            seed=42,
        )
        rewards.append(results["mean_reward"])

    # With stochastic sampling, we expect some variation
    # (not all rewards should be identical)
    unique_rewards = set(rewards)
    assert len(unique_rewards) > 1, "Stochastic evaluation should produce variation"


def test_deterministic_vs_stochastic_differ():
    """Test that deterministic and stochastic modes can produce different results."""
    env = DrugDoseEnv(max_steps=100, randomize_k=True, noise_std=0.01)
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)

    det_results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=10,
        deterministic=True,
        seed=42,
    )

    stoch_results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=10,
        deterministic=False,
        seed=42,
    )

    # The std_reward should differ - deterministic should have lower variance
    # across episodes since it always takes the mean action
    # (Note: with same seed for env, variation comes only from policy sampling)
    assert (
        det_results["std_reward"] != stoch_results["std_reward"]
        or det_results["mean_reward"] != stoch_results["mean_reward"]
    ), "Deterministic and stochastic modes should produce different statistics"


def test_evaluate_agent_returns_expected_keys():
    """Test that evaluate_agent returns all expected result keys."""
    env = DrugDoseEnv(max_steps=10)
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)

    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=1,
        deterministic=True,
        seed=1,
    )

    assert "mean_reward" in results
    assert "std_reward" in results
    assert "mean_length" in results
    assert "mean_in_range_pct" in results


def test_in_range_percentage_calculation():
    """Test that in_range_pct is calculated correctly."""
    env = DrugDoseEnv(max_steps=10, c_min=2.0, c_max=3.0)
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs)

    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=1,
        deterministic=True,
        seed=1,
    )

    # in_range_pct should be between 0 and 100
    assert 0.0 <= results["mean_in_range_pct"] <= 100.0
