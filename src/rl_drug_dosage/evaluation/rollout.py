"""Evaluation and visualization utilities for trained agents."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_drug_dosage.agents.continuous import ContinuousAgent
from rl_drug_dosage.envs.drug_dose import DrugDoseEnv


class _DummyEnvs:
    """Dummy envs wrapper for loading agent."""

    def __init__(self, env: DrugDoseEnv):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space


def evaluate_agent(
    agent: ContinuousAgent,
    env: DrugDoseEnv,
    num_episodes: int = 5,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int = 1,
) -> dict:
    """Lightweight evaluation for use during training.

    Args:
        agent: The agent to evaluate.
        env: The environment to evaluate in.
        num_episodes: Number of episodes to run.
        deterministic: Use mean action (True) or sample (False).
        device: Device for inference.
        seed: Base random seed.

    Returns:
        Dictionary with mean_reward, std_reward, mean_length, mean_in_range_pct.
    """
    agent.eval()
    episode_rewards = []
    episode_lengths = []
    in_range_pcts = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        steps = 0
        in_range_steps = 0

        for _ in range(env.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    action = agent.get_deterministic_action(obs_t)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1

            # Track time in therapeutic range
            if env.c_min <= info["C"] <= env.c_max:
                in_range_steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        in_range_pcts.append(100.0 * in_range_steps / steps if steps > 0 else 0.0)

    agent.train()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_in_range_pct": np.mean(in_range_pcts),
    }


def run_evaluation(
    model_path: Path,
    output_dir: Path | None = None,
    num_episodes: int = 1,
    seed: int = 1,
    device: str = "cpu",
    save_data: bool = False,
    deterministic: bool = True,
    track: bool = False,
    wandb_project: str = "rl-drug-dosage",
    wandb_entity: str | None = None,
    wandb_run_id: str | None = None,
    wandb_tags: list[str] | None = None,
) -> dict:
    """Evaluate a trained agent and generate visualization plots.

    Args:
        model_path: Path to the trained model (.pt file).
        output_dir: Directory to save plots. Defaults to model's directory.
        num_episodes: Number of episodes to evaluate.
        seed: Random seed for reproducibility.
        device: Device to run inference on (cpu/cuda).
        save_data: Whether to save rollout data as CSV.
        deterministic: Use mean action (True) or sample from policy (False).
        track: Whether to track with Weights & Biases.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (team/user).
        wandb_run_id: Optional W&B run ID to resume.
        wandb_tags: Optional list of W&B tags.

    Returns:
        Dictionary with evaluation results including wandb_url if tracking.
    """
    from rl_drug_dosage.utils.wandb_utils import WandbLogger

    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B logger
    logger = WandbLogger(
        track=track,
        project=wandb_project,
        entity=wandb_entity,
        run_id=wandb_run_id,
        tags=wandb_tags,
        name=f"eval-{model_path.stem}",
    )
    logger.init(
        config={
            "model_path": str(model_path),
            "num_episodes": num_episodes,
            "seed": seed,
            "device": device,
            "deterministic": deterministic,
        }
    )

    # Create environment
    env = DrugDoseEnv(max_steps=100, randomize_k=True, noise_std=0.01)

    # Create agent and load weights
    dummy_envs = _DummyEnvs(env)
    agent = ContinuousAgent(dummy_envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()

    all_results = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        concentrations = []
        doses = []
        rewards_list = []

        for _ in range(env.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    action = agent.get_deterministic_action(obs_t)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t)
            action_np = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action_np)
            concentrations.append(info["C"])
            doses.append(info["dose"])
            rewards_list.append(reward)

            if terminated or truncated:
                break

        # Calculate in-range percentage
        in_range_steps = sum(1 for c in concentrations if env.c_min <= c <= env.c_max)
        in_range_pct = 100.0 * in_range_steps / len(concentrations) if concentrations else 0.0

        episode_results = {
            "episode": episode,
            "concentrations": concentrations,
            "doses": doses,
            "rewards": rewards_list,
            "total_reward": sum(rewards_list),
            "steps": len(concentrations),
            "in_range_pct": in_range_pct,
        }
        all_results.append(episode_results)

        # Log per-episode metrics to W&B
        logger.log({
            "episode/reward": episode_results["total_reward"],
            "episode/steps": episode_results["steps"],
            "episode/in_range_pct": in_range_pct,
        }, step=episode)

        print(
            f"Episode {episode + 1}: Total reward = {episode_results['total_reward']:.2f}, "
            f"Steps = {episode_results['steps']}, In range = {in_range_pct:.1f}%"
        )

    # Generate plots for the last episode
    concentrations = all_results[-1]["concentrations"]
    doses = all_results[-1]["doses"]

    # Concentration plot
    plt.figure(figsize=(10, 4))
    plt.axhspan(env.c_min, env.c_max, alpha=0.2, color="green", label="Therapeutic window")
    plt.plot(concentrations, "b-", linewidth=1.5, label="Concentration")
    plt.axhline(y=env.c_min, color="g", linestyle="--", alpha=0.5)
    plt.axhline(y=env.c_max, color="g", linestyle="--", alpha=0.5)
    plt.xlabel("Timestep")
    plt.ylabel("Concentration C")
    plt.title("Drug Concentration During Rollout")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    concentration_plot_path = output_dir / "rollout_concentration.png"
    plt.savefig(concentration_plot_path, dpi=200)
    plt.close()

    # Dose plot
    plt.figure(figsize=(10, 4))
    plt.plot(doses, "r-", linewidth=1.5)
    plt.xlabel("Timestep")
    plt.ylabel("Dose")
    plt.title("Dose Actions During Rollout")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    dose_plot_path = output_dir / "rollout_dose.png"
    plt.savefig(dose_plot_path, dpi=200)
    plt.close()

    # Concentration vs Dose scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(doses, concentrations, alpha=0.6, c=range(len(doses)), cmap="viridis")
    plt.colorbar(scatter, label="Timestep")
    plt.axhline(y=env.c_min, color="g", linestyle="--", alpha=0.5, label="Therapeutic range")
    plt.axhline(y=env.c_max, color="g", linestyle="--", alpha=0.5)
    plt.xlabel("Dose")
    plt.ylabel("Concentration")
    plt.title("Concentration vs Dose (colored by timestep)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    conc_dose_plot_path = output_dir / "concentration_vs_dose.png"
    plt.savefig(conc_dose_plot_path, dpi=200)
    plt.close()

    print(f"Saved plots to: {output_dir}")

    # Log plots to W&B
    logger.log_image(
        "plots/concentration", concentration_plot_path, caption="Drug concentration over time"
    )
    logger.log_image("plots/dose", dose_plot_path, caption="Dose actions over time")
    logger.log_image(
        "plots/concentration_vs_dose", conc_dose_plot_path, caption="Concentration vs Dose"
    )

    # Save data if requested
    if save_data:
        import csv

        csv_path = output_dir / "rollout_data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "concentration", "dose", "reward"])
            for i, (c, d, r) in enumerate(zip(concentrations, doses, all_results[-1]["rewards"])):
                writer.writerow([i, c, d, r])
        print(f"Saved data to: {csv_path}")

    # Calculate summary statistics
    mean_reward = np.mean([r["total_reward"] for r in all_results])
    std_reward = np.std([r["total_reward"] for r in all_results])
    mean_in_range_pct = np.mean([r["in_range_pct"] for r in all_results])

    # Log summary metrics to W&B
    logger.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
        "eval/mean_in_range_pct": mean_in_range_pct,
        "eval/num_episodes": num_episodes,
    })

    # Log episode summary table to W&B
    table_data = [
        [r["episode"], r["total_reward"], r["steps"], r["in_range_pct"]] for r in all_results
    ]
    logger.log_table(
        "eval/episode_summary",
        columns=["Episode", "Reward", "Steps", "In Range %"],
        data=table_data,
    )

    # Get W&B URL before finishing
    wandb_url = logger.run_url

    # Finish W&B run
    logger.finish()

    return {
        "episodes": all_results,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_in_range_pct": mean_in_range_pct,
        "concentration_plot": concentration_plot_path,
        "dose_plot": dose_plot_path,
        "conc_dose_plot": conc_dose_plot_path,
        "wandb_url": wandb_url,
    }
