"""PPO training for continuous action spaces."""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_drug_dosage.agents.continuous import ContinuousAgent
from rl_drug_dosage.envs.drug_dose import DrugDoseEnv
from rl_drug_dosage.evaluation.rollout import evaluate_agent


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""

    gym_id: str = "DrugDose-v0"
    total_timesteps: int = 200000
    learning_rate: float = 3e-4
    seed: int = 1
    num_envs: int = 1
    num_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    norm_adv: bool = True
    clip_vloss: bool = True
    target_kl: float | None = None
    track: bool = False
    wandb_project: str = "rl-drug-dosage"
    wandb_entity: str | None = None
    capture_video: bool = False
    cuda: bool = True
    output_dir: Path = Path("runs")
    exp_name: str | None = None
    torch_deterministic: bool = True
    use_gae: bool = True


def make_env(seed: int, idx: int, capture_video: bool, run_name: str):
    """Create a DrugDoseEnv environment factory."""

    def thunk():
        env = DrugDoseEnv(
            max_steps=100,
            randomize_k=True,
            noise_std=0.01,
            max_dose=1.0,
            c_min=2.0,
            c_max=3.0,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def train_continuous(
    gym_id: str = "DrugDose-v0",
    total_timesteps: int = 200000,
    learning_rate: float = 3e-4,
    seed: int = 1,
    num_envs: int = 1,
    num_steps: int = 2048,
    num_minibatches: int = 32,
    update_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    anneal_lr: bool = True,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    target_kl: float | None = None,
    track: bool = False,
    wandb_project: str = "rl-drug-dosage",
    wandb_entity: str | None = None,
    capture_video: bool = False,
    cuda: bool = True,
    output_dir: Path = Path("runs"),
    exp_name: str | None = None,
    torch_deterministic: bool = True,
    use_gae: bool = True,
    eval_freq: int = 10,
    eval_episodes: int = 5,
) -> Path:
    """Train a PPO agent with continuous actions on the DrugDoseEnv.

    Args:
        eval_freq: Evaluate every N updates (0 to disable).
        eval_episodes: Number of evaluation episodes.

    Returns:
        Path to the saved model.
    """
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches

    if exp_name is None:
        exp_name = "ppo_continuous"
    run_name = f"{gym_id}__{exp_name}__{seed}__{int(time.time())}"

    if track:
        import wandb

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            sync_tensorboard=True,
            config={
                "gym_id": gym_id,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "seed": seed,
                "num_envs": num_envs,
                "num_steps": num_steps,
                "num_minibatches": num_minibatches,
                "update_epochs": update_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_coef": clip_coef,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "eval_freq": eval_freq,
                "eval_episodes": eval_episodes,
            },
            name=run_name,
            save_code=True,
        )

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(
            [
                f"|{k}|{v}|"
                for k, v in {
                    "gym_id": gym_id,
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "num_envs": num_envs,
                    "num_steps": num_steps,
                }.items()
            ]
        ),
    )

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(seed + i, i, capture_video, run_name) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = ContinuousAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Evaluation environment (separate from training)
    eval_env = None
    if eval_freq > 0:
        eval_env = DrugDoseEnv(max_steps=100, randomize_k=True, noise_std=0.01)

    # Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            # Log episode stats
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if use_gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

        # Periodic evaluation
        if eval_freq > 0 and update % eval_freq == 0:
            eval_results = evaluate_agent(
                agent=agent,
                env=eval_env,
                num_episodes=eval_episodes,
                deterministic=True,
                device=str(device),
                seed=seed,
            )
            writer.add_scalar("eval/mean_reward", eval_results["mean_reward"], global_step)
            writer.add_scalar("eval/std_reward", eval_results["std_reward"], global_step)
            writer.add_scalar("eval/mean_length", eval_results["mean_length"], global_step)
            writer.add_scalar("eval/in_range_pct", eval_results["mean_in_range_pct"], global_step)

            print(
                f"Eval: reward={eval_results['mean_reward']:.2f}, "
                f"in_range={eval_results['mean_in_range_pct']:.1f}%"
            )

    # Save model
    model_path = run_dir / "agent.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"Saved agent to {model_path}")

    envs.close()
    if eval_env is not None:
        eval_env.close()
    writer.close()

    return model_path
