"""Command-line interface for RL Drug Dosage."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="rl-drug-dosage",
    help="Reinforcement Learning for Drug Dosage Control using PPO",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    total_timesteps: int = typer.Option(200000, help="Total training timesteps"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
    seed: int = typer.Option(1, help="Random seed"),
    num_envs: int = typer.Option(1, help="Number of parallel environments"),
    num_steps: int = typer.Option(2048, help="Steps per rollout"),
    num_minibatches: int = typer.Option(32, help="Number of minibatches"),
    update_epochs: int = typer.Option(10, help="PPO update epochs"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    clip_coef: float = typer.Option(0.2, help="PPO clip coefficient"),
    ent_coef: float = typer.Option(0.0, help="Entropy coefficient"),
    vf_coef: float = typer.Option(0.5, help="Value function coefficient"),
    max_grad_norm: float = typer.Option(0.5, help="Max gradient norm"),
    anneal_lr: bool = typer.Option(True, help="Anneal learning rate"),
    track: bool = typer.Option(False, help="Track with Weights & Biases"),
    wandb_project: str = typer.Option("rl-drug-dosage", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity"),
    capture_video: bool = typer.Option(False, help="Capture video"),
    cuda: bool = typer.Option(True, help="Use CUDA if available"),
    output_dir: Path = typer.Option(Path("runs"), help="Output directory"),
    exp_name: str | None = typer.Option(None, help="Experiment name"),
    eval_freq: int = typer.Option(10, help="Evaluate every N updates (0 to disable)"),
    eval_episodes: int = typer.Option(5, help="Number of evaluation episodes"),
):
    """Train a PPO agent for drug dosage control."""
    from rl_drug_dosage.training import train_continuous

    console.print(Panel("[bold green]Training PPO for Drug Dosage Control[/bold green]"))

    model_path = train_continuous(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        seed=seed,
        num_envs=num_envs,
        num_steps=num_steps,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        anneal_lr=anneal_lr,
        track=track,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        capture_video=capture_video,
        cuda=cuda,
        output_dir=output_dir,
        exp_name=exp_name,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
    )

    console.print(f"[green]Model saved to: {model_path}[/green]")


@app.command()
def evaluate(
    model_path: Path = typer.Option(..., help="Path to trained model (.pt file)"),
    output_dir: Path | None = typer.Option(None, help="Output directory for plots"),
    num_episodes: int = typer.Option(1, help="Number of episodes to evaluate"),
    seed: int = typer.Option(1, help="Random seed"),
    device: str = typer.Option("cpu", help="Device (cpu/cuda)"),
    save_data: bool = typer.Option(False, help="Save rollout data as CSV"),
    deterministic: bool = typer.Option(True, help="Use mean action (deterministic)"),
    track: bool = typer.Option(False, help="Track with Weights & Biases"),
    wandb_project: str = typer.Option("rl-drug-dosage", help="W&B project name"),
    wandb_entity: str | None = typer.Option(None, help="W&B entity"),
    wandb_run_id: str | None = typer.Option(None, help="W&B run ID to resume"),
    wandb_tags: str | None = typer.Option(None, help="Comma-separated W&B tags"),
):
    """Evaluate a trained agent and generate visualization plots."""
    from rl_drug_dosage.evaluation import run_evaluation

    console.print(Panel(f"[bold blue]Evaluating model: {model_path}[/bold blue]"))

    # Parse tags if provided
    tags = wandb_tags.split(",") if wandb_tags else None

    results = run_evaluation(
        model_path=model_path,
        output_dir=output_dir,
        num_episodes=num_episodes,
        seed=seed,
        device=device,
        save_data=save_data,
        deterministic=deterministic,
        track=track,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_id=wandb_run_id,
        wandb_tags=tags,
    )

    console.print(f"[green]Mean reward: {results['mean_reward']:.2f}[/green]")

    if results.get("wandb_url"):
        console.print(f"[blue]W&B run: {results['wandb_url']}[/blue]")


@app.command()
def info():
    """Show project and environment information."""
    from rl_drug_dosage import __version__
    from rl_drug_dosage.envs import DrugDoseEnv

    console.print(Panel(f"[bold]RL Drug Dosage v{__version__}[/bold]"))

    env = DrugDoseEnv()
    console.print("\n[bold]DrugDoseEnv Configuration:[/bold]")
    console.print(f"  Observation space: {env.observation_space}")
    console.print(f"  Action space: {env.action_space}")
    console.print(f"  Therapeutic range: [{env.c_min}, {env.c_max}]")
    console.print(f"  Max steps: {env.max_steps}")
    console.print(f"  Clearance rate (k): {env.k_base}")
    console.print(f"  Max dose: {env.max_dose}")


if __name__ == "__main__":
    app()
