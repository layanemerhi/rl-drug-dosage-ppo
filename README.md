# RL Drug Dosage Control with PPO

Reinforcement Learning proof-of-concept implementing PPO (Proximal Policy Optimization) for drug dosage control.

Based on the ICLR 2022 blog post [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

## Features

- Custom `DrugDoseEnv` simulating a 1-compartment pharmacokinetic model
- PPO implementation for continuous action spaces
- Unified CLI for training and evaluation
- TensorBoard and Weights & Biases integration
- Deterministic evaluation for reproducible results

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
# Clone the repository
git clone https://github.com/layanemerhi/rl-drug-dosage-poc.git
cd rl-drug-dosage-poc

# Install with uv (or use make install)
uv sync

# Install with optional W&B support
make install-wandb

# Install with development tools
make install-dev
```

## Weights & Biases Integration

Track experiments with [Weights & Biases](https://wandb.ai) for visualization and comparison.

### Setup

1. Install with W&B support:
   ```bash
   uv sync --extra wandb
   ```

2. Create a free account at https://wandb.ai/signup

3. Get your API key from https://wandb.ai/authorize

4. Authenticate (choose one):
   ```bash
   # Option 1: Interactive login
   wandb login

   # Option 2: Environment variable
   export WANDB_API_KEY=your_api_key_here
   ```

### Usage

```bash
# Training with tracking
uv run rl-drug-dosage train --track

# Custom project and entity
uv run rl-drug-dosage train --track --wandb-project my-project --wandb-entity my-team

# Evaluation with tracking
uv run rl-drug-dosage evaluate --model-path models/agent.pt --track

# Evaluation with custom settings and tags
uv run rl-drug-dosage evaluate \
    --model-path models/agent.pt \
    --num-episodes 10 \
    --track \
    --wandb-project my-project \
    --wandb-tags "production,v2"
```

## Usage

### Training

```bash
# Train with default settings
uv run rl-drug-dosage train

# Train with custom settings
uv run rl-drug-dosage train --total-timesteps 500000 --learning-rate 1e-4

# Train with experiment tracking
uv run rl-drug-dosage train --track --wandb-project my-project

# Quick validation run
uv run rl-drug-dosage train --total-timesteps 1000
```

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total-timesteps` | 200,000 | Total environment steps |
| `--learning-rate` | 3e-4 | Adam optimizer learning rate |
| `--num-envs` | 1 | Parallel environments |
| `--num-steps` | 2,048 | Steps per rollout |
| `--num-minibatches` | 32 | Minibatches per update |
| `--update-epochs` | 10 | PPO epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-coef` | 0.2 | PPO clip coefficient |
| `--ent-coef` | 0.0 | Entropy coefficient |
| `--vf-coef` | 0.5 | Value function coefficient |
| `--max-grad-norm` | 0.5 | Gradient clipping |

### Evaluation During Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval-freq` | 10 | Evaluate every N updates (0 to disable) |
| `--eval-episodes` | 5 | Episodes per evaluation |

Periodic evaluation uses deterministic actions (policy mean) for consistent metrics.

### Tracked Metrics

Metrics are logged to TensorBoard (always) and W&B (when `--track` is enabled).

**Training Metrics:**
- `charts/episodic_return` - Episode total reward
- `charts/episodic_length` - Episode length
- `charts/learning_rate` - Current learning rate (with annealing)
- `charts/SPS` - Steps per second (throughput)

**Loss Metrics:**
- `losses/policy_loss` - PPO policy loss
- `losses/value_loss` - Value function loss
- `losses/entropy` - Policy entropy
- `losses/approx_kl` - KL divergence estimate
- `losses/clipfrac` - Fraction of clipped updates
- `losses/explained_variance` - Value prediction quality

**Evaluation Metrics (every `--eval-freq` updates):**
- `eval/mean_reward` - Average return (deterministic policy)
- `eval/std_reward` - Standard deviation across episodes
- `eval/mean_length` - Average episode length
- `eval/in_range_pct` - Percentage of time in therapeutic window

### Evaluation

```bash
# Evaluate deterministically (default) - uses mean action for reproducibility
uv run rl-drug-dosage evaluate --model-path models/agent.pt

# Evaluate stochastically - samples from policy distribution
uv run rl-drug-dosage evaluate --model-path models/agent.pt --no-deterministic

# Multiple episodes with CSV export
uv run rl-drug-dosage evaluate --model-path models/agent.pt --num-episodes 10 --save-data

# Custom output directory
uv run rl-drug-dosage evaluate --model-path models/agent.pt --output-dir results/

# Track evaluation in W&B
uv run rl-drug-dosage evaluate --model-path models/agent.pt --track

# W&B with custom project and tags
uv run rl-drug-dosage evaluate \
    --model-path models/agent.pt \
    --num-episodes 10 \
    --track \
    --wandb-project my-project \
    --wandb-tags "eval,production"
```

**Evaluation Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | (required) | Path to trained model (.pt file) |
| `--output-dir` | model's directory | Output directory for plots |
| `--num-episodes` | 1 | Number of evaluation episodes |
| `--seed` | 1 | Random seed |
| `--device` | cpu | Device (cpu/cuda) |
| `--save-data` | False | Save rollout data as CSV |
| `--deterministic` | True | Use mean action (deterministic) |
| `--track` | False | Track with Weights & Biases |
| `--wandb-project` | rl-drug-dosage | W&B project name |
| `--wandb-entity` | None | W&B entity (team/user) |
| `--wandb-run-id` | None | W&B run ID to resume |
| `--wandb-tags` | None | Comma-separated W&B tags |

**Generated Plots:**
- `rollout_concentration.png` - Drug concentration over time with therapeutic window
- `rollout_dose.png` - Dose actions over time
- `concentration_vs_dose.png` - Scatter plot showing dose-concentration relationship (colored by timestep)

**W&B Logged Metrics (when `--track` enabled):**
- Per-episode: `episode/reward`, `episode/steps`, `episode/in_range_pct`
- Summary: `eval/mean_reward`, `eval/std_reward`, `eval/mean_in_range_pct`
- Plots uploaded as images
- Episode summary table

### Viewing Training Logs

Training runs are saved to `runs/{run_name}/` with TensorBoard logs and the trained model.

```bash
# Start TensorBoard
uv run tensorboard --logdir runs/

# View specific run
uv run tensorboard --logdir runs/DrugDose-v0__ppo_continuous__1__timestamp/
```

Open http://localhost:6006 in your browser to view training curves.

### Information

```bash
# Show project and environment info
uv run rl-drug-dosage info
```

### Alternative: Using Python directly

```bash
# Using module syntax
uv run python -m rl_drug_dosage train

# Or after activating virtual environment
source .venv/bin/activate
rl-drug-dosage train
```

## Project Structure

```
rl-drug-dosage-poc/
├── src/rl_drug_dosage/      # Main package
│   ├── cli.py               # Typer CLI
│   ├── envs/                # Gymnasium environments
│   ├── agents/              # Neural network agents
│   ├── training/            # PPO training logic
│   ├── evaluation/          # Evaluation utilities
│   └── utils/               # Shared utilities (W&B logging)
├── models/                   # Pre-trained models
├── runs/                     # Training outputs
└── tests/                    # Test suite
```

## DrugDoseEnv

The custom environment simulates a 1-compartment pharmacokinetic model:

- **Observation**: Drug concentration `C` (single float)
- **Action**: Dose in `[0, max_dose]`
- **Dynamics**: `C(t+1) = C(t) * (1-k) + dose + noise`
- **Reward**: +1 in therapeutic range `[c_min, c_max]`, else negative distance

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Run linting
make lint

# Run type checking
make typecheck

# See all available commands
make help
```

## Citation

```bibtex
@inproceedings{shengyi2022the37implementation,
  author = {Huang, Shengyi and others},
  title = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year = {2022},
  url = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}
```

## License

MIT
