.PHONY: help install install-dev install-wandb lint typecheck test train evaluate info clean

# Default target
help:
	@echo "RL Drug Dosage Control - Available commands:"
	@echo ""
	@echo "  Installation:"
	@echo "    make install        Install project dependencies"
	@echo "    make install-dev    Install with development tools"
	@echo "    make install-wandb  Install with W&B support"
	@echo ""
	@echo "  Development:"
	@echo "    make lint           Run ruff linter"
	@echo "    make typecheck      Run pyrefly type checker"
	@echo "    make test           Run pytest"
	@echo ""
	@echo "  Training & Evaluation:"
	@echo "    make train          Train PPO agent (default settings)"
	@echo "    make train-quick    Quick training run (1000 steps)"
	@echo "    make evaluate       Evaluate pre-trained model"
	@echo "    make info           Show environment info"
	@echo ""
	@echo "  Utility:"
	@echo "    make clean          Remove build artifacts and caches"

# Installation targets
install:
	uv sync

install-dev:
	uv sync --extra dev

install-wandb:
	uv sync --extra wandb

# Development targets
lint:
	uv run ruff check src/

typecheck:
	uv run pyrefly check src/

test:
	uv run pytest tests/ -v

# Training & Evaluation targets
train:
	uv run rl-drug-dosage train

train-quick:
	uv run rl-drug-dosage train --total-timesteps 1000

evaluate:
	uv run rl-drug-dosage evaluate --model-path models/agent.pt

info:
	uv run rl-drug-dosage info

# Utility targets
clean:
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf .venv/
	rm -rf dist/
	rm -rf *.egg-info/
