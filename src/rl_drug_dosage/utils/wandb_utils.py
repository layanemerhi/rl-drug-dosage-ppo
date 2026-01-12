"""Weights & Biases utilities with graceful fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb as wandb_module


def is_wandb_available() -> bool:
    """Check if wandb is installed and available."""
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        return False


def require_wandb() -> None:
    """Raise a helpful error if wandb is not installed."""
    if not is_wandb_available():
        raise ImportError(
            "wandb is not installed. Install it with:\n"
            "  uv sync --extra wandb\n"
            "or:\n"
            "  pip install wandb"
        )


class WandbLogger:
    """Wrapper for W&B logging with graceful fallbacks.

    When `track=False` or wandb is unavailable, all methods become no-ops.

    Example:
        logger = WandbLogger(track=True, project="my-project")
        logger.init(config={"lr": 0.001})
        logger.log({"loss": 0.5})
        logger.log_image("plot", image_path)
        logger.finish()
    """

    def __init__(
        self,
        track: bool = False,
        project: str = "rl-drug-dosage",
        entity: str | None = None,
        run_id: str | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
    ):
        """Initialize the W&B logger.

        Args:
            track: Whether to actually log to W&B.
            project: W&B project name.
            entity: W&B entity (team/user).
            run_id: Optional run ID to resume.
            tags: Optional list of tags.
            name: Optional run name.
        """
        self._track = track
        self._project = project
        self._entity = entity
        self._run_id = run_id
        self._tags = tags
        self._name = name
        self._run: wandb_module.sdk.wandb_run.Run | None = None
        self._wandb: Any = None

    @property
    def tracking(self) -> bool:
        """Return whether tracking is active."""
        return self._track and self._run is not None

    @property
    def run(self) -> wandb_module.sdk.wandb_run.Run | None:
        """Return the current W&B run, if any."""
        return self._run

    @property
    def run_url(self) -> str | None:
        """Return the URL to the current W&B run."""
        if self._run is not None:
            return self._run.get_url()  # type: ignore[union-attr]
        return None

    def init(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize a W&B run.

        Args:
            config: Configuration dictionary to log.
            **kwargs: Additional arguments passed to wandb.init().
        """
        if not self._track:
            return

        require_wandb()
        import wandb

        self._wandb = wandb

        init_kwargs: dict[str, Any] = {
            "project": self._project,
            "config": config,
        }
        if self._entity:
            init_kwargs["entity"] = self._entity
        if self._run_id:
            init_kwargs["id"] = self._run_id
            init_kwargs["resume"] = "allow"
        if self._tags:
            init_kwargs["tags"] = self._tags
        if self._name:
            init_kwargs["name"] = self._name

        init_kwargs.update(kwargs)
        self._run = wandb.init(**init_kwargs)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B.

        Args:
            data: Dictionary of metrics to log.
            step: Optional step number.
        """
        if not self.tracking:
            return

        log_kwargs: dict[str, Any] = {}
        if step is not None:
            log_kwargs["step"] = step

        self._wandb.log(data, **log_kwargs)

    def log_image(self, key: str, image_path: str | Path, caption: str | None = None) -> None:
        """Log an image to W&B.

        Args:
            key: Key name for the image.
            image_path: Path to the image file.
            caption: Optional caption for the image.
        """
        if not self.tracking:
            return

        image = self._wandb.Image(str(image_path), caption=caption)
        self._wandb.log({key: image})

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table to W&B.

        Args:
            key: Key name for the table.
            columns: List of column names.
            data: List of rows, where each row is a list of values.
        """
        if not self.tracking:
            return

        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({key: table})

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        paths: list[str | Path],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact to W&B.

        Args:
            name: Name of the artifact.
            artifact_type: Type of artifact (e.g., "model", "dataset").
            paths: List of file paths to include.
            metadata: Optional metadata dictionary.
        """
        if not self.tracking:
            return

        artifact = self._wandb.Artifact(name, type=artifact_type, metadata=metadata)
        for path in paths:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)  # type: ignore[union-attr]

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.tracking:
            return

        self._wandb.finish()
        self._run = None
