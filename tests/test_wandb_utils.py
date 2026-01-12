"""Tests for W&B utilities."""

from unittest.mock import MagicMock, patch

import pytest

from rl_drug_dosage.utils.wandb_utils import WandbLogger, is_wandb_available, require_wandb


def test_is_wandb_available_returns_bool():
    """Test that is_wandb_available returns a boolean."""
    result = is_wandb_available()
    assert isinstance(result, bool)


def test_require_wandb_raises_when_unavailable():
    """Test that require_wandb raises ImportError when wandb is not installed."""
    with patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=False):
        with pytest.raises(ImportError, match="wandb is not installed"):
            require_wandb()


def test_require_wandb_does_not_raise_when_available():
    """Test that require_wandb does not raise when wandb is installed."""
    with patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=True):
        require_wandb()  # Should not raise


def test_wandb_logger_is_noop_when_track_false():
    """Test that WandbLogger is a no-op when track=False."""
    logger = WandbLogger(track=False)

    # These should all be no-ops and not raise
    logger.init(config={"test": 1})
    logger.log({"metric": 0.5})
    logger.log_image("test", "/path/to/image.png")
    logger.log_table("test", ["col1"], [[1]])
    logger.log_artifact("test", "model", ["/path/to/file"])
    logger.finish()

    assert not logger.tracking
    assert logger.run is None
    assert logger.run_url is None


def test_wandb_logger_init_calls_wandb_init():
    """Test that WandbLogger.init calls wandb.init when tracking."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    with (
        patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=True),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        logger = WandbLogger(
            track=True,
            project="test-project",
            entity="test-entity",
            tags=["tag1", "tag2"],
            name="test-run",
        )
        logger.init(config={"lr": 0.001})

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            config={"lr": 0.001},
            entity="test-entity",
            tags=["tag1", "tag2"],
            name="test-run",
        )


def test_wandb_logger_log_calls_wandb_log():
    """Test that WandbLogger.log calls wandb.log when tracking."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    with (
        patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=True),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        logger = WandbLogger(track=True)
        logger.init()

        logger.log({"loss": 0.5}, step=10)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=10)


def test_wandb_logger_tracking_property():
    """Test the tracking property reflects actual tracking state."""
    # Not tracking when track=False
    logger = WandbLogger(track=False)
    assert not logger.tracking

    # Not tracking when track=True but not initialized
    logger2 = WandbLogger(track=True)
    assert not logger2.tracking


def test_wandb_logger_with_run_id_resume():
    """Test that WandbLogger passes run_id and resume correctly."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    with (
        patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=True),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        logger = WandbLogger(
            track=True,
            project="test-project",
            run_id="abc123",
        )
        logger.init()

        mock_wandb.init.assert_called_once_with(
            project="test-project",
            config=None,
            id="abc123",
            resume="allow",
        )


def test_wandb_logger_finish_resets_run():
    """Test that finish() resets the run to None."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    with (
        patch("rl_drug_dosage.utils.wandb_utils.is_wandb_available", return_value=True),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        logger = WandbLogger(track=True)
        logger.init()

        assert logger.run is mock_run

        logger.finish()

        mock_wandb.finish.assert_called_once()
        assert logger.run is None
