"""Tests for core engine."""

import pytest
from neuralog import Engine
from neuralog.core.config import Config


def test_engine_initialization():
    """Test that engine initializes correctly."""
    config = Config()
    engine = Engine(config)

    assert engine is not None
    assert engine.config is not None


def test_engine_config():
    """Test engine configuration."""
    config = Config()
    engine = Engine(config)

    stats = engine.get_statistics()
    assert "config" in stats
    assert "components" in stats
