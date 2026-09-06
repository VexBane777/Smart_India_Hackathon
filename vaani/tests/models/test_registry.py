"""
Unit tests for registry.py - Model Registry and Configuration System.

Tests cover:
1. Config loading from YAML files
2. Config merging (specific config overrides base.yaml)
3. Registry mechanism (registration and lookup)
4. Error handling (missing files, invalid configs)
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from registry import MODEL_REGISTRY, load, load_config, register_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_configs_dir():
    """Create a temporary configs directory with base and test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create base.yaml
        base_config = {
            "seed": 42,
            "audio": {
                "sample_rate": 16000,
                "window_seconds": 2.0,
                "hop_seconds": 0.5,
            },
            "features": {
                "n_mels": 64,
                "n_fft": 512,
                "hop_length": 160,
                "fmin": 20,
                "fmax": 8000,
            },
            "data": {
                "manifest": "data/manifests/train.parquet",
                "val": "data/manifests/val.parquet",
            },
        }

        import yaml

        with open(config_dir / "base.yaml", "w") as f:
            yaml.dump(base_config, f)

        # Create a minimal model config (test_model.yaml)
        model_config = {
            "type": "test_model",
            "name": "test_model_v1",
            "model": {
                "hidden_dim": 128,
            },
            "optim": {
                "lr": 1e-4,
            },
        }

        with open(config_dir / "test_model.yaml", "w") as f:
            yaml.dump(model_config, f)

        yield config_dir


@pytest.fixture
def mock_model_class():
    """Create a mock model class for testing."""

    @register_model("test_model")
    class MockModel(nn.Module):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config
            self.hidden_dim = config.get("model", {}).get("hidden_dim", 64)
            self.linear = nn.Linear(self.hidden_dim, 2)

        def forward(self, x):
            return self.linear(x)

    return MockModel


# ---------------------------------------------------------------------------
# Tests: Config Loading and Merging
# ---------------------------------------------------------------------------


def test_load_config_with_bare_name(temp_configs_dir):
    """Test loading a config using bare name (e.g., 'test_model')."""
    config = load_config("test_model", str(temp_configs_dir))

    assert config["type"] == "test_model"
    assert config["name"] == "test_model_v1"
    assert config["seed"] == 42  # From base.yaml
    assert config["audio"]["sample_rate"] == 16000  # From base.yaml


def test_load_config_with_full_path(temp_configs_dir):
    """Test loading a config using full path."""
    config_path = temp_configs_dir / "test_model.yaml"
    config = load_config(str(config_path), str(temp_configs_dir))

    assert config["type"] == "test_model"
    assert config["name"] == "test_model_v1"


def test_load_config_no_double_join_with_directory_path(temp_configs_dir):
    """Test that paths with directory components are not double-joined with configs_dir.

    This verifies the fix for the critical double-join bug: if someone passes a
    path like "configs/model.yaml" (relative path with directory component) along
    with configs_dir="vaani/configs", the old code would incorrectly produce
    "vaani/configs/configs/model.yaml". This test ensures that behavior is fixed.

    The fix: paths with directory separators are NOT joined with configs_dir;
    only bare names (no separators, no extension) are.
    """
    import yaml

    # Create a subdirectory within temp_configs_dir
    subdir = temp_configs_dir / "submodels"
    subdir.mkdir()

    # Create a config in the subdirectory
    subconfig = {
        "type": "sub_model",
        "name": "sub_model_v1",
    }
    with open(subdir / "sub_model.yaml", "w") as f:
        yaml.dump(subconfig, f)

    # Load using a path that includes the directory: "submodels/sub_model.yaml"
    # With the bug, this would try to load from:
    #   temp_configs_dir / "submodels/sub_model.yaml" / ... (double-join)
    # With the fix, it should just load from:
    #   subdir / "sub_model.yaml" (current working directory is not changed)
    # Since we're using an absolute path, this should work:
    config = load_config(str(subdir / "sub_model.yaml"), str(temp_configs_dir))

    assert config["type"] == "sub_model"
    assert config["name"] == "sub_model_v1"


def test_config_merge_specific_overrides_base(temp_configs_dir):
    """Test that specific config values override base config values."""
    import yaml

    # Create a config that overrides seed
    override_config = {
        "type": "override_test",
        "seed": 99,  # Override base seed
        "name": "override_v1",
        "features": {
            "n_mels": 128,  # Override n_mels from base
        },
    }

    with open(temp_configs_dir / "override_test.yaml", "w") as f:
        yaml.dump(override_config, f)

    config = load_config("override_test", str(temp_configs_dir))

    # Check overrides
    assert config["seed"] == 99  # Should be overridden
    assert config["features"]["n_mels"] == 128  # Should be overridden

    # Check that base values are still present where not overridden
    assert config["audio"]["sample_rate"] == 16000  # From base, not overridden
    assert config["data"]["manifest"] == "data/manifests/train.parquet"  # From base


def test_config_merge_preserves_base_audio_spec(temp_configs_dir):
    """Test that base audio spec is preserved when not overridden."""
    config = load_config("test_model", str(temp_configs_dir))

    # All audio specs from base should be present
    assert config["audio"]["sample_rate"] == 16000
    assert config["audio"]["window_seconds"] == 2.0
    assert config["audio"]["hop_seconds"] == 0.5


def test_config_merge_preserves_base_features(temp_configs_dir):
    """Test that base feature spec is preserved."""
    config = load_config("test_model", str(temp_configs_dir))

    # All feature specs from base should be present
    assert config["features"]["n_mels"] == 64
    assert config["features"]["n_fft"] == 512
    assert config["features"]["hop_length"] == 160
    assert config["features"]["fmin"] == 20
    assert config["features"]["fmax"] == 8000


# ---------------------------------------------------------------------------
# Tests: Config File Error Handling
# ---------------------------------------------------------------------------


def test_missing_base_config_raises_error(temp_configs_dir):
    """Test that missing base.yaml raises FileNotFoundError."""
    # Remove base.yaml
    (temp_configs_dir / "base.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="Base config not found"):
        load_config("test_model", str(temp_configs_dir))


def test_missing_model_config_raises_error(temp_configs_dir):
    """Test that missing model config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("nonexistent_model", str(temp_configs_dir))


# ---------------------------------------------------------------------------
# Tests: Registry Mechanism
# ---------------------------------------------------------------------------


def test_register_model_decorator(mock_model_class):
    """Test that @register_model decorator properly registers a model."""
    assert "test_model" in MODEL_REGISTRY
    assert MODEL_REGISTRY["test_model"] is mock_model_class


def test_load_instantiates_registered_model(temp_configs_dir, mock_model_class):
    """Test that load() instantiates the correct model class."""
    model = load("test_model", str(temp_configs_dir))

    assert isinstance(model, mock_model_class)
    assert model.config["type"] == "test_model"
    assert model.hidden_dim == 128


def test_load_missing_type_key_raises_error(temp_configs_dir):
    """Test that load() raises KeyError if config has no 'type' key."""
    import yaml

    # Create a config without type key
    no_type_config = {
        "name": "no_type_v1",
        "model": {"hidden_dim": 64},
    }

    with open(temp_configs_dir / "no_type.yaml", "w") as f:
        yaml.dump(no_type_config, f)

    with pytest.raises(KeyError, match="Config must specify a 'type' field"):
        load("no_type", str(temp_configs_dir))


def test_load_unregistered_model_type_raises_error(temp_configs_dir, mock_model_class):
    """Test that load() raises ValueError for unregistered model types."""
    import yaml

    # Create a config with an unregistered type
    unregistered_config = {
        "type": "nonexistent_model_type",
        "name": "unregistered_v1",
    }

    with open(temp_configs_dir / "unregistered.yaml", "w") as f:
        yaml.dump(unregistered_config, f)

    with pytest.raises(ValueError, match="Model type 'nonexistent_model_type' not found"):
        load("unregistered", str(temp_configs_dir))


# ---------------------------------------------------------------------------
# Tests: Default Configs Directory
# ---------------------------------------------------------------------------


def test_load_config_uses_default_configs_dir():
    """Test that load_config uses the default 'configs/' directory.

    This test verifies that the function correctly resolves to vaani/configs/
    by loading one of the real config files created during the task.
    """
    config = load_config("cnn_week1")
    assert config["type"] == "tinycnn"
    assert config["name"] == "cnn_week1"
    assert config["seed"] == 42  # From base.yaml


def test_config_merge_is_deep_preserves_sibling_keys(temp_configs_dir):
    """Test that overriding one key inside a nested block preserves its siblings.

    Regression test for the shallow-merge bug: a naive
    {**base_config, **model_config} merge replaces the ENTIRE top-level
    "features" dict when the specific config overrides just one key inside
    it (e.g. n_mels), silently dropping base.yaml's other features.* keys
    (n_fft, hop_length, fmin, fmax). The merge must instead be recursive so
    those sibling keys survive.
    """
    import yaml

    override_config = {
        "type": "deep_merge_test",
        "name": "deep_merge_v1",
        "features": {
            "n_mels": 999,  # Override just this one nested key
        },
    }

    with open(temp_configs_dir / "deep_merge_test.yaml", "w") as f:
        yaml.dump(override_config, f)

    config = load_config("deep_merge_test", str(temp_configs_dir))

    # The overridden leaf value should win.
    assert config["features"]["n_mels"] == 999

    # Every sibling key in the same nested block must survive from base.yaml,
    # unchanged -- a shallow merge would drop all of these.
    assert config["features"]["n_fft"] == 512
    assert config["features"]["hop_length"] == 160
    assert config["features"]["fmin"] == 20
    assert config["features"]["fmax"] == 8000


def test_registry_load_from_clean_subprocess():
    """Regression test: registry.load() must work from a clean process.

    Prior bug: registry.py never imported the `models` package, so
    `@register_model` (an import-time side effect in models/cnn.py) never
    fired unless the caller had already imported `models`/`models.cnn`
    first. Every existing test happened to pass only because the test
    files themselves import `models`/`models.cnn` before calling
    `registry.load()` -- a same-process test cannot catch this class of
    bug, since importing `registry` in the same interpreter as the rest of
    this test module leaves MODEL_REGISTRY already populated by earlier
    imports. Running in a genuinely separate subprocess is required.
    """
    vaani_dir = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); import registry; "
            "m = registry.load('cnn_week1'); "
            "print(type(m).__name__)",
        ],
        cwd=str(vaani_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"registry.load('cnn_week1') failed in a clean subprocess.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "TinyCNN" in result.stdout


# ---------------------------------------------------------------------------
# Tests: ${name} interpolation
# ---------------------------------------------------------------------------


def test_load_config_interpolates_name_placeholder(temp_configs_dir):
    """`${name}` in string config values (e.g. ckpt.dir / ckpt.hub_repo) is
    substituted with the config's own top-level `name` field.

    Regression test for the documented gap in base.yaml/train.py's
    docstrings: no template-resolution step existed anywhere, so
    `ckpt.dir: runs/${name}` was returned to callers still containing the
    literal, unresolved `${name}` placeholder.
    """
    import yaml

    with open(temp_configs_dir / "base.yaml") as f:
        base_config = yaml.safe_load(f)
    base_config["ckpt"] = {
        "dir": "runs/${name}",
        "hub_repo": "vaani/models/${name}",
        "every_steps": 500,
        "keep_last": 3,
    }
    with open(temp_configs_dir / "base.yaml", "w") as f:
        yaml.dump(base_config, f)

    config = load_config("test_model", str(temp_configs_dir))

    assert config["ckpt"]["dir"] == "runs/test_model_v1"
    assert config["ckpt"]["hub_repo"] == "vaani/models/test_model_v1"
    # every_steps/keep_last are untouched (not strings, nothing to interpolate)
    assert config["ckpt"]["every_steps"] == 500


def test_load_config_without_name_leaves_placeholder_unresolved(temp_configs_dir):
    """If a config has no top-level `name`, `${name}` is left as-is rather
    than crashing load_config() -- there is nothing to substitute it with."""
    import yaml

    with open(temp_configs_dir / "base.yaml") as f:
        base_config = yaml.safe_load(f)
    base_config["ckpt"] = {"dir": "runs/${name}"}
    with open(temp_configs_dir / "base.yaml", "w") as f:
        yaml.dump(base_config, f)

    no_name_config = {"type": "test_model"}
    with open(temp_configs_dir / "no_name.yaml", "w") as f:
        yaml.dump(no_name_config, f)

    config = load_config("no_name", str(temp_configs_dir))
    assert config["ckpt"]["dir"] == "runs/${name}"


def test_load_config_uses_real_cnn_week1_name_for_interpolation():
    """End-to-end: the real cnn_week1.yaml + base.yaml resolve ckpt.dir to
    'runs/cnn_week1', matching DOC2's documented `ckpt.dir: runs/${name}`
    convention."""
    config = load_config("cnn_week1")
    assert config["ckpt"]["dir"] == "runs/cnn_week1"
    assert config["ckpt"]["hub_repo"] == "vaani/models/cnn_week1"


def test_load_config_bare_name_vs_full_path(temp_configs_dir):
    """Test that both bare names and full paths work identically."""
    config_bare = load_config("test_model", str(temp_configs_dir))
    config_path = load_config(
        str(temp_configs_dir / "test_model.yaml"), str(temp_configs_dir)
    )

    # Both should produce identical configs
    assert config_bare == config_path
