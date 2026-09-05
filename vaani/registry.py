"""
Model Registry and Configuration System

Provides a central registry for model classes and a configuration loader that
merges model-specific configs with shared base defaults.

Config schema deviation from the docs (intentional, not a bug):

    The config schema actually implemented here/in vaani/configs/*.yaml is
    the canonical one for this codebase. It intentionally differs from the
    schema shown in vaani/TDDs/TDD_MOD_B_01_Models.md and
    vaani/DOC2_TRAINING_CONFIGS.md in these ways:

      - `type` is a top-level config key (the docs nest it under `model.type`).
      - `audio.sr` / `audio.win_s` (docs) are `audio.sample_rate` /
        `audio.window_seconds` here.
      - `features.hop` (docs) is `features.hop_length` here.

    These renames/restructuring were judged clearer than the doc spec and
    were kept; the doc files themselves are left as-is (not updated) -- this
    note exists so a future reader comparing code to docs doesn't mistake
    the difference for an implementation bug.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml


# Global model registry: maps model type names to their class constructors
MODEL_REGISTRY: Dict[str, Type] = {}


def _ensure_models_imported() -> None:
    """
    Make sure model modules that register themselves via `@register_model`
    have actually been imported before we look anything up in
    MODEL_REGISTRY.

    Without this, `registry.load(...)` only works by accident: it depends on
    the caller having already imported `models`/`models.cnn` (an import-time
    side effect that populates MODEL_REGISTRY) before calling `load()`. From
    a clean process where nothing has imported `models` yet, `load()` would
    otherwise fail with "Model type '...' not found in MODEL_REGISTRY.
    Available types: []" even for a perfectly valid, registered model type.

    This is a lazy import (done here, not at module load time) specifically
    to avoid an import cycle: `models/cnn.py` imports `register_model` from
    this module, so `registry.py` cannot import `models` at the top of the
    file.
    """
    import models  # noqa: F401  (import side effect: registers models)


def register_model(name: str):
    """
    Decorator to register a model class in MODEL_REGISTRY.

    Usage:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """
    def decorator(cls: Type) -> Type:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def load_config(config_path: str, configs_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge a model configuration with base.yaml.

    Args:
        config_path: Either a bare config name (e.g., "cnn_week1") or a full/relative
                    path to a YAML file.
                    - Bare names (no directory separators, no extension) are resolved
                      relative to configs_dir (e.g., "cnn_week1" → configs_dir/cnn_week1.yaml)
                    - Absolute paths are used as-is.
                    - Relative paths are used as-is, relative to the current working directory.
        configs_dir: Directory containing config files. Defaults to 'configs/' relative
                    to this module.

    Returns:
        Merged configuration dict where specific config overrides base.yaml values.

    Raises:
        FileNotFoundError: If config file cannot be found.
        yaml.YAMLError: If YAML parsing fails.
    """
    if configs_dir is None:
        # Default to 'configs/' directory relative to this module
        module_dir = Path(__file__).parent
        configs_dir = str(module_dir / "configs")

    configs_dir = Path(configs_dir)

    # Determine the actual config path
    def is_bare_name(path: str) -> bool:
        """Check if path is a bare name (no directory separators, no file extension)."""
        has_separators = "/" in path or "\\" in path
        has_extension = path.endswith(".yaml") or path.endswith(".yml")
        return not has_separators and not has_extension

    if is_bare_name(config_path):
        # Bare name: resolve to configs_dir/name.yaml
        actual_config_path = configs_dir / f"{config_path}.yaml"
    else:
        # Full or relative path: use as-is (absolute or relative to current working directory)
        actual_config_path = Path(config_path)

    # Load base config
    base_config_path = configs_dir / "base.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f) or {}

    # Load model-specific config
    if not actual_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {actual_config_path}")

    with open(actual_config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

    # Merge: model config overrides base config, recursively. A shallow
    # {**base_config, **model_config} would silently drop every sibling key
    # in a nested block (e.g. base.yaml's features.n_fft/hop_length/...)
    # whenever the specific config overrides just one key in that same
    # block (e.g. features.n_mels) -- since base.yaml is entirely nested
    # blocks, that's not an acceptable merge semantics for this schema.
    merged_config = _deep_merge(base_config, model_config)

    return merged_config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`, returning a new dict.

    - For keys present in both where both values are dicts, merge recursively
      (so sibling keys at every nesting level are preserved from `base`).
    - Otherwise, `override`'s value wins (including replacing a non-dict
      base value with a dict, or vice versa).
    """
    merged: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load(config_path: str, configs_dir: Optional[str] = None) -> Any:
    """
    Load a model configuration and instantiate the corresponding model.

    Args:
        config_path: Either a bare config name (e.g., "cnn_week1") or a full/relative
                    path to a YAML file.
        configs_dir: Directory containing config files.

    Returns:
        An instance of the model class specified in the config's 'type' field.

    Raises:
        KeyError: If config does not specify a 'type' field.
        ValueError: If the specified model type is not registered.
        FileNotFoundError: If config file cannot be found.
    """
    config = load_config(config_path, configs_dir)

    if "type" not in config:
        raise KeyError(f"Config must specify a 'type' field naming the model registry key")

    model_type = config["type"]

    _ensure_models_imported()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' not found in MODEL_REGISTRY. "
            f"Available types: {sorted(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Instantiate the model with the config
    return model_class(config)
