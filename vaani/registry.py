"""
Model Registry and Configuration System

Provides a central registry for model classes and a configuration loader that
merges model-specific configs with shared base defaults.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml


# Global model registry: maps model type names to their class constructors
MODEL_REGISTRY: Dict[str, Type] = {}


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

    # Merge: model config overrides base config
    merged_config = {**base_config, **model_config}

    return merged_config


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

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Model type '{model_type}' not found in MODEL_REGISTRY. "
            f"Available types: {sorted(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]

    # Instantiate the model with the config
    return model_class(config)
