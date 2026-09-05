"""
TinyCNN - A lightweight CNN for audio-based deepfake detection.

Architecture: 4x [Conv2d → BN → GELU → MaxPool] followed by
AdaptiveAvgPool and a Linear head.
"""

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from registry import register_model


@register_model("tinycnn")
class TinyCNN(nn.Module):
    """
    Tiny CNN model for binary audio classification.

    Accepts input audio spectrograms of shape (Batch, 1, n_mels, time_frames)
    and produces logits of shape (Batch, 2).

    `input_kind = "mel"` (class attribute) declares this model's input
    contract explicitly -- as opposed to `SSLHead`'s `"waveform"` -- since
    the two registry-loadable models are interchangeable only at
    construction time, not at call time: a caller passing raw waveforms to
    `TinyCNN`, or mel spectrograms to `SSLHead`, will fail at the first
    forward pass. See `state.md`'s known-gaps list for the corresponding
    `train.py` gap.

    Args:
        config: Either a config dict (from registry.load()) or None.
                When config is a dict, n_mels, channels, and dropout are
                extracted from it.
                When config is None, they default to the provided keyword arguments.
        n_mels: Number of mel-frequency bins (default: 64). Only used if config is None.
        chs: Tuple of channel dimensions for each block (default: (32, 64, 128, 256)).
             Only used if config is None.
        dropout: Dropout probability applied in the classification head, before
                the final Linear layer (default: 0.0, i.e. no dropout). Only
                used if config is None.
    """

    input_kind: str = "mel"

    def __init__(
        self,
        config: Union[Dict[str, Any], None] = None,
        n_mels: int = 64,
        chs: Tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        super().__init__()

        # Handle config-based initialization (from registry.load())
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("config must be a dict or None")
            # Extract n_mels from config.features.n_mels
            n_mels = config.get("features", {}).get("n_mels", n_mels)
            # Extract channels from config.model.channels
            channels_list = config.get("model", {}).get("channels", chs)
            chs = tuple(channels_list)
            # Extract dropout probability from config.model.dropout
            dropout = config.get("model", {}).get("dropout", dropout)

        self.n_mels = n_mels
        self.chs = chs
        self.dropout = dropout
        in_channels = 1  # Spectrogram has 1 channel

        # Build 4 convolutional blocks: Conv2d → BN → GELU → MaxPool
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for out_channels in chs:
            block = nn.Sequential(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.blocks.append(block)
            current_channels = out_channels

        # Adaptive average pool to (out_channels, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head: Dropout (regularization) -> Linear to 2 logits
        # (binary classification). Dropout is a no-op at p=0.0 (the default
        # when neither config nor a keyword argument supplies one).
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(chs[-1], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x: Input tensor of shape (Batch, 1, n_mels, time_frames)

        Returns:
            Logits tensor of shape (Batch, 2)
        """
        # Pass through all convolutional blocks
        for block in self.blocks:
            x = block(x)

        # Adaptive average pooling to (Batch, channels, 1, 1)
        x = self.avg_pool(x)

        # Flatten to (Batch, channels)
        x = x.view(x.size(0), -1)

        # Linear head to (Batch, 2)
        logits = self.head(x)

        return logits
