"""
SSLHead - A self-supervised-learning (SSL) teacher model for audio-based
deepfake detection, built on top of a pretrained wav2vec2 checkpoint.

Architecture: a frozen `Wav2Vec2Model` feature extractor produces one
hidden-state tensor per transformer layer (plus the input embedding layer).
These are combined via a learnable, softmax-normalized weighted sum
(SUPERB-style layer weighting), then passed through a 2-layer MLP
classification head (768 -> 256 -> 2).
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from registry import register_model

# wav2vec2-base has 12 transformer layers. `output_hidden_states=True` on
# `Wav2Vec2Model` returns 13 hidden-state tensors: the input embedding
# output plus the output of each of the 12 transformer layers. The
# learnable layer-weighting vector `w` is sized to 13 (one weight per
# returned hidden-state tensor, including the embedding layer) rather than
# 12, so every hidden state the checkpoint actually returns participates in
# the weighted sum -- matching standard SUPERB-style layer weighting, which
# weights the full hidden_states stack. This was confirmed empirically at
# implementation time: `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
# (x, output_hidden_states=True).hidden_states` has length 13 for the real
# checkpoint.
NUM_HIDDEN_STATES = 13


@register_model("ssl_head")
class SSLHead(nn.Module):
    """
    SSL teacher model for binary audio classification.

    Wraps a pretrained `Wav2Vec2Model` as a frozen feature extractor, learns
    a softmax-normalized weighted sum over its hidden states (one weight per
    hidden-state layer), and classifies the resulting 768-dim mixed
    representation with a 2-layer MLP (768 -> 256 -> 2).

    Accepts raw audio waveforms of shape (Batch, num_samples) and produces
    logits of shape (Batch, 2).

    Args:
        config: Either a config dict (from registry.load()) or None.
                When config is a dict, `base` is extracted from
                config["model"]["base"] (falling back to a top-level
                "base" key, then to the `base` keyword argument).
        base: HuggingFace Hub identifier (or local path) of the pretrained
              wav2vec2 checkpoint to load (default: "facebook/wav2vec2-base").
              Only used if config is None or does not specify a base.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], None] = None,
        base: str = "facebook/wav2vec2-base",
    ):
        super().__init__()

        # Handle config-based initialization (from registry.load()), mirroring
        # TinyCNN's config-dict-or-kwargs resolution pattern in models/cnn.py.
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("config must be a dict or None")
            # Extract base checkpoint id from config.model.base, falling back
            # to a top-level "base" key, then to the constructor default.
            base = config.get("model", {}).get("base", config.get("base", base))

        self.base_name = base

        # Frozen SSL feature extractor.
        self.feature_extractor = Wav2Vec2Model.from_pretrained(base)
        hidden_size = self.feature_extractor.config.hidden_size

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # Learnable layer-weighting vector: one raw (unnormalized) weight per
        # hidden-state tensor returned by the feature extractor. Softmax is
        # applied at forward time to normalize it into a convex combination.
        self.w = nn.Parameter(torch.zeros(NUM_HIDDEN_STATES))

        # 2-layer MLP classification head: 768 -> 256 -> 2.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSL teacher.

        Args:
            x: Raw audio waveform tensor of shape (Batch, num_samples).

        Returns:
            Logits tensor of shape (Batch, 2).
        """
        # The feature extractor is frozen: never track gradients through it.
        with torch.no_grad():
            outputs = self.feature_extractor(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T, hidden_size)

        # Stack to (num_layers, B, T, hidden_size) and combine with a
        # softmax-normalized weighted sum over the layer dimension.
        stacked = torch.stack(hidden_states, dim=0)
        weights = F.softmax(self.w, dim=0).view(-1, 1, 1, 1)
        mixed = (stacked * weights).sum(dim=0)  # (B, T, hidden_size)

        # Mean-pool over time to get a fixed-size per-utterance representation.
        pooled = mixed.mean(dim=1)  # (B, hidden_size)

        logits = self.classifier(pooled)  # (B, 2)
        return logits
