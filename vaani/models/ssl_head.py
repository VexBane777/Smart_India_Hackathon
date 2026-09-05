"""
SSLHead - A self-supervised-learning (SSL) teacher model for audio-based
deepfake detection, built on top of a pretrained wav2vec2 checkpoint.

Architecture: a frozen `Wav2Vec2Model` feature extractor produces one
hidden-state tensor per transformer layer (plus the input embedding layer).
These are combined via a learnable, softmax-normalized weighted sum
(SUPERB-style layer weighting), then passed through a 2-layer MLP
classification head (768 -> 256 -> 2).

Known config-vs-code deviation (documented here per the pattern
`registry.py`'s module docstring uses for other schema divergences):
`configs/ssl_teacher.yaml`'s `freeze: [feature_extractor]` uses the HF
idiom for that name -- the conv front-end only (`Wav2Vec2Model.
feature_extractor`) -- and its `optim` block (`bfloat16`,
`grad_checkpointing`, `accum`, `lr: 1.0e-5`) reads as a full-backbone
fine-tune. Neither is consumed by `SSLHead`: this class unconditionally
freezes the *entire* backbone (all of `Wav2Vec2Model`, not just its conv
front-end) via `requires_grad = False` plus a hard `torch.no_grad()` in
`forward()`, making it a linear probe over frozen SSL features (~0.2% of
total params trainable: `w` plus the MLP head). No config value can
unfreeze any of the backbone without editing this class. This was an
explicit choice for Task 3's brief ("freeze the SSL feature extractor")
and is deliberately left as-is here -- loosening the freeze scope to match
the config's apparent fine-tune intent is a design decision for whoever
implements real teacher training (see `state.md`'s known-gaps list), not a
bug to silently fix.
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import register_model


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

    `input_kind = "waveform"` (class attribute) declares this model's input
    contract explicitly -- as opposed to `TinyCNN`'s `"mel"` -- since the two
    registry-loadable models are interchangeable only at construction time,
    not at call time: a caller passing raw waveforms to `TinyCNN`, or mel
    spectrograms to `SSLHead`, will fail at the first forward pass. See
    `state.md`'s known-gaps list for the corresponding `train.py` gap.

    Args:
        config: Either a config dict (from registry.load()) or None.
                When config is a dict, `base` is extracted from
                config["model"]["base"] (falling back to a top-level
                "base" key, then to the `base` keyword argument).
        base: HuggingFace Hub identifier (or local path) of the pretrained
              wav2vec2 checkpoint to load (default: "facebook/wav2vec2-base").
              Only used if config is None or does not specify a base.
    """

    input_kind: str = "waveform"

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

        # `transformers` is imported here, not at module scope, so that
        # `import models` (which every `registry.load()` call triggers via
        # `_ensure_models_imported()`) stays cheap and does not force a
        # `transformers` dependency onto callers who only ever use
        # TinyCNN. There is no dependency manifest in this repo, so an
        # eager module-level import would silently turn every registry
        # lookup -- SSL or not -- into a hard `transformers` requirement.
        try:
            from transformers import Wav2Vec2Model
        except ImportError as exc:
            raise ImportError(
                "transformers is required for SSLHead: pip install transformers"
            ) from exc

        # Frozen SSL feature extractor.
        self.feature_extractor = Wav2Vec2Model.from_pretrained(base)
        hidden_size = self.feature_extractor.config.hidden_size

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        # Learnable layer-weighting vector: one raw (unnormalized) weight per
        # hidden-state tensor returned by the feature extractor. Softmax is
        # applied at forward time to normalize it into a convex combination.
        #
        # Sized dynamically from the loaded checkpoint's actual transformer
        # depth (`num_hidden_layers + 1`), not a hardcoded constant: `base`
        # is a config knob (`configs/ssl_teacher.yaml`'s `model.base`), and a
        # non-12-layer checkpoint (e.g. wav2vec2-large, or any custom
        # config) must not construct successfully only to crash with a
        # shape mismatch at the first forward pass. `output_hidden_states=
        # True` on `Wav2Vec2Model` returns one hidden-state tensor per
        # transformer layer *plus* the input embedding output, hence `+ 1`
        # -- confirmed empirically for wav2vec2-base (12 layers -> 13
        # hidden states) at implementation time.
        num_hidden_states = self.feature_extractor.config.num_hidden_layers + 1
        self.w = nn.Parameter(torch.zeros(num_hidden_states))

        # 2-layer MLP classification head: 768 -> 256 -> 2.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

    def train(self, mode: bool = True):
        """
        Override `nn.Module.train()` so switching this model into training
        mode (as `train.py` does at the top of every epoch) does not also
        switch the frozen `feature_extractor` into training mode.

        `nn.Module.train()` recurses into every child module by default. Since
        `feature_extractor.eval()` is only set once in `__init__`, a later
        `model.train()` call would otherwise silently re-enable wav2vec2's
        internal dropout (hidden/attention/feat-proj) and SpecAugment
        time-masking (`apply_spec_augment=True`). `requires_grad=False` plus
        `forward()`'s `no_grad()` block still keep gradients frozen either
        way, but the *values* the extractor emits would become
        nondeterministic and randomly masked -- wrong for a teacher whose
        job is a stable representation. Keeping `feature_extractor` pinned
        to eval mode here, regardless of `mode`, avoids that.
        """
        super().train(mode)
        self.feature_extractor.eval()
        return self

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
