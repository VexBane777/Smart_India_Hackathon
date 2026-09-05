"""
VAANI Models Module

Contains model implementations including TinyCNN and SSLHead for
audio-based deepfake detection.
"""

from .cnn import TinyCNN
from .ssl_head import SSLHead

__all__ = ["TinyCNN", "SSLHead"]
