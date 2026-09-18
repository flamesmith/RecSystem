"""Reusable components for full-catalog SigLIP2 training."""

from .schema import ProductInput, TransformResult
from .text import DescriptionProcessor

__all__ = ["DescriptionProcessor", "ProductInput", "TransformResult"]
__version__ = "0.1.0"
