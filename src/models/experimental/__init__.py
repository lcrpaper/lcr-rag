"""
Experimental Model Implementations

**Status**: EXPERIMENTAL / IN DEVELOPMENT
**Warning**: These implementations are NOT validated for paper reproduction.

This module contains active research branches that are being explored but have
not yet been validated or incorporated into the main paper results.

Contents:
---------
- adaptive_refinement.py: Learning alpha dynamically per-example
- multi_scale_detector.py: Hierarchical conflict detection
- contrastive_classifier.py: Alternative training objective for taxonomy

Usage Warning:
--------------
These models are provided for research exploration only. They may:
- Produce different results than reported in the paper
- Have bugs or incomplete implementations
- Be removed or significantly changed in future versions
- Require additional dependencies not in requirements.txt

For paper reproduction, use models from src/models/ (parent directory).

Example:
--------
>>> # EXPERIMENTAL - use at your own risk
>>> from src.models.experimental import AdaptiveRefinementModule
>>>
>>> # For paper results, use this instead:
>>> from src.models.refinement_module import RefinementModule
"""

import warnings

warnings.warn(
    "Importing from src.models.experimental - these are EXPERIMENTAL models "
    "not validated for paper reproduction. Use src.models for official results.",
    UserWarning,
    stacklevel=2
)

def __getattr__(name):
    if name == "AdaptiveRefinementModule":
        from .adaptive_refinement import AdaptiveRefinementModule
        return AdaptiveRefinementModule
    elif name == "MultiScaleDetector":
        from .multi_scale_detector import MultiScaleDetector
        return MultiScaleDetector
    elif name == "ContrastiveClassifier":
        from .contrastive_classifier import ContrastiveClassifier
        return ContrastiveClassifier
    elif name == "GatedRefinementModule":
        from .gated_refinement import GatedRefinementModule
        return GatedRefinementModule
    elif name == "TransformerRefinementModule":
        from .transformer_refinement import TransformerRefinementModule
        return TransformerRefinementModule
    raise AttributeError(f"module 'src.models.experimental' has no attribute '{name}'")

__all__ = [
    "AdaptiveRefinementModule",
    "MultiScaleDetector",
    "ContrastiveClassifier",
    "GatedRefinementModule",
    "TransformerRefinementModule",
]
