"""
LCR Models Package

Provides the core model components for Latent Conflict Refinement:
- ConflictDetector: 2M parameter binary classifier
- RefinementModule: 6M parameter iterative refinement
- LCRSystem: Combined system for full pipeline
"""

from .conflict_detector import ConflictDetector, create_detector
from .refinement_module import RefinementModule, create_refinement

__all__ = [
    'ConflictDetector',
    'RefinementModule',
    'create_detector',
    'create_refinement',
]
