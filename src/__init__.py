"""
LCR: Latent Conflict Refinement for Efficient RAG Verification

A lightweight 8M-parameter module for resolving conflicts in RAG systems.
Achieves self-consistency performance on shallow conflicts at 2% of the token cost.

Main components:
- models: Conflict detector, refinement module, taxonomy classifier
- training: Loss functions and training scripts
- data: Dataset loaders and utilities
- evaluation: Metrics and evaluation pipelines
"""

__version__ = "1.0.0"
__author__ = "Anonymous"
__paper__ = "Test-Time Latent Refinement for Efficient RAG Verification"

from src.models.conflict_detector import ConflictDetector
from src.models.refinement_module import RefinementModule
from src.models.taxonomy_classifier import TaxonomyClassifier
from src.models.lcr_system import LCRSystem

__all__ = [
    'ConflictDetector',
    'RefinementModule', 
    'TaxonomyClassifier',
    'LCRSystem',
]
