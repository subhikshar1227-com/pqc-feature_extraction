"""
Preprocessing module for Phase 2 feature detection.

Provides reusable preprocessing components extracted from the proven 
Phase 1 preprocessing algorithms.
"""

from .product_isolation import ProductIsolationPreprocessor

__all__ = ['ProductIsolationPreprocessor']