"""Probabilistic motor subtype progression modelling for Parkinson's disease.

A hierarchical Bayesian extension of the clinical TD/PIGD classification,
built on the PPMI longitudinal dataset.
"""

from .config import Config, ConfigError, load_config

__all__ = ["Config", "ConfigError", "load_config"]
__version__ = "0.1.0"
