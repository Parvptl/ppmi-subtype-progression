"""Readers for the PPMI source files.

Each loader returns a raw frame with missing values intact. Nothing here
imputes, filters on outcome, or derives anything the modelling code should own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


class MissingDataFileError(FileNotFoundError):
    """Raised when a required source file has not been downloaded."""


_DOWNLOAD_HINT = """
Download it from the PPMI portal:

    LONI IDA -> PPMI -> Download -> Study Data -> Motor Assessments

and place it in Data/06_Motor_UPDRS/. See docs/DATA_REQUIREMENTS.md for the
full list and the exact columns each file must contain.
""".rstrip()


def _resolve_single(cfg: Config, key: str, description: str) -> Path:
    matches = cfg.glob(key)
    if not matches:
        raise MissingDataFileError(
            f"{description} not found (config paths.{key} = "
            f"{cfg.raw['paths'][key]!r}).\n{_DOWNLOAD_HINT}"
        )
    if len(matches) > 1:
        # Deterministic: newest by name, which for PPMI encodes the cut date.
        return matches[-1]
    return matches[0]


def load_curated_cut(cfg: Config) -> pd.DataFrame:
    """Longitudinal curated data cut: one row per patient-visit."""
    path = cfg.path("curated_cut")
    if not path.exists():
        raise MissingDataFileError(f"curated data cut not found at {path}")
    return pd.read_excel(path, sheet_name=cfg.raw["paths"]["curated_cut_sheet"])


def load_curated_dictionary(cfg: Config) -> pd.DataFrame:
    path = cfg.path("curated_cut")
    return pd.read_excel(path, sheet_name=cfg.raw["paths"]["curated_cut_dictionary_sheet"])


def load_updrs_part2(cfg: Config) -> pd.DataFrame:
    """MDS-UPDRS Part II, patient questionnaire (module NUPDRS2P)."""
    path = _resolve_single(cfg, "updrs_part2_glob", "MDS-UPDRS Part II item-level file")
    return pd.read_csv(path, low_memory=False)


def load_updrs_part3(cfg: Config) -> pd.DataFrame:
    """MDS-UPDRS Part III, motor examination (module NUPDRS3TRT)."""
    path = _resolve_single(cfg, "updrs_part3_glob", "MDS-UPDRS Part III item-level file")
    return pd.read_csv(path, low_memory=False)


def load_archived_original_cohort(cfg: Config) -> pd.DataFrame:
    """Apr-2020 snapshot, the only in-repo source carrying a 3-class label."""
    path = cfg.path("archived_original_cohort")
    if not path.exists():
        raise MissingDataFileError(f"archived cohort file not found at {path}")
    return pd.read_csv(path, low_memory=False)


def updrs_files_available(cfg: Config) -> bool:
    """True when both item-level MDS-UPDRS files are on disk."""
    return bool(cfg.glob("updrs_part2_glob")) and bool(cfg.glob("updrs_part3_glob"))
