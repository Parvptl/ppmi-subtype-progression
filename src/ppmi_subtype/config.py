"""Configuration loading and validation.

The config is read once and validated eagerly, so a typo in config.yaml fails
at startup rather than halfway through a long pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when config.yaml is missing a key or holds an invalid value."""


def project_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SubtypeConfig:
    """Parameters of the TD/PIGD classification."""

    tremor_items: tuple[str, ...]
    pigd_items: tuple[str, ...]
    td_threshold: float
    pigd_threshold: float
    labels: dict[str, int]
    missing_value_codes: tuple[float, ...]
    valid_score_range: tuple[float, float]
    max_missing_tremor_items: int
    max_missing_pigd_items: int
    medication_state: str
    pdstate_column: str
    treatment_column: str

    @property
    def all_items(self) -> tuple[str, ...]:
        return self.tremor_items + self.pigd_items

    @property
    def label_names(self) -> dict[int, str]:
        return {v: k for k, v in self.labels.items()}


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Resolve a `paths.<key>` entry against the project root."""
        paths = self.raw["paths"]
        if key not in paths:
            raise ConfigError(f"paths.{key} is not defined in config.yaml")
        return self.root / paths[key]

    def glob(self, key: str) -> list[Path]:
        """Resolve a `paths.<key>` glob pattern to sorted matching files."""
        paths = self.raw["paths"]
        if key not in paths:
            raise ConfigError(f"paths.{key} is not defined in config.yaml")
        return sorted(self.root.glob(paths[key]))

    @property
    def subtype(self) -> SubtypeConfig:
        s = self.raw["subtype"]
        return SubtypeConfig(
            tremor_items=tuple(s["tremor_items"]),
            pigd_items=tuple(s["pigd_items"]),
            td_threshold=float(s["td_threshold"]),
            pigd_threshold=float(s["pigd_threshold"]),
            labels=dict(s["labels"]),
            missing_value_codes=tuple(float(c) for c in s["missing_value_codes"]),
            valid_score_range=(float(s["valid_score_range"][0]), float(s["valid_score_range"][1])),
            max_missing_tremor_items=int(s["max_missing_tremor_items"]),
            max_missing_pigd_items=int(s["max_missing_pigd_items"]),
            medication_state=str(s["medication_state"]).upper(),
            pdstate_column=s["pdstate_column"],
            treatment_column=s["treatment_column"],
        )

    @property
    def cohort(self) -> dict[str, Any]:
        return self.raw["cohort"]

    @property
    def stability_predictors(self) -> dict[str, list[str]]:
        return self.raw["stability_predictors"]

    @property
    def splits(self) -> dict[str, Any]:
        return self.raw["splits"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]


_REQUIRED_SECTIONS = (
    "seed",
    "paths",
    "subtype",
    "cohort",
    "stability_predictors",
    "splits",
    "model",
)


def _validate(raw: dict[str, Any]) -> None:
    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(f"config.yaml is missing required section(s): {missing}")

    s = raw["subtype"]
    for key in (
        "tremor_items",
        "pigd_items",
        "td_threshold",
        "pigd_threshold",
        "labels",
        "missing_value_codes",
        "valid_score_range",
        "max_missing_tremor_items",
        "max_missing_pigd_items",
        "medication_state",
    ):
        if key not in s:
            raise ConfigError(f"config.yaml is missing subtype.{key}")

    lo, hi = s["valid_score_range"]
    if not float(lo) < float(hi):
        raise ConfigError(f"subtype.valid_score_range must be increasing, got {[lo, hi]}")

    inside = [c for c in s["missing_value_codes"] if float(lo) <= float(c) <= float(hi)]
    if inside:
        raise ConfigError(
            f"subtype.missing_value_codes {inside} fall inside valid_score_range "
            f"{[lo, hi]}; real scores would be discarded as missing"
        )

    if not s["tremor_items"] or not s["pigd_items"]:
        raise ConfigError("subtype.tremor_items and subtype.pigd_items must be non-empty")

    overlap = set(s["tremor_items"]) & set(s["pigd_items"])
    if overlap:
        raise ConfigError(f"items appear in both tremor and PIGD lists: {sorted(overlap)}")

    if not float(s["pigd_threshold"]) < float(s["td_threshold"]):
        raise ConfigError(
            "subtype.pigd_threshold must be strictly less than subtype.td_threshold "
            f"(got {s['pigd_threshold']} and {s['td_threshold']})"
        )

    if set(s["labels"]) != {"TD", "Indeterminate", "PIGD"}:
        raise ConfigError(
            f"subtype.labels must define exactly TD, Indeterminate and PIGD; got {sorted(s['labels'])}"
        )

    if len(set(s["labels"].values())) != 3:
        raise ConfigError("subtype.labels must map to three distinct codes")

    if s["medication_state"].upper() not in {"OFF", "ON", "ANY"}:
        raise ConfigError("subtype.medication_state must be one of OFF, ON, ANY")

    ratios = raw["splits"]
    total = float(ratios["train"]) + float(ratios["test"])
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(f"splits.train + splits.test must sum to 1.0, got {total}")

    m = raw["model"]
    for key in ("horizons", "emission_floor", "n_restarts", "bootstrap_draws"):
        if key not in m:
            raise ConfigError(f"config.yaml is missing model.{key}")

    if not 0.0 < float(m["emission_floor"]) < 1.0:
        raise ConfigError(
            f"model.emission_floor must lie in (0, 1); got {m['emission_floor']}"
        )
    if not m["horizons"] or any(float(h) <= 0 for h in m["horizons"]):
        raise ConfigError("model.horizons must be a non-empty list of positive years")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate config.yaml."""
    root = project_root()
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    _validate(raw)
    return Config(raw=raw, root=root)
