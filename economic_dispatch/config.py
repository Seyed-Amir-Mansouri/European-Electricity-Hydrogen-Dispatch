"""Run configuration and tunable assumptions.

Everything a user might reasonably want to change lives here so the model code
stays free of magic numbers. Values flagged "ASSUMPTION" are documented in the
README and are the ones to revisit if results look off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ALL_ZONES = [
    "AT00", "BE00", "BEOF", "CZ00", "DE00", "DEKF", "FR00", "FR15", "HR00",
    "HU00", "LUB1", "LUF1", "LUG1", "LUV1", "NL00", "NLLL", "PL00",
    "RO00", "SI00", "SK00",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "XLSXs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_ZONES_DB = DEFAULT_EXPORTS_DIR / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = DEFAULT_EXPORTS_DIR / "networks_2030.parquet"
DEFAULT_H2_REF = DEFAULT_EXPORTS_DIR / "ReferenceGrid_Hydrogen.xlsx"
DEFAULT_PLEXOS_REF = DEFAULT_DATA_DIR / "MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx"
DEFAULT_MARGINAL_PRICE_ELEC_DB = DEFAULT_EXPORTS_DIR / "marginal_price_electricity_2030.parquet"
DEFAULT_MARGINAL_PRICE_H2_DB = DEFAULT_EXPORTS_DIR / "marginal_price_hydrogen_2030.parquet"

HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8736


_ZONE_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2,3}$")
_EXCLUDE_ZONES = {"NL6H", "PL00E", "PL00I"}


def discover_zones_from_xlsx(data_dir=DEFAULT_DATA_DIR) -> list[str]:
    """Zone codes = every ``*.xlsx`` in ``data_dir`` whose name matches a zone code.

    Returns them sorted for reproducibility. Excel lock files (``~$*``),
    ``Networks.xlsx``, non-zone workbooks, and ``_EXCLUDE_ZONES`` are skipped.
    Empty list if the folder can't be read. Only used by build_db.py to
    build zones_2030.parquet in the first place -- discover_zones() below
    reads that database directly at runtime, so normal use never needs the
    raw XLSXs/ folder at all.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return sorted(
        p.stem for p in data_dir.glob("*.xlsx")
        if _ZONE_RE.match(p.stem) and p.stem not in _EXCLUDE_ZONES
        and not p.name.startswith("~$")
    )


@lru_cache(maxsize=8)
def discover_zones(zones_db=DEFAULT_ZONES_DB, data_dir=DEFAULT_DATA_DIR) -> list[str]:
    zones_db = Path(zones_db)
    if zones_db.exists():
        import pandas as pd
        return sorted(pd.read_parquet(zones_db, columns=["zone"])["zone"].unique().tolist())
    return discover_zones_from_xlsx(data_dir)


def _expand_to_countries(zones: list[str], zones_db) -> list[str]:
    all_zones = discover_zones(zones_db)
    countries = {z[:2] for z in zones}
    return sorted(set(zones) | {z for z in all_zones if z[:2] in countries})


@dataclass
class RunConfig:
    zones: list[str] = field(default_factory=lambda: list(ALL_ZONES))
    start_day: int = 1
    end_day: int = 1
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    exports_dir: Path = DEFAULT_EXPORTS_DIR
    zones_db: Path = DEFAULT_ZONES_DB
    networks_db: Path = DEFAULT_NETWORKS_DB
    out_tag: str | None = None

    enable_h2_storage: bool = True
    cyclic_storage: bool = True
    enable_uc: bool = False
    subtract_dsr_implicit: bool = False
    electricity_only: bool = False

    fuel_per_thermal: bool = True
    co2_per_thermal: bool = True
    default_efficiency: float = 0.5
    voll_eur_per_mwh: float = 3_000.0
    h2_terminal_price: float = 150.0
    dump_penalty_eur_per_mwh: float = 0.0
    storage_op_cost_eur_per_mwh: float = 0.01

    initial_soc_fraction: float = 0.5
    ramp_scale: float = 1.0
    default_pump_efficiency: float = 0.8
    default_closed_ps_efficiency: float = 0.75
    h2_storage_hours: float = 168.0
    h2_storage_efficiency: float = 1.0
    default_hydro_efficiency: float = 1.0

    solver_name: str = "highs"
    mip_rel_gap: float = 1e-4

    def __post_init__(self) -> None:
        self.zones = _expand_to_countries(self.zones, self.zones_db)

    def resolved_output_dir(self) -> Path:
        """Output folder for this run: outputs/ or outputs/<out_tag>/ if tagged."""
        base = Path(self.output_dir)
        return base / self.out_tag if self.out_tag else base

    def hour_slice(self) -> tuple[int, int]:
        """Return (start_row, end_row) 0-based half-open into the 8736-hour year.

        Covers the inclusive day range [start_day, end_day], i.e.
        ``num_days() * 24`` hours.
        """
        start = (self.start_day - 1) * HOURS_PER_DAY
        end = self.end_day * HOURS_PER_DAY
        return start, end

    def num_days(self) -> int:
        return self.end_day - self.start_day + 1

    def month_index(self) -> int:
        """Approx calendar month (0-based) of the first day, for must-run selection.

        The dataset year is 364 days (52 weeks); we map to 12 equal ~30.33-day
        months purely to index the 12-value must-run lists. For a multi-day
        horizon the first day's month is used for the whole run.
        """
        day0 = self.start_day - 1
        return min(11, int(day0 / (364 / 12)))
