"""Run configuration and tunable assumptions.

Everything a user might reasonably want to change lives here so the model code
stays free of magic numbers. Values flagged "ASSUMPTION" are documented in the
README and are the ones to revisit if results look off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The 23 zone codes shipped in XLSXs/. Used only as a fallback when the data
# folder can't be scanned; the actual zone set is normally auto-discovered from
# the workbooks present (see discover_zones), so adding/removing a zone file
# "just works".
ALL_ZONES = [
    "AT00", "BE00", "BEOF", "CZ00", "DE00", "DEKF", "FR00", "HR00",
    "HU00", "LUB1", "LUF1", "LUG1", "LUV1", "NL00", "NLLL", "PL00",
    "PL00E", "PL00I", "RO00", "SI00", "SK00",
]

# Repo layout: this file is Project 1/economic_dispatch/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "XLSXs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_ZONES_DB = DEFAULT_EXPORTS_DIR / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = DEFAULT_EXPORTS_DIR / "networks_2030.parquet"
DEFAULT_H2_REF = DEFAULT_EXPORTS_DIR / "ReferenceGrid_Hydrogen.xlsx"

HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8736  # 364 days * 24


# A zone code is a 2-letter country prefix + 2-3 alphanumeric subzone id
# (e.g. AT00, BEOF, DE00, NL6H, PL00E). This deliberately excludes Networks.xlsx,
# the PLEXOS MMStandardOutputFile, and any other non-zone workbook that may sit
# in the data folder.
_ZONE_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2,3}$")
# Zones excluded from the study (e.g. empty/degenerate nodes).
_EXCLUDE_ZONES = {"FR15", "NL6H"}


def discover_zones(data_dir=DEFAULT_DATA_DIR) -> list[str]:
    """Zone codes = every ``*.xlsx`` in ``data_dir`` whose name matches a zone code.

    Returns them sorted for reproducibility. Excel lock files (``~$*``),
    ``Networks.xlsx``, non-zone workbooks, and ``_EXCLUDE_ZONES`` are skipped.
    Empty list if the folder can't be read.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return sorted(
        p.stem for p in data_dir.glob("*.xlsx")
        if _ZONE_RE.match(p.stem) and p.stem not in _EXCLUDE_ZONES
        and not p.name.startswith("~$")
    )


@dataclass
class RunConfig:
    # --- Scope -------------------------------------------------------------
    zones: list[str] = field(default_factory=lambda: list(ALL_ZONES))
    start_day: int = 1                 # 1-based first day of the horizon
    end_day: int = 1                   # 1-based last day (inclusive); == start_day for one day
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    exports_dir: Path = DEFAULT_EXPORTS_DIR   # inputs/ result databases (parquet)
    zones_db: Path = DEFAULT_ZONES_DB         # consolidated zone data (parquet)
    networks_db: Path = DEFAULT_NETWORKS_DB   # line topology + prices (parquet)
    out_tag: str | None = None         # write to outputs/<out_tag>/ to keep runs side by side

    # --- Feature flags -----------------------------------------------------
    enable_storage: bool = True        # battery + hydro reservoir/pumped storage
    enable_ramps: bool = True          # generator ramp-rate limits
    enable_reserves: bool = False      # FCR/FRR headroom constraints (off by default)
    enable_h2_terminal: bool = True    # allow external H2 supply at import terminals
    enable_h2_storage: bool = True     # model H2 storage (Injection/Withdraw Hydrogen power)
    cyclic_storage: bool = True        # end-of-day SoC must return to initial SoC
    compute_prices: bool = True        # zonal marginal prices (LP re-solve with commitment fixed)

    # --- Economics (ASSUMPTIONS) ------------------------------------------
    # Marginal cost = VOM Price + fuel_term + co2_term, where
    #   fuel_term = Fuel / eff  if fuel_per_thermal else Fuel
    #   co2_term  = (CO2Factor / eff if co2_per_thermal else CO2Factor) * CO2Price
    # Both the Fuel and CO2Factor columns are already per MWh_elec (per power
    # generation), so neither is divided by efficiency. Flip a flag to True only
    # if the corresponding column is provided per MWh_thermal instead.
    fuel_per_thermal: bool = False
    co2_per_thermal: bool = False
    default_efficiency: float = 0.5    # fallback when Efficiency is 0/missing
    voll_eur_per_mwh: float = 10_000.0  # value of lost load (elec & H2 shedding penalty)
    h2_terminal_price: float = 150.0   # EUR/MWh cost of terminal H2 imports (ASSUMPTION)
    dump_penalty_eur_per_mwh: float = 0.0  # penalty for dumping/curtailing excess supply

    # --- Physics defaults --------------------------------------------------
    initial_soc_fraction: float = 0.5  # storage state of charge at hour 0
    ramp_scale: float = 1.0            # multiplier on ramp-rate column
    default_pump_efficiency: float = 0.8   # round-trip eff for pumped hydro if missing
    # H2 storage energy capacity (MWh) = Withdraw (Hydrogen) power x h2_storage_hours.
    # ASSUMPTION: the data gives only injection/withdrawal power, no energy capacity.
    h2_storage_hours: float = 168.0
    h2_storage_efficiency: float = 1.0     # H2 storage round-trip efficiency (ASSUMPTION)
    default_hydro_efficiency: float = 1.0  # reservoir/pondage (water, no conversion loss)

    # --- Solver ------------------------------------------------------------
    solver_name: str = "highs"
    mip_rel_gap: float = 1e-4
    recover_prices: bool = False       # fix commitment, re-solve LP for shadow prices
    rolling_block_days: int = 0        # >0: rolling-horizon solve in day-blocks (long horizons)

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
