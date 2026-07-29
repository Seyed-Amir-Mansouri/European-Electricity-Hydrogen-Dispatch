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
    cyclic_storage: bool = True        # end-of-horizon SoC >= initial SoC (full storage cycle)

    # --- Economics (ASSUMPTIONS) ------------------------------------------
    # Marginal cost = VOM Price + fuel_term + co2_term, where
    #   fuel_term = Fuel / eff  if fuel_per_thermal else Fuel
    #   co2_term  = (CO2Factor / eff if co2_per_thermal else CO2Factor) * CO2Price
    # The Fuel and CO2Factor columns are both per MWh_thermal (fuel input, NCV
    # basis) -- e.g. every Gas sub-fleet shares the same ~22.6 EUR/MWh fuel
    # price and 0.1857 t/MWh CO2 factor regardless of efficiency, confirmed
    # against the ENTSO-E "Common Data" reference (Fuel & CO2 price inputs are
    # commodity-basis, not per-MWh_elec) -- so both must be divided by
    # efficiency, or fleets of very different efficiency end up with the same
    # marginal cost and the LP can't tell an OCGT from a CCGT. Validated against
    # PLEXOS: with co2_per_thermal=False the model's price sat ~9.50 EUR/MWh
    # below PLEXOS on average (non-scarcity hours); flipping it to True (this
    # default) cuts that to +1.67 EUR/MWh -- Gas (ccgt_pre2), the largest and
    # cheapest flexible fleet (sets price ~30% of hours), was understated by
    # 15.25 EUR/MWh without this term. See Formulation.md Sec 8.
    fuel_per_thermal: bool = True
    co2_per_thermal: bool = True
    default_efficiency: float = 0.5    # fallback when Efficiency is 0/missing
    voll_eur_per_mwh: float = 3_000.0  # value of lost load (elec & H2 shedding penalty),
                                        # matching PLEXOS's own scarcity price for DE00
    h2_terminal_price: float = 150.0   # EUR/MWh cost of terminal H2 imports (ASSUMPTION)
    dump_penalty_eur_per_mwh: float = 0.0  # penalty for dumping/curtailing excess supply
    # Small per-MWh cost on storage throughput (charge + discharge), to forbid
    # simultaneous charge/discharge without a binary. Not needed for lossy
    # devices (efficiency < 1, e.g. batteries): there charging+discharging at
    # once already wastes energy. It matters for lossless devices (efficiency =
    # 1, e.g. hydro reservoir/pumped-storage and H2 storage), where without it
    # the LP could do both at once. Keep tiny so it doesn't distort economics.
    storage_op_cost_eur_per_mwh: float = 0.01

    # --- Physics defaults --------------------------------------------------
    initial_soc_fraction: float = 0.5  # storage state of charge at hour 0
    ramp_scale: float = 1.0            # multiplier on ramp-rate column
    default_pump_efficiency: float = 0.8   # round-trip eff for open-loop pumped hydro
    # closed_ps gets its own figure rather than sharing default_pump_efficiency:
    # replaying PLEXOS's own hourly turbine/pump MW for DE00 through this
    # model's SoC recursion, sum(discharge)/sum(charge) is 0.8015 for open_ps
    # (matches the 0.8 default almost exactly) but 0.7500 for closed_ps -- at
    # 0.75 closed_ps's SoC trajectory fits its declared energy capacity almost
    # exactly (touches both 0 and the cap over the year); at 0.8 it would still
    # balance, just with slack to spare. See Formulation.md Sec 14.5.
    default_closed_ps_efficiency: float = 0.75  # round-trip eff for closed-loop pumped hydro
    # H2 storage energy capacity (MWh) = Withdraw (Hydrogen) power x h2_storage_hours.
    # ASSUMPTION: the data gives only injection/withdrawal power, no energy capacity.
    h2_storage_hours: float = 168.0
    h2_storage_efficiency: float = 1.0     # H2 storage round-trip efficiency (ASSUMPTION)
    default_hydro_efficiency: float = 1.0  # reservoir/pondage (water, no conversion loss)

    # --- Solver ------------------------------------------------------------
    solver_name: str = "highs"
    mip_rel_gap: float = 1e-4

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
