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
    "AT00", "BE00", "BEOF", "CZ00", "DE00", "DEKF", "FR00", "FR15", "HR00",
    "HU00", "LUB1", "LUF1", "LUG1", "LUV1", "NL00", "NLLL", "PL00",
    "RO00", "SI00", "SK00",
]

# Repo layout: this file is Project 1/economic_dispatch/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "XLSXs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_ZONES_DB = DEFAULT_EXPORTS_DIR / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = DEFAULT_EXPORTS_DIR / "networks_2030.parquet"
DEFAULT_H2_REF = DEFAULT_EXPORTS_DIR / "ReferenceGrid_Hydrogen.xlsx"
# PLEXOS reference output (source of truth for marginal prices outside this
# model's own solve -- e.g. pricing cross-border trade against each
# neighbour's own realized marginal cost; see marginal_price_loader.py).
DEFAULT_PLEXOS_REF = DEFAULT_DATA_DIR / "MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx"
DEFAULT_MARGINAL_PRICE_ELEC_DB = DEFAULT_EXPORTS_DIR / "marginal_price_electricity_2030.parquet"
DEFAULT_MARGINAL_PRICE_H2_DB = DEFAULT_EXPORTS_DIR / "marginal_price_hydrogen_2030.parquet"

HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8736  # 364 days * 24


# A zone code is a 2-letter country prefix + 2-3 alphanumeric subzone id
# (e.g. AT00, BEOF, DE00, NL6H, PL00E). This deliberately excludes Networks.xlsx,
# the PLEXOS MMStandardOutputFile, and any other non-zone workbook that may sit
# in the data folder.
_ZONE_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2,3}$")
# Zones excluded from the study (e.g. empty/degenerate nodes). PL00E/PL00I
# (zero-capacity, zero-demand interconnector hub placeholders for Poland's
# CZ00/DE00/SK00 borders) were replaced by direct links in the updated
# Networks.xlsx and are no longer modelled as separate zones.
_EXCLUDE_ZONES = {"NL6H", "PL00E", "PL00I"}


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
    # Unit commitment (off by default -- the ONE place this project uses
    # integer variables; see model.build_model's fixed_uc_profile /
    # pipeline.solve_scenario's two-pass solve). Currently covers min
    # up/down time (+ minimum stable level while committed); more UC-related
    # constraints (e.g. start-up cost) are meant to layer onto this same flag
    # over time. Only applies to committable fleets with pmin_floor == 0
    # (must-run fleets are already permanently on) AND Min Time On/Off > 1h (a
    # 1h minimum is a no-op at hourly resolution) -- for DE00 that's just
    # Gas (conv_old1) and Gas (ccgt_old1). Verified to change only those two
    # fleets' own dispatch pattern (single-hour "blips" -> real
    # >=MinTime-long committed runs); system-wide price/shed are unaffected
    # at this scale. See Formulation.md Sec 9a.
    enable_uc: bool = False
    # Electricity-only, external-trade pricing: replace the fixed, unpriced
    # net external exchange (exports_loader) with per-neighbour controllable
    # import/export legs, each capped at that border's REAL physical line
    # capacity (Networks.xlsx rating, both directions) and priced at the
    # neighbour's OWN PLEXOS marginal price (0 if the neighbour has no
    # PLEXOS price data, e.g. outside the modelled ENTSO-E area). Only
    # affects the electricity balance; hydrogen external exchange is
    # unaffected. Validated single-zone across all 21 CORE zones against two
    # alternatives -- capping at historical (PLEXOS-realized) flow (mean
    # corr 0.754) and leaving trade fully uncapped (mean corr 0.909) -- real
    # line capacity scored highest (mean corr 0.958) and is the physically
    # correct choice: a real joint solve is limited by actual transmission
    # capacity, not by whatever volume PLEXOS's own solve happened to use.
    # See model._priced_external_elec / exports_loader.elec_border_legs.
    priced_external_elec: bool = False
    # Hydrogen analogue of priced_external_elec: replace the fixed, unpriced
    # net H2 cross-border exchange with per-neighbour-COUNTRY controllable
    # import/export legs (PLEXOS's H2 side is country-granular, not
    # zone-granular -- legs originate from each country's "main H2 zone",
    # see model._h2_main_zones), each capped at that border's REAL physical
    # pipeline capacity (Networks.xlsx "Hydrogen Pipelines" rating, both
    # directions) and priced at the neighbour's own PLEXOS H2 marginal
    # price. Steam-Methane-Reformer domestic production is NOT part of this
    # (it isn't cross-border trade) -- it stays a fixed injection either
    # way. Only affects the hydrogen balance; electricity external exchange
    # is unaffected unless priced_external_elec is also set. See
    # model._priced_external_h2 / exports_loader.h2_border_legs.
    priced_external_h2: bool = False
    # Fix each zone's electrolyser electricity consumption to PLEXOS's own
    # historical "Electrolyser (load) [MW]" profile instead of letting the LP
    # optimise it -- for price-tracking validation against PLEXOS, which
    # dispatches the electrolyser exogenously (outside this LP's economic
    # dispatch). See marginal_price_loader.DEFAULT_ELECTROLYSER_LOAD_DB.
    # When hydrogen is also modelled (not electricity_only), this ALSO fixes
    # the H2-side output directly to PLEXOS's own realized "Electrolyser
    # (gen.) [MWH2]" (country-level -- allocated across a country's zones by
    # each zone's own fixed load share) instead of deriving it from this
    # model's assumed efficiency applied to the electricity-side fix above.
    # See model._build (ely_h2_gen_da) / marginal_price_loader.DEFAULT_ELECTROLYSER_GEN_H2_DB.
    fix_electrolyser_to_plexos: bool = False
    # Subtract PLEXOS's own "Demand Side Response Implicit [MW]" from the
    # electricity demand target, matching how PLEXOS itself defines net
    # demand (this is a signed correction -- activation reduces net demand,
    # deactivation raises it -- not a one-directional relief term). Needed
    # for a fair price-tracking comparison against PLEXOS: without it,
    # DE00's full-year correlation vs PLEXOS regressed from 0.978 to 0.88
    # even after fixing the priced_external_elec sign bug; with both fixes
    # together it's back to 0.978. See marginal_price_loader.DEFAULT_DSR_IMPLICIT_DB.
    subtract_dsr_implicit: bool = False
    # Skip the hydrogen side of the model entirely (no H2 balance, H2
    # storage, H2 terminal imports, or H2 network flows) -- for
    # electricity-only price-tracking validation runs. Hydrogen-fired
    # plants (h2_fuel=True) still dispatch normally (their cost is just VOM,
    # independent of any H2 balance either way); the electrolyser's
    # electricity consumption still subtracts from the elec balance as
    # usual. Cuts LP size substantially at multi-zone scale.
    electricity_only: bool = False
    # REPLACE every renewable generator's hourly availability with PLEXOS's
    # own realized generation for that technology, discarding this model's
    # own capacity x capacity-factor profile calculation entirely (PLEXOS
    # dispatches these zero-marginal-cost, must-take resources at their
    # full available output in virtually every hour, so its realized
    # generation IS its available power). Single-variable techs (wind
    # onshore, wind offshore, run-of-river) get gen_p's own upper bound set
    # directly to the PLEXOS value. Techs where this model splits one
    # PLEXOS category across several of its own generators (solar PV =
    # Solar + Solar (rooftop); solar thermal = Solar (thermal) + Solar
    # (thermal_with_storage); other renewables = the 5 Other RES
    # sub-types) -- since PLEXOS doesn't publish that split -- get each
    # individual generator's own bound set to the full category total (a
    # generous, individually non-binding ceiling) plus a joint
    # sum(gen_p) <= PLEXOS constraint enforcing the real combined limit.
    # Zones with no generator of a technology to begin with are skipped
    # (this replaces an existing generator's bound; it can't fabricate a
    # generator that isn't there at all, e.g. BEOF's missing wind data).
    # See model._override_renewable_upper_with_plexos /
    # model._joint_renewable_constraints.
    cap_renewables_to_plexos: bool = False
    # For hydro storage with real installed capacity but a "...Flow Energy"
    # inflow profile that's all-zero in the zone's own XLSX (a dead resource
    # otherwise -- with charge power 0 for these kinds, zero inflow under the
    # cyclic end-of-horizon closure forces discharge=0 for the whole
    # horizon), use PLEXOS's own realized generation for that hydro kind as
    # the hourly inflow instead. Applies to reservoir, pondage, and
    # open-loop pumped storage only -- NOT closed-loop pumped storage, which
    # has no natural inflow at all (it only recirculates what it pumps).
    # Found via FR15: "Reservoir Flow Energy" was 0 for all 8,736 hours
    # despite 182 MW / 56,800 MWh installed reservoir capacity, which on its
    # own explained all 190 hours of real shedding that zone showed relative
    # to PLEXOS (which never sheds). See model._build_storage.
    fill_missing_hydro_inflow_from_plexos: bool = False

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
