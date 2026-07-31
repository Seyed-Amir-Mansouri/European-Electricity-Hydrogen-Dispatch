"""Build the coupled electricity + hydrogen dispatch LP with linopy.

Design
------
* One global index of "generators" (``gen``) spanning every supply resource
  across all zones, plus indices for storage, electrolysers, and network lines.
* Zone balances are assembled with incidence DataArrays (``A[gen, zone]``) via
  ``(A * var).sum("gen")`` — fully vectorised over zones and hours.
* The model is a pure LP: thermal fleets dispatch continuously between a
  must-run floor and capacity (no integer commitment), and a small storage
  throughput cost forbids simultaneous charge/discharge without a binary.
* Inter-temporal constraints (ramps, storage state-of-charge) are expressed as
  vectorised recursions with ``.shift()``; everything else is vectorised too.
* Bidirectional lines are split into two non-negative flow variables so that the
  fractional line loss can be applied on the receiving end unambiguously.

The returned :class:`BuildResult` carries the linopy model and all lookup tables
needed by report.py to extract and validate the solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import RunConfig
from . import data_loader as dl
from . import exports_loader
from . import network_loader as nl
from .data_loader import ZoneData
from .network_loader import NetworkData, Line

HOUR = "hour"
GEN = "gen"
ZONE = "zone"
STO = "sto"

# UC data (cfg.enable_uc): "Minimum Up Time (h)", "Minimum Down Time (h)",
# "Start-Up Cost (EUR)" are now zone-specific Characteristics columns (added
# directly to each zone's own XLSX, sourced from XLSXs/Common Data.xlsx's
# per-technology values -- warm-start fuel+wear cost, converted to a flat
# EUR/MW-of-capacity figure -- so the model no longer reads Common Data.xlsx
# itself; every zone carries its own values for every thermal tech). A tech
# is only a real UC candidate if it ALSO has no must-run floor (must-run
# fleets are already permanently on) and > 1h min time (1h is a no-op at
# hourly resolution) -- see uc_candidates() below.


def _num(arr) -> np.ndarray:
    """Coerce to float array with NaN/inf replaced by 0 (blank profile cells)."""
    return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Resource assembly (plain Python tables built from the parsed workbooks)
# --------------------------------------------------------------------------- #
@dataclass
class BuildResult:
    model: linopy.Model
    cfg: RunConfig
    zones: list[str]
    hours: pd.Index
    gens: pd.DataFrame          # indexed by gen_id: zone, tech, category, mc, ...
    commit: pd.DataFrame        # subset of gens = dispatchable thermal fleets (must-run floor)
    storage: pd.DataFrame       # indexed by sto_id
    gen_upper: xr.DataArray     # (gen, hour) available capacity
    demand_e: xr.DataArray      # (zone, hour)
    demand_h: xr.DataArray
    external_e: xr.DataArray    # fixed electricity exchange with non-modelled zones
    external_h2: xr.DataArray   # fixed hydrogen exchange with non-modelled zones
    elines: list[Line]
    hlines: list[Line]
    net: NetworkData
    price_e: xr.DataArray | None = None   # elec marginal price (zone, hour), EUR/MWh
    price_h: xr.DataArray | None = None   # H2 marginal price (zone, hour), EUR/MWh
    uc_gens: list[str] | None = None      # gen_ids with cfg.enable_uc's commitment binary (pass 1 only)
    startup_cost_eur: float = 0.0         # total UC start-up cost incurred (pass 1's y_start solution x cost)


def _marginal_cost(zd: ZoneData, tech: str, h2_fuel: bool, co2_price: float,
                   cfg: RunConfig) -> float:
    """Short-run marginal cost (EUR/MWh_elec) for a dispatchable/profile tech."""
    vom = zd.char_val(tech, "Price (EUR/MWh)", 0.0)
    fuel = zd.char_val(tech, "Fuel (EUR/MWh)", 0.0)
    co2f = zd.char_val(tech, "CO2 Factor (ton/MWh)", 0.0)
    if h2_fuel:
        # Hydrogen comes from the H2 balance (priced endogenously) — do NOT also
        # charge the exogenous H2 fuel price, or it would be double counted.
        return vom
    eff = zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0
    e = eff if eff > 0 else cfg.default_efficiency
    fuel_term = fuel / e if cfg.fuel_per_thermal else fuel
    co2_term = (co2f / e if cfg.co2_per_thermal else co2f) * co2_price
    return vom + fuel_term + co2_term


def _build_generators(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig):
    """Return (gens DataFrame, per-gen hourly upper-bound array as dict)."""
    month = cfg.month_index()
    rows: list[dict] = []
    upper: dict[str, np.ndarray] = {}
    H = len(zdata[cfg.zones[0]].profiles)

    for z in cfg.zones:
        zd = zdata[z]
        for tech, cap in zd.capacities.items():
            cap = float(cap or 0.0)
            category, h2_fuel = dl.classify(tech)
            gid = f"{z}|{tech}"

            if category == dl.CAT_COMMIT:
                if cap <= 0:
                    continue
                units = int(round(zd.char_val(tech, "Number of Units", 0.0)))
                units = max(units, 1)  # capacity>0 implies at least one unit
                # "Maximum Number of Units in Maintenance" is a scheduling ceiling
                # (often == total units), not a forced outage, so we do not derate
                # by it in a single-day dispatch. Full fleet is committable.
                max_units = units
                pmax_unit = cap / units
                msp = zd.char_val(tech, "Minimum Stable Power (%)", 0.0) / 100.0
                pmin_unit = pmax_unit * msp
                ramp_pu = zd.char_val(tech, "Ramp-Up Rate (MW/h)", 0.0)
                ramp_dn = zd.char_val(tech, "Ramp-Down Rate (MW/h)", 0.0)
                # Must-run floor: "Must Run (%)" is the share of INSTALLED
                # CAPACITY that must run (not a unit count -- "Must Run (Number
                # of units)" is a separate, unreliable column that reads 0 even
                # for fleets "Must Run (%)" shows as partially must-run).
                mustrun_pct = zd.must_run_pct(tech, month)
                mustrun_pct = float(min(max(mustrun_pct, 0.0), 100.0))
                pmin_floor = (mustrun_pct / 100.0) * cap if mustrun_pct > 0 else 0.0
                # Missing/zero efficiency -> use the default (avoids a 1/eff = 1e6
                # coefficient in the H2 balance that ruins the LP conditioning).
                eff = zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0
                eff = eff if eff > 1e-3 else cfg.default_efficiency
                # UC data (cfg.enable_uc), all zone-specific Characteristics
                # columns now: Min Up/Down Time (h) and Start-Up Cost (EUR),
                # the latter already a flat EUR/MW-of-capacity figure -- just
                # scale by fleet capacity for the total cost per start event.
                min_up_h = zd.char_val(tech, "Minimum Up Time (h)", 0.0)
                min_down_h = zd.char_val(tech, "Minimum Down Time (h)", 0.0)
                startup_cost_per_mw = zd.char_val(tech, "Start-Up Cost (EUR)", 0.0)
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=h2_fuel,
                    mc=_marginal_cost(zd, tech, h2_fuel, net.co2_price, cfg),
                    eff=eff,
                    units=max_units, pmin_unit=pmin_unit, pmax_unit=pmax_unit,
                    ramp_up=ramp_pu * units * cfg.ramp_scale,
                    ramp_dn=ramp_dn * units * cfg.ramp_scale,
                    mustrun_pct=mustrun_pct, pmin_floor=pmin_floor, pmax=cap,
                    msl_frac=msp,  # "Minimum Stable Power (%)" as a fraction, for cfg.enable_uc
                    min_up_h=min_up_h, min_down_h=min_down_h,
                    startup_cost_eur=startup_cost_per_mw * cap,  # total EUR per start event
                ))
                upper[gid] = np.full(H, cap, dtype=float)

            elif category == dl.CAT_VRES:
                if cap <= 0:
                    continue
                col = dl.VRES_PROFILE.get(tech)
                if col is None or col not in zd.profiles:
                    continue
                cf = _num(zd.profiles[col].to_numpy())
                avail = np.clip(cf, 0.0, None) * cap
                if avail.max() <= 0:
                    continue
                rows.append(dict(gen=gid, zone=z, tech=tech, category=category,
                                 h2_fuel=False, mc=0.0, eff=1.0, pmax=cap))
                upper[gid] = avail

            elif category == dl.CAT_ROR:
                col = "River Flow Energy"
                if col not in zd.profiles:
                    continue
                inflow = np.clip(_num(zd.profiles[col].to_numpy()), 0.0, None)
                avail = np.minimum(inflow, cap) if cap > 0 else inflow
                if avail.max() <= 0:
                    continue
                rows.append(dict(gen=gid, zone=z, tech=tech, category=category,
                                 h2_fuel=False, mc=0.0, eff=1.0, pmax=float(avail.max())))
                upper[gid] = avail

            elif category == dl.CAT_PROFILE:
                col = dl.profile_gen_column(tech)
                if col not in zd.profiles:
                    continue
                avail = np.clip(_num(zd.profiles[col].to_numpy()), 0.0, None)
                if avail.max() <= 0:
                    continue
                # "Number of Hours (h)" (DSR blocks ONLY): a daily
                # activation-hours limit from the source contract/product
                # definition -- e.g. a block with Hours=1 can only be worth
                # its full capacity for ~1 hour's worth of energy per day.
                # Modelled as a continuous daily ENERGY cap (pmax * hours),
                # not a true discrete hour-count (would need a binary per
                # hour): with a fixed daily budget, cost-minimization
                # naturally concentrates dispatch on the highest-value
                # hour(s) anyway, a close LP-only approximation. inf
                # (missing data, or non-DSR techs) = no limit -- confirmed
                # "Other Non-RES1/2/3" carry Hours=0 in this data, which is
                # an unpopulated-field artifact, not a real activation limit
                # (PLEXOS actually dispatches them freely); scoping this to
                # DSR avoids incorrectly zeroing those out.
                hours_limit = (zd.char_val(tech, "Number of Hours (h)", float("inf"))
                              if tech.startswith("DSR") else float("inf"))
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=False,
                    mc=_marginal_cost(zd, tech, False, net.co2_price, cfg),
                    eff=1.0, pmax=float(avail.max()), daily_hours_limit=hours_limit))
                upper[gid] = avail

    if rows:
        gens = pd.DataFrame(rows).set_index("gen")
    else:
        # A zone can legitimately have zero generators (e.g. a pure
        # interconnector/offshore node with no local demand and a capacity
        # whose availability profile is all-zero, like BEOF) -- an empty
        # gens table with the columns every downstream lookup needs (zone,
        # category, h2_fuel, mc, eff, pmax) is a valid, solvable case (zero
        # generation, balance closes via network flows/shed/dump), not an
        # error.
        gens = pd.DataFrame(
            columns=["zone", "tech", "category", "h2_fuel", "mc", "eff", "pmax"]
        ).set_index(pd.Index([], name="gen"))
    return gens, upper


# Storage device specs: (name, discharge cap source, charge cap source, energy key,
# inflow profile column, efficiency source).  Sources resolved per zone below.
def _build_storage(zdata: dict[str, ZoneData], cfg: RunConfig):
    rows: list[dict] = []
    inflow: dict[str, np.ndarray] = {}
    H = len(zdata[cfg.zones[0]].profiles)
    zero = np.zeros(H)

    for z in cfg.zones:
        zd = zdata[z]
        cap = zd.capacities
        e = zd.storage_energy
        prof = zd.profiles

        def col(name):
            return np.clip(_num(prof[name].to_numpy()), 0.0, None) if name in prof else zero

        specs = [
            # kind, pdis, pchg, ecap, inflow, eff, carrier
            ("Battery",
             zd.char_val("Battery (MWh)", "Net maximum capacity - generation perspective (MW)"),
             zd.char_val("Battery (MWh)", "Net maximum capacity - demand perspective (MW)"),
             e.get("Battery (MWh)", 0.0), zero,
             max(zd.char_val("Battery (MWh)", "Efficiency (%)", 92.0) / 100.0, 0.1), "electricity"),
            ("Hydro reservoir", cap.get("Hydro (reservoir) (MW)", 0.0), 0.0,
             e.get("Hydro (reservoir) (MWh)", 0.0), col("Reservoir Flow Energy"),
             cfg.default_hydro_efficiency, "electricity"),
            ("Hydro pondage", cap.get("Hydro (pondage) (MW)", 0.0), 0.0,
             e.get("Hydro (pondage) (MWh)", 0.0), col("Pondage Flow Energy"),
             cfg.default_hydro_efficiency, "electricity"),
            ("Hydro open_ps", cap.get("Hydro (open_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (open_ps_pump) (MW)", 0.0)),
             e.get("Hydro (open_ps) (MWh)", 0.0), col("Open_PS Flow Energy"),
             cfg.default_pump_efficiency, "electricity"),
            ("Hydro closed_ps", cap.get("Hydro (closed_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (closed_ps_pump) (MW)", 0.0)),
             e.get("Hydro (closed_ps) (MWh)", 0.0), col("Closed_PS Flow Energy"),
             cfg.default_closed_ps_efficiency, "electricity"),
        ]
        if cfg.enable_h2_storage and not cfg.electricity_only:
            wd = zd.h2_assets.get("Withdraw (Hydrogen) (MW)", 0.0)      # discharge power
            inj = zd.h2_assets.get("Injection (Hydrogen) (MW)", 0.0)   # charge power
            specs.append(("H2 storage", wd, inj, wd * cfg.h2_storage_hours, zero,
                          cfg.h2_storage_efficiency, "hydrogen"))
        for kind, pdis, pchg, ecap, inf, eff, carrier in specs:
            if ecap <= 0 or pdis <= 0:
                continue
            sid = f"{z}|{kind}"
            rows.append(dict(sto=sid, zone=z, kind=kind, pdis=float(pdis),
                             pchg=float(pchg), ecap=float(ecap), eff=float(eff),
                             carrier=carrier))
            inflow[sid] = inf

    storage = pd.DataFrame(rows).set_index("sto") if rows else pd.DataFrame(
        columns=["zone", "kind", "pdis", "pchg", "ecap", "eff", "carrier"]).rename_axis("sto")
    return storage, inflow


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _incidence(members: pd.Series, zones: list[str], dim: str) -> xr.DataArray:
    """One-hot (member, zone) matrix from a Series mapping member -> zone."""
    A = np.zeros((len(members), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, z in enumerate(members.to_numpy()):
        A[i, zpos[z]] = 1.0
    return xr.DataArray(A, coords={dim: members.index, ZONE: zones}, dims=[dim, ZONE])


def uc_candidates(gens: pd.DataFrame) -> list[str]:
    """Fleets eligible for cfg.enable_uc: no must-run floor (so they can
    genuinely be off -- must-run fleets are already permanently on via their
    continuous floor) AND min_up_h/min_down_h > 1h (a 1h minimum is a no-op
    at hourly resolution)."""
    return [gid for gid, row in gens.iterrows()
            if row.get("pmin_floor", 0.0) == 0.0
            and max(row.get("min_up_h", 0.0), row.get("min_down_h", 0.0)) > 1.0]


def build_model(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig,
                cyclic: bool | None = None,
                fixed_uc_profile: dict[str, np.ndarray] | None = None) -> BuildResult:
    """Build the dispatch LP (or, with ``cfg.enable_uc``, a small MILP).

    ``cyclic`` controls the end-of-horizon storage closure:
      * ``None`` -> use ``cfg.cyclic_storage``;
      * ``True`` -> ``soc[T-1] >= soc0`` (every device ends no lower than it
        started, i.e. a full storage cycle over the horizon);
      * ``False`` -> no closure constraint.

    ``fixed_uc_profile``: only meaningful with ``cfg.enable_uc``.
    ``None`` (pass 1) builds the MILP: a commitment binary + start/stop +
    min-up/down-time constraints for ``uc_candidates(gens)``. A dict
    ``{gen_id: np.array(H)}`` (pass 2) instead bakes a solved 0/pmax
    commitment schedule in as fixed capacity data and builds a PURE LP with
    no binaries at all -- needed because HiGHS/linopy cannot return duals
    (marginal prices) once any integer variable exists in the model, even
    after it's solved. See pipeline.solve_scenario for the two-pass orchestration.
    """
    zones = cfg.zones
    H = len(zdata[zones[0]].profiles)
    hours = pd.Index(range(H), name=HOUR)
    zidx = pd.Index(zones, name=ZONE)

    gens, gupper = _build_generators(zdata, net, cfg)
    if cfg.cap_renewables_to_plexos:
        gens, gupper = _override_renewable_upper_with_plexos(zdata, gens, gupper, zones, cfg)
    storage, sinflow = _build_storage(zdata, cfg)

    uc_gens = uc_candidates(gens) if cfg.enable_uc else []

    m = linopy.Model()

    # ---- generation (pure LP, no commitment binary -- except uc_gens when -#
    # ---- cfg.enable_uc is set) -------------------------------------------- #
    # Each fleet's output floats between a fixed floor and its available
    # capacity. The floor is "Must Run (%)" of installed capacity; resources
    # with no must-run requirement have a zero floor, so there is no need for
    # an integer on/off variable -- except uc_gens, optionally (see below).
    gen_index = gens.index
    upper_mat = np.vstack([gupper[g] for g in gen_index]) if len(gen_index) > 0 else np.zeros((0, H))
    gen_upper = xr.DataArray(upper_mat, coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR])
    floor_vec = np.nan_to_num(gens["pmin_floor"].to_numpy(float)) \
        if "pmin_floor" in gens.columns else np.zeros(len(gen_index))
    gen_lower = xr.DataArray(
        np.tile(floor_vec[:, None], (1, H)), coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR]
    )
    if fixed_uc_profile is not None:
        gen_upper = gen_upper.copy()
        for gid, prof in fixed_uc_profile.items():
            msl_frac = float(gens.loc[gid, "msl_frac"])
            gen_upper.loc[{GEN: gid}] = prof
            gen_lower.loc[{GEN: gid}] = msl_frac * prof  # prof is 0 (off) or pmax (on)
    gen_p = m.add_variables(lower=gen_lower, upper=gen_upper, name="gen_p")

    if cfg.cap_renewables_to_plexos:
        _joint_renewable_constraints(m, gens, gen_p, zones, hours, cfg)

    A_gen = _incidence(gens["zone"], zones, GEN)
    gen_by_zone = (A_gen * gen_p).sum(GEN)

    # ---- unit commitment: min up/down time (cfg.enable_uc) ------- #
    # Standard rolling-window formulation (Rajan & Takriti): a commitment
    # binary x_on, start/stop indicators y/z tied to it by x_t - x_{t-1} =
    # y_t - z_t (unit assumed off before the horizon), and a window-sum cap
    # linking start-ups/shut-downs to the current on/off state over the
    # technology's own Min Time On/Off. gen_p is capped at pmax when on, 0
    # when off, and floored at msl_frac*pmax when on (Minimum Stable Power
    # (%), this zone's own data) -- without that floor the commitment
    # constraint is satisfiable with zero real output for most of a
    # committed block, which looks identical to the "blip" behaviour it's
    # meant to prevent. Only built when fixed_uc_profile is None (pass 1);
    # pass 2 bakes the solved schedule in as data above instead.
    uc_x_on = None
    uc_startup_obj = 0.0
    if fixed_uc_profile is None and uc_gens:
        uc_idx = pd.Index(uc_gens, name=GEN)
        uc_x_on = m.add_variables(binary=True, coords=[uc_idx, hours], name="uc_on")
        uc_y_start = m.add_variables(lower=0.0, upper=1.0, coords=[uc_idx, hours], name="uc_start")
        uc_z_stop = m.add_variables(lower=0.0, upper=1.0, coords=[uc_idx, hours], name="uc_stop")

        uc_cap = xr.DataArray(gens.loc[uc_gens, "pmax"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN])
        uc_msl = xr.DataArray(gens.loc[uc_gens, "msl_frac"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN])
        m.add_constraints(gen_p.sel({GEN: uc_idx}) <= uc_cap * uc_x_on, name="uc_cap_link")
        m.add_constraints(gen_p.sel({GEN: uc_idx}) >= uc_msl * uc_cap * uc_x_on, name="uc_msl_link")

        delta_x = uc_x_on - uc_x_on.shift({HOUR: 1}, fill_value=0.0)
        m.add_constraints(delta_x == uc_y_start - uc_z_stop, name="uc_startstop")

        for gid in uc_gens:
            min_on = max(int(round(gens.loc[gid, "min_up_h"])), 1)
            min_off = max(int(round(gens.loc[gid, "min_down_h"])), 1)
            y_g = uc_y_start.sel({GEN: gid})
            z_g = uc_z_stop.sel({GEN: gid})
            x_g = uc_x_on.sel({GEN: gid})
            roll_y = sum(y_g.shift({HOUR: k}, fill_value=0.0) for k in range(min_on))
            roll_z = sum(z_g.shift({HOUR: k}, fill_value=0.0) for k in range(min_off))
            m.add_constraints(roll_y <= x_g, name=f"uc_minup_{gid}")
            m.add_constraints(roll_z <= 1 - x_g, name=f"uc_mindown_{gid}")

        # Start-up cost: charged once per start-up event (uc_y_start==1), not
        # per hour committed -- gens["startup_cost_eur"] is the zone's own
        # "Start-Up Cost (EUR)" characteristic (EUR/MW) x fleet capacity.
        uc_startup_cost = xr.DataArray(
            gens.loc[uc_gens, "startup_cost_eur"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN]
        )
        uc_startup_obj = (uc_startup_cost * uc_y_start).sum()

    commit = gens[gens["category"] == dl.CAT_COMMIT].copy()

    # ---- daily activation-hours cap (DSR blocks) -------------------------- #
    # Continuous energy-cap approximation of "Number of Hours (h)" (see
    # _build_generators): per calendar day, total dispatch <= pmax * hours.
    if "daily_hours_limit" in gens.columns:
        capped = gens[np.isfinite(gens["daily_hours_limit"].to_numpy(float))]
        if len(capped) > 0:
            cap_idx = capped.index
            day_budget = xr.DataArray(
                capped["pmax"].to_numpy(float) * capped["daily_hours_limit"].to_numpy(float),
                coords={GEN: cap_idx}, dims=[GEN],
            )
            gp_capped = gen_p.sel({GEN: cap_idx})
            n_days = len(hours) // 24
            for d in range(n_days):
                day_hours = hours[d * 24:(d + 1) * 24]
                window = gp_capped.sel({HOUR: day_hours}).sum(HOUR)
                m.add_constraints(window <= day_budget, name=f"daily_hours_cap_{d}")

    # ---- storage --------------------------------------------------------- #
    have_sto = cfg.enable_storage and len(storage) > 0
    if have_sto:
        sidx = storage.index
        pdis = xr.DataArray(storage["pdis"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        pchg = xr.DataArray(storage["pchg"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        ecap = xr.DataArray(storage["ecap"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        eff = storage["eff"].to_numpy(float)
        dis = m.add_variables(lower=0.0, upper=_bc(pdis, hours), name="dis")
        ch = m.add_variables(lower=0.0, upper=_bc(pchg, hours), name="ch")
        soc = m.add_variables(lower=0.0, upper=_bc(ecap, hours), name="soc")
        spill = m.add_variables(lower=0.0, name="spill", coords=[sidx, hours])

        A_sto = _incidence(storage["zone"], zones, STO)
        # Route each device's charge/discharge to its carrier's balance.
        carr = storage["carrier"].to_numpy()
        mask_e = xr.DataArray((carr == "electricity").astype(float), coords={STO: sidx}, dims=[STO])
        mask_h = xr.DataArray((carr == "hydrogen").astype(float), coords={STO: sidx}, dims=[STO])
        dis_by_zone = (A_sto * mask_e * dis).sum(STO)
        ch_by_zone = (A_sto * mask_e * ch).sum(STO)
        dis_h2_by_zone = (A_sto * mask_h * dis).sum(STO)
        ch_h2_by_zone = (A_sto * mask_h * ch).sum(STO)

        soc0 = cfg.initial_soc_fraction * storage["ecap"].to_numpy(float)
        inflow_mat = np.vstack([sinflow[s] for s in sidx])            # (sto, H)
        eff_da = xr.DataArray(eff, coords={STO: sidx}, dims=[STO])
        # One vectorised recursion instead of a per-hour loop:
        #   soc[h] - soc[h-1] - eff*ch[h] + dis[h] + spill[h] = inflow[h]
        # ``soc.shift(hour=1)`` is empty at h=0, so inject the initial SoC into
        # the RHS only there (where "soc[h-1]" would otherwise be the start value).
        rhs_mat = inflow_mat.copy()
        rhs_mat[:, 0] = rhs_mat[:, 0] + soc0
        rhs = xr.DataArray(rhs_mat, coords={STO: sidx, HOUR: hours}, dims=[STO, HOUR])
        m.add_constraints(soc - soc.shift({HOUR: 1}) - eff_da * ch + dis + spill == rhs,
                          name="soc_balance")
        if cfg.cyclic_storage if cyclic is None else cyclic:
            # Full storage cycle: every device ends the horizon no lower than it
            # began it, soc[T-1] >= soc0 (the initial state of charge).
            end = xr.DataArray(soc0, coords={STO: sidx}, dims=[STO])
            m.add_constraints(soc.sel({HOUR: H - 1}) >= end, name="soc_cyclic")
    else:
        dis_by_zone = ch_by_zone = 0.0
        dis_h2_by_zone = ch_h2_by_zone = 0.0

    # ---- electrolysers & H2 terminals ----------------------------------- #
    # ely_p always exists (it subtracts from the elec balance either way);
    # everything else in this block is hydrogen-side and skipped under
    # cfg.electricity_only.
    ely_cap = np.array([zdata[z].capacities.get("Electrolyser (MW)", 0.0) for z in zones])
    ely_eff = np.array([max(zdata[z].char_val("Electrolyser (MW)", "Efficiency (%)", 68.0) / 100.0, 1e-6)
                        for z in zones])
    ely_p = m.add_variables(lower=0.0, upper=_bc_z(ely_cap, zidx, hours), name="ely_p")

    if cfg.fix_electrolyser_to_plexos:
        # Pin electrolyser consumption to PLEXOS's own historical dispatch
        # (exogenous to this LP) instead of letting it optimise -- used for
        # price-tracking validation. Clip to this model's own capacity so the
        # equality can't exceed ely_p's upper bound.
        from . import marginal_price_loader as mpl
        h0, h1 = cfg.hour_slice()
        ghours = pd.RangeIndex(h0, h1)
        ely_hist = mpl.load_zone_series(zones, ghours, mpl.DEFAULT_ELECTROLYSER_LOAD_DB).to_numpy().T
        ely_hist = np.minimum(np.clip(ely_hist, 0.0, None), np.vstack([ely_cap] * H).T)
        ely_fixed_da = xr.DataArray(ely_hist, coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        m.add_constraints(ely_p == ely_fixed_da, name="ely_p_fixed")

    if not cfg.electricity_only:
        ely_eff_da = xr.DataArray(ely_eff, coords={ZONE: zidx}, dims=[ZONE])
        if cfg.enable_h2_terminal:
            term_cap = np.array([zdata[z].h2_assets.get("Terminal (Hydrogen) (MW)", 0.0) for z in zones])
        else:
            term_cap = np.zeros(len(zones))
        term_h2 = m.add_variables(lower=0.0, upper=_bc_z(term_cap, zidx, hours), name="term_h2")

        # H2 consumed by hydrogen-fired plants: gen_p / eff, mapped to zone
        A_h2 = A_gen.copy()
        h2_coeff = np.where(gens["h2_fuel"].to_numpy(), 1.0 / gens["eff"].to_numpy(), 0.0)
        A_h2 = A_h2 * xr.DataArray(h2_coeff, coords={GEN: gen_index}, dims=[GEN])
        h2_cons_by_zone = (A_h2 * gen_p).sum(GEN)
    else:
        term_cap = np.zeros(len(zones))

    # ---- network flows (split directional, loss on receiving end) -------- #
    net_e, fe_pos, fe_neg = _flow_terms(m, net.elec, zones, hours, "e")
    net_h, fh_pos, fh_neg = (0.0, None, None) if cfg.electricity_only \
        else _flow_terms(m, net.hydrogen, zones, hours, "h")

    # ---- demand / fixed exchange ---------------------------------------- #
    demand_e = _profile_da(zdata, zones, hours, "Electricity Demand Profile")
    demand_h = _profile_da(zdata, zones, hours, "Hydrogen Demand Profile") if not cfg.electricity_only else None
    if cfg.subtract_dsr_implicit:
        # PLEXOS's own "Demand Side Response Implicit [MW]" is a signed
        # correction (activation reduces net demand, deactivation raises it)
        # PLEXOS applies on top of raw demand; our own demand target should
        # match it for a fair price comparison. Validated on DE00 (part of
        # closing a correlation regression from 0.74 -> 0.98 alongside the
        # priced_external_elec sign fix).
        from . import marginal_price_loader as mpl
        h0, h1 = cfg.hour_slice()
        ghours = pd.RangeIndex(h0, h1)
        dsr_df = mpl.load_zone_series(zones, ghours, mpl.DEFAULT_DSR_IMPLICIT_DB)
        dsr_da = xr.DataArray(dsr_df.to_numpy().T, coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        demand_e = demand_e - dsr_da
    ext_e_obj = 0.0
    if cfg.priced_external_elec:
        external_e, ext_e_obj = _priced_external_elec(m, zones, hours, cfg)
        _, external_h2 = _external_exchange_all(zdata, zones, hours, cfg)
    else:
        external_e, external_h2 = _external_exchange_all(zdata, zones, hours, cfg)

    shed_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_e")
    # Dump/curtailment slacks absorb EXCESS supply (e.g. a fixed net import that
    # exceeds absorbable load) — the counterpart of shedding, as in PLEXOS's
    # "Dumped" category. Without them the equality balance can be infeasible.
    dump_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_e")
    if not cfg.electricity_only:
        shed_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_h")
        dump_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_h")

    # ---- balances -------------------------------------------------------- #
    # external_e / external_h2 are net imports (import +), so they add to supply.
    elec_lhs = (gen_by_zone + dis_by_zone - ch_by_zone - ely_p + net_e
                + external_e + shed_e - dump_e)
    m.add_constraints(elec_lhs == demand_e, name="elec_balance")

    if not cfg.electricity_only:
        h2_lhs = (ely_eff_da * ely_p + term_h2 + net_h + external_h2
                  + dis_h2_by_zone - ch_h2_by_zone + shed_h
                  - h2_cons_by_zone - dump_h)
        m.add_constraints(h2_lhs == demand_h, name="h2_balance")

    # ---- ramps ----------------------------------------------------------- #
    # uc_gens are excluded here (in both MILP and fixed-profile passes): with
    # an MSL floor, start-up means jumping straight to msl_frac*pmax in one
    # hour, which their physical ramp rate can't reach -- PLEXOS itself
    # exempts start/stop transitions from ramp limits for exactly this
    # reason (Sec 10); their timing is governed by the commitment + min
    # up/down-time logic instead.
    ramp_commit = commit.drop(index=uc_gens, errors="ignore") if uc_gens else commit
    ramp_cidx = ramp_commit.index
    if cfg.enable_ramps and len(ramp_commit) > 0:
        rup = xr.DataArray(ramp_commit["ramp_up"].to_numpy(float), coords={GEN: ramp_cidx}, dims=[GEN])
        rdn = xr.DataArray(ramp_commit["ramp_dn"].to_numpy(float), coords={GEN: ramp_cidx}, dims=[GEN])
        gp_c = gen_p.sel({GEN: ramp_cidx})
        # delta[h] = gen[h] - gen[h-1] for h >= 1 (drop hour 0: no predecessor).
        delta = (gp_c - gp_c.shift({HOUR: 1})).isel({HOUR: slice(1, None)})
        if ramp_commit["ramp_up"].to_numpy(float).max() > 0:
            m.add_constraints(delta <= rup, name="ramp_up")
        if ramp_commit["ramp_dn"].to_numpy(float).max() > 0:
            m.add_constraints(-delta <= rdn, name="ramp_dn")

    # ---- reserves (optional) -------------------------------------------- #
    if cfg.enable_reserves:
        sto_reserve = (storage, dis) if have_sto else None
        _add_reserves(m, zdata, zones, hours, commit, gen_p, sto_reserve)

    # ---- objective ------------------------------------------------------- #
    mc = xr.DataArray(gens["mc"].to_numpy(float), coords={GEN: gen_index}, dims=[GEN])
    obj = (mc * gen_p).sum() \
        + cfg.voll_eur_per_mwh * shed_e.sum() \
        + cfg.dump_penalty_eur_per_mwh * dump_e.sum() \
        + uc_startup_obj + ext_e_obj
    if not cfg.electricity_only:
        obj = obj + cfg.h2_terminal_price * term_h2.sum() \
            + cfg.voll_eur_per_mwh * shed_h.sum() \
            + cfg.dump_penalty_eur_per_mwh * dump_h.sum()
    # A small per-MWh throughput cost on every storage device forbids charging
    # and discharging in the same hour without a binary. It is not needed for
    # lossy devices (round-trip efficiency < 1, e.g. batteries at ~92%): there
    # simultaneous charge+discharge already loses energy, so the LP avoids it on
    # its own. It matters for lossless devices (efficiency = 1, e.g. hydro
    # reservoir / pumped-storage and H2 storage as modelled), where without it
    # the LP could charge and discharge at once (a harmless wash) — the tiny
    # cost cleans that up and keeps the model a pure LP.
    if have_sto:
        obj = obj + cfg.storage_op_cost_eur_per_mwh * (ch.sum() + dis.sum())
    m.add_objective(obj)

    br = BuildResult(m, cfg, zones, hours, gens, commit, storage, gen_upper,
                     demand_e, demand_h, external_e, external_h2, net.elec, net.hydrogen, net,
                     uc_gens=(uc_gens if uc_x_on is not None else None))
    br._ely_eff = pd.Series(ely_eff, index=zones)  # for exact H2-balance validation
    br._ely_cap = pd.Series(ely_cap, index=zones)      # electrolyser power capacity (MW)
    br._term_cap = pd.Series(term_cap, index=zones)    # H2 terminal import capacity (MW, as used)
    return br


def marginal_prices(build: BuildResult):
    """Zonal marginal prices (EUR/MWh) as the duals of the nodal balances.

    The dispatch is a pure LP, so the balance duals come straight from the
    already-solved model — no commitment-fixing re-solve is needed. Returns
    (price_e, price_h) DataArrays over (zone, hour). The dual of
    ``balance == demand`` is d(cost)/d(demand) = the marginal price of supply.
    """
    price_e = build.model.constraints["elec_balance"].dual
    price_h = build.model.constraints["h2_balance"].dual if not build.cfg.electricity_only else None
    return price_e, price_h


def uc_fixed_profile_and_cost(build: BuildResult) -> tuple[dict[str, np.ndarray], float]:
    """From a solved pass-1 MILP build (``build.uc_gens`` non-empty): the
    solved 0/pmax commitment profile per gen (for pass 2's fixed_uc_profile)
    and the total start-up cost actually incurred (sum of
    ``startup_cost_eur * uc_start`` over the solved schedule) -- this is the
    only place that total is computable, since pass 2 has no uc_start
    variable at all (the commitment decision is already fixed by then)."""
    x_on_sol = build.model.solution["uc_on"]
    y_start_sol = build.model.solution["uc_start"]
    fixed_profile: dict[str, np.ndarray] = {}
    total_cost = 0.0
    for gid in build.uc_gens:
        cap = float(build.gens.loc[gid, "pmax"])
        onoff = np.round(x_on_sol.sel({GEN: gid}).to_numpy())
        fixed_profile[gid] = onoff * cap
        starts = y_start_sol.sel({GEN: gid}).to_numpy().sum()
        total_cost += starts * float(build.gens.loc[gid, "startup_cost_eur"])
    return fixed_profile, total_cost


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bc(da_over_sto: xr.DataArray, hours: pd.Index) -> xr.DataArray:
    """Broadcast a per-storage DataArray to (sto, hour)."""
    return da_over_sto.expand_dims({HOUR: hours}).transpose(STO, HOUR)


def _bc_z(values: np.ndarray, zidx: pd.Index, hours: pd.Index) -> xr.DataArray:
    da = xr.DataArray(values, coords={ZONE: zidx}, dims=[ZONE])
    return da.expand_dims({HOUR: hours}).transpose(ZONE, HOUR)


def _profile_da(zdata, zones, hours, col) -> xr.DataArray:
    mat = np.vstack([_num(zdata[z].profiles[col].to_numpy()) if col in zdata[z].profiles
                     else np.zeros(len(hours)) for z in zones])
    return xr.DataArray(mat, coords={ZONE: pd.Index(zones, name=ZONE), HOUR: hours},
                        dims=[ZONE, HOUR])


def _h2_main_zones(zdata, zones) -> dict[str, str]:
    """Main H2 zone per country = the country's selected zone with the most H2 demand."""
    best: dict[str, str] = {}
    dem: dict[str, float] = {}
    for z in zones:
        prof = zdata[z].profiles
        d = float(_num(prof["Hydrogen Demand Profile"].to_numpy()).sum()) \
            if "Hydrogen Demand Profile" in prof else 0.0
        c = z[:2]
        if c not in best or d > dem[c]:
            best[c], dem[c] = z, d
    return best


def _priced_external_elec(m: linopy.Model, zones: list[str], hours: pd.Index, cfg: RunConfig):
    """Priced/controllable import & export legs for every zone's EXTERNAL
    neighbours (any neighbour outside ``zones``): each leg is a decision
    variable capped at that border's REAL physical line capacity
    (Networks.xlsx rating, both directions -- see
    network_loader.border_line_caps), priced at the neighbour's own PLEXOS
    marginal price (0 if the neighbour has no PLEXOS price data). Returns
    (net_injection_expr, objective_cost_expr) to use in place of the fixed
    ``external_e`` term -- see ``cfg.priced_external_elec``.

    Real line capacity was validated (single-zone, all 21 CORE zones)
    against two alternatives: capping at historical realized flow (mean
    corr vs PLEXOS 0.754) and leaving trade uncapped (mean corr 0.909) --
    real line capacity scored highest (mean corr 0.958) and is the
    physically correct choice.
    """
    from . import marginal_price_loader as mpl

    zidx = pd.Index(zones, name=ZONE)
    edf = pd.read_parquet(Path(cfg.exports_dir) / "crossborder_electricity_2030.parquet")
    # elec_border_legs is used only to discover which (zone, neighbour) pairs
    # are real external borders -- the historical flow values themselves are
    # no longer used as the capacity bound (see docstring above).
    legs = exports_loader.elec_border_legs(zones, edf)

    if not legs:
        zero = xr.DataArray(np.zeros((len(zones), len(hours))),
                            coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        return zero, 0.0

    h0, h1 = cfg.hour_slice()
    pairs = sorted(legs)  # [(zone, neighbor), ...], deterministic order
    pname = "extleg"
    pidx = pd.Index([f"{z}|{n}" for z, n in pairs], name=pname)

    line_caps = nl.border_line_caps("electricity", cfg.networks_db)

    def _border_cap(z: str, n: str) -> tuple[float, float]:
        """(import cap z<-n, export cap z->n) MW, from whichever orientation
        the line record was stored in -- 0 if no line exists at all."""
        if (z, n) in line_caps:
            ft, tf = line_caps[(z, n)]
            return tf, ft
        if (n, z) in line_caps:
            ft, tf = line_caps[(n, z)]
            return ft, tf
        return 0.0, 0.0

    imp_vec = np.array([_border_cap(z, n)[0] for z, n in pairs])
    exp_vec = np.array([_border_cap(z, n)[1] for z, n in pairs])
    imp_cap = np.tile(imp_vec[:, None], (1, len(hours)))
    exp_cap = np.tile(exp_vec[:, None], (1, len(hours)))

    neighbors = sorted({n for _, n in pairs})
    ghours = pd.RangeIndex(h0, h1)
    price_df = mpl.load_zone_series(neighbors, ghours, mpl.DEFAULT_MARGINAL_PRICE_ELEC_DB)
    price_mat = np.vstack([price_df[n].to_numpy() for _, n in pairs])  # (npairs, H)

    imp_cap_da = xr.DataArray(imp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    exp_cap_da = xr.DataArray(exp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    price_da = xr.DataArray(price_mat, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])

    imp = m.add_variables(lower=0.0, upper=imp_cap_da, name="ext_imp")
    exp = m.add_variables(lower=0.0, upper=exp_cap_da, name="ext_exp")

    A = np.zeros((len(pairs), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, (z, _n) in enumerate(pairs):
        A[i, zpos[z]] = 1.0
    A_da = xr.DataArray(A, coords={pname: pidx, ZONE: zidx}, dims=[pname, ZONE])

    net_injection = (A_da * imp).sum(pname) - (A_da * exp).sum(pname)
    obj = (price_da * imp).sum() - (price_da * exp).sum()
    return net_injection, obj


# Single-variable renewable techs: one of this model's own generators maps
# 1:1 to one PLEXOS category, so its availability is fully REPLACED by
# PLEXOS's realized generation (not just an upper bound layered on top of
# this model's own capacity x profile calculation -- see
# cfg.cap_renewables_to_plexos).
_RENEWABLE_SINGLE = [
    ("wind_onshore", "Wind (onshore) (MW)"),
    ("wind_offshore", "Wind (offshore) (MW)"),
    ("ror", "Hydro (river) (MW)"),
]
# Joint renewable techs: PLEXOS publishes only ONE aggregate category where
# this model has several of its own generators (no PV/rooftop or
# thermal/thermal+storage split in PLEXOS's output). Each present
# generator's own upper bound is set to the full PLEXOS category total
# (a generous, individually non-binding ceiling), and the real limit is
# enforced afterwards as a joint sum(gen_p) <= PLEXOS constraint.
_RENEWABLE_JOINT = [
    ("solar_pv", ["Solar (MW)", "Solar (rooftop) (MW)"]),
    ("solar_thermal", ["Solar (thermal) (MW)", "Solar (thermal_with_storage) (MW)"]),
    ("other_res", ["Other RES (biomass) (MW)", "Other RES (geothermal) (MW)",
                   "Other RES (marine) (MW)", "Other RES (waste) (MW)", "Other RES (unknown) (MW)"]),
]


def _renewable_plexos_dbs():
    from . import marginal_price_loader as mpl
    return {
        "wind_onshore": mpl.DEFAULT_WIND_ONSHORE_DB, "wind_offshore": mpl.DEFAULT_WIND_OFFSHORE_DB,
        "ror": mpl.DEFAULT_ROR_DB, "solar_pv": mpl.DEFAULT_SOLAR_PV_DB,
        "solar_thermal": mpl.DEFAULT_SOLAR_THERMAL_DB, "other_res": mpl.DEFAULT_OTHER_RES_DB,
    }


def _new_renewable_row(z: str, tech: str, category: str, pmax: float) -> dict:
    """Minimal gens row for a renewable generator created purely from PLEXOS
    data (see _override_renewable_upper_with_plexos) -- same column subset
    _build_generators itself uses for CAT_VRES/CAT_ROR rows; every other
    column (pmin_floor, ramp_up, ...) is implicitly NaN via pd.concat,
    exactly as for a normally-built VRES/ROR row."""
    return dict(gen=f"{z}|{tech}", zone=z, tech=tech, category=category,
               h2_fuel=False, mc=0.0, eff=1.0, pmax=pmax)


def _override_renewable_upper_with_plexos(zdata: dict[str, ZoneData], gens: pd.DataFrame,
                                          gupper: dict[str, np.ndarray], zones: list[str],
                                          cfg: RunConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Replace every renewable generator's hourly availability with PLEXOS's
    own realized generation for that technology, discarding this model's
    own capacity x capacity-factor profile calculation entirely -- see
    cfg.cap_renewables_to_plexos. Also CREATES a generator for any zone
    that has real installed capacity for a technology but was skipped by
    _build_generators because its own profile happened to be all-zero
    (e.g. BEOF/DEKF's offshore wind: real capacity, empty profile data) --
    a zone with genuinely zero capacity is left alone; nothing to source
    generation from either way. Returns the (possibly extended) ``gens``
    and ``gupper``; must run BEFORE gen_p's upper bound DataArray is built.
    """
    from . import marginal_price_loader as mpl
    dbs = _renewable_plexos_dbs()
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    gidx = set(gens.index)
    new_rows: list[dict] = []

    def _cap(z: str, tech: str) -> float:
        return float(zdata[z].capacities.get(tech, 0.0) or 0.0)

    for key, tech in _RENEWABLE_SINGLE:
        category = dl.CAT_ROR if key == "ror" else dl.CAT_VRES
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            gid = f"{z}|{tech}"
            vals = px[z].to_numpy()
            if gid in gidx:
                gupper[gid] = vals
            elif _cap(z, tech) > 0:
                new_rows.append(_new_renewable_row(z, tech, category, float(vals.max())))
                gupper[gid] = vals
                gidx.add(gid)

    for key, techs in _RENEWABLE_JOINT:
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            vals = px[z].to_numpy()
            for t in techs:
                gid = f"{z}|{t}"
                if gid in gidx:
                    gupper[gid] = vals
                elif _cap(z, t) > 0:
                    new_rows.append(_new_renewable_row(z, t, dl.CAT_VRES, float(vals.max())))
                    gupper[gid] = vals
                    gidx.add(gid)

    if new_rows:
        gens = pd.concat([gens, pd.DataFrame(new_rows).set_index("gen")])
    return gens, gupper


def _joint_renewable_constraints(m: linopy.Model, gens: pd.DataFrame, gen_p, zones: list[str],
                                 hours: pd.Index, cfg: RunConfig) -> None:
    """For renewable techs where PLEXOS publishes one aggregate category
    covering several of this model's own generators (see
    _RENEWABLE_JOINT), add the real limit: sum(gen_p over the group) <=
    PLEXOS's realized total for that category. Each individual generator's
    own upper bound was already set to the (generous) full category total
    by _override_renewable_upper_with_plexos, so this is what actually
    enforces the combined ceiling."""
    from . import marginal_price_loader as mpl
    dbs = _renewable_plexos_dbs()
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    gidx = set(gens.index)

    for key, techs in _RENEWABLE_JOINT:
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            present = [f"{z}|{t}" for t in techs if f"{z}|{t}" in gidx]
            if not present:
                continue
            cap_da = xr.DataArray(px[z].to_numpy(), coords={HOUR: hours}, dims=[HOUR])
            expr = sum(gen_p.sel({GEN: gid}) for gid in present)
            m.add_constraints(expr <= cap_da, name=f"plexos_cap_{key}_{z}")


def _external_exchange_all(zdata, zones, hours, cfg):
    """Return (external_e, external_h2) net-injection arrays (import +).

    Computed from the ``inputs/`` result databases so neighbours track the zone
    selection (see exports_loader / inputs/EXPORTS_CALCULATION.md).
    """
    main_map = _h2_main_zones(zdata, zones)
    return exports_loader.load_external_injection(cfg, zones, hours, main_map)


def _flow_terms(m: linopy.Model, lines: list[Line], zones: list[str], hours: pd.Index, tag: str):
    """Create directional flow vars and return the per-zone net-import expression."""
    if not lines:
        return 0.0, None, None
    lidx = pd.Index([f"{tag}{i}:{l.frm}->{l.to}" for i, l in enumerate(lines)], name=f"line_{tag}")
    cap_ft = xr.DataArray([l.cap_ft for l in lines], coords={lidx.name: lidx}, dims=[lidx.name])
    cap_tf = xr.DataArray([l.cap_tf for l in lines], coords={lidx.name: lidx}, dims=[lidx.name])
    loss = np.array([l.loss for l in lines])

    fpos = m.add_variables(lower=0.0, upper=_bc_line(cap_ft, hours), name=f"f{tag}_pos")
    fneg = m.add_variables(lower=0.0, upper=_bc_line(cap_tf, hours), name=f"f{tag}_neg")

    Cfrom = np.zeros((len(lines), len(zones)))
    Cto = np.zeros((len(lines), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, l in enumerate(lines):
        Cfrom[i, zpos[l.frm]] = 1.0
        Cto[i, zpos[l.to]] = 1.0
    dim = lidx.name
    Cfrom = xr.DataArray(Cfrom, coords={dim: lidx, ZONE: zones}, dims=[dim, ZONE])
    Cto = xr.DataArray(Cto, coords={dim: lidx, ZONE: zones}, dims=[dim, ZONE])
    lloss = xr.DataArray(loss, coords={dim: lidx}, dims=[dim])

    coeff_pos = Cto * (1 - lloss) - Cfrom
    coeff_neg = Cfrom * (1 - lloss) - Cto
    net_import = (coeff_pos * fpos).sum(dim) + (coeff_neg * fneg).sum(dim)
    return net_import, fpos, fneg


def _bc_line(da: xr.DataArray, hours: pd.Index) -> xr.DataArray:
    dim = da.dims[0]
    return da.expand_dims({HOUR: hours}).transpose(dim, HOUR)


def _add_reserves(m, zdata, zones, hours, commit, gen_p, sto_reserve=None):
    """FCR+FRR: spare headroom of thermal fleets + electricity storage >= requirement.

    PLEXOS draws its reserve pool from every available resource, not just
    thermal plant: battery and hydro (reservoir/pumped-storage) discharge
    headroom count towards FCR/FRR alongside spare thermal capacity. With no
    commitment binary, the whole thermal fleet capacity is available for
    reserve (headroom = pmax(fleet) - gen_p); each electricity storage device
    contributes its spare discharge headroom (Pdis - dis).
    """
    cidx = commit.index
    # With no commitment binary, the whole fleet capacity is available for
    # reserve: headroom = fleet capacity - output = pmax(fleet) - gen_p.
    capacity = xr.DataArray(commit["pmax"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
    headroom = capacity - gen_p.sel({GEN: cidx})
    A = _incidence(commit["zone"], zones, GEN)
    head_by_zone = (A * headroom).sum(GEN)

    if sto_reserve is not None:
        storage, dis = sto_reserve
        e_sto = storage[storage["carrier"] == "electricity"]
        if len(e_sto) > 0:
            sidx = e_sto.index
            pdis = xr.DataArray(e_sto["pdis"].to_numpy(float), coords={STO: sidx}, dims=[STO])
            sto_headroom = pdis - dis.sel({STO: sidx})
            A_sto = _incidence(e_sto["zone"], zones, STO)
            head_by_zone = head_by_zone + (A_sto * sto_headroom).sum(STO)

    req = []
    for z in zones:
        r = zdata[z].reserves
        req.append(r.get("Total (FCR) (MW/h)", 0.0) + r.get("Total (FRR) (MW/h)", 0.0))
    req_da = xr.DataArray(np.array(req), coords={ZONE: pd.Index(zones, name=ZONE)}, dims=[ZONE])
    m.add_constraints(head_by_zone >= req_da, name="reserves")
