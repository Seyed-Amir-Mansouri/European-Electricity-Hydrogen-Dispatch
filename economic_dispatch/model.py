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

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import RunConfig
from . import data_loader as dl
from . import exports_loader
from .data_loader import ZoneData
from .network_loader import NetworkData, Line

HOUR = "hour"
GEN = "gen"
ZONE = "zone"
STO = "sto"


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
                mustrun = zd.must_run_units(tech, month)
                mustrun = float(min(max(mustrun, 0.0), max_units))
                # Pure-LP dispatch (no commitment binary): instead of
                # n*Pmin <= p <= n*Pmax with integer n, the fleet output floats
                # between a fixed floor and its capacity. A tech with a must-run
                # requirement keeps ``mustrun`` units online at their minimum
                # stable power, so its floor is max(min-stable, must-run) held
                # online = mustrun * pmin_unit; a tech with no must-run can idle
                # at 0. (mustrun is a monthly unit count from the data.)
                pmin_floor = mustrun * pmin_unit
                # Missing/zero efficiency -> use the default (avoids a 1/eff = 1e6
                # coefficient in the H2 balance that ruins the LP conditioning).
                eff = zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0
                eff = eff if eff > 1e-3 else cfg.default_efficiency
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=h2_fuel,
                    mc=_marginal_cost(zd, tech, h2_fuel, net.co2_price, cfg),
                    eff=eff,
                    units=max_units, pmin_unit=pmin_unit, pmax_unit=pmax_unit,
                    ramp_up=ramp_pu * units * cfg.ramp_scale,
                    ramp_dn=ramp_dn * units * cfg.ramp_scale,
                    mustrun=mustrun, pmin_floor=pmin_floor, pmax=cap,
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
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=False,
                    mc=_marginal_cost(zd, tech, False, net.co2_price, cfg),
                    eff=1.0, pmax=float(avail.max())))
                upper[gid] = avail

    gens = pd.DataFrame(rows).set_index("gen")
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
             cfg.default_pump_efficiency, "electricity"),
        ]
        if cfg.enable_h2_storage:
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


def build_model(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig,
                soc_init: pd.Series | None = None, cyclic: bool | None = None) -> BuildResult:
    zones = cfg.zones
    H = len(zdata[zones[0]].profiles)
    hours = pd.Index(range(H), name=HOUR)
    zidx = pd.Index(zones, name=ZONE)

    gens, gupper = _build_generators(zdata, net, cfg)
    storage, sinflow = _build_storage(zdata, cfg)

    m = linopy.Model()

    # ---- generation (pure LP, no commitment binary) ---------------------- #
    # Each fleet's output floats between a fixed floor and its available
    # capacity. The floor is the must-run minimum (mustrun units held at their
    # minimum stable power); resources with no must-run requirement have a zero
    # floor, so there is no need for an integer on/off variable.
    gen_index = gens.index
    upper_mat = np.vstack([gupper[g] for g in gen_index])
    gen_upper = xr.DataArray(upper_mat, coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR])
    floor_vec = np.nan_to_num(gens["pmin_floor"].to_numpy(float)) \
        if "pmin_floor" in gens.columns else np.zeros(len(gen_index))
    gen_lower = xr.DataArray(floor_vec, coords={GEN: gen_index}, dims=[GEN])
    gen_p = m.add_variables(lower=gen_lower, upper=gen_upper, name="gen_p")

    A_gen = _incidence(gens["zone"], zones, GEN)
    gen_by_zone = (A_gen * gen_p).sum(GEN)

    commit = gens[gens["category"] == dl.CAT_COMMIT].copy()
    cidx = commit.index

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

        default_init = cfg.initial_soc_fraction * storage["ecap"].to_numpy(float)
        if soc_init is None:
            soc0 = default_init
        else:  # rolling horizon: carry the previous block's end-of-block SoC
            soc0 = pd.Series(soc_init).reindex(sidx).fillna(
                pd.Series(default_init, index=sidx)).to_numpy(float)
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
            end = xr.DataArray(soc0, coords={STO: sidx}, dims=[STO])
            m.add_constraints(soc.sel({HOUR: H - 1}) == end, name="soc_cyclic")
    else:
        dis_by_zone = ch_by_zone = 0.0
        dis_h2_by_zone = ch_h2_by_zone = 0.0

    # ---- electrolysers & H2 terminals ----------------------------------- #
    ely_cap = np.array([zdata[z].capacities.get("Electrolyser (MW)", 0.0) for z in zones])
    ely_eff = np.array([max(zdata[z].char_val("Electrolyser (MW)", "Efficiency (%)", 68.0) / 100.0, 1e-6)
                        for z in zones])
    ely_p = m.add_variables(lower=0.0, upper=_bc_z(ely_cap, zidx, hours), name="ely_p")
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

    # ---- network flows (split directional, loss on receiving end) -------- #
    net_e, fe_pos, fe_neg = _flow_terms(m, net.elec, zones, hours, "e")
    net_h, fh_pos, fh_neg = _flow_terms(m, net.hydrogen, zones, hours, "h")

    # ---- demand / fixed exchange ---------------------------------------- #
    demand_e = _profile_da(zdata, zones, hours, "Electricity Demand Profile")
    demand_h = _profile_da(zdata, zones, hours, "Hydrogen Demand Profile")
    external_e, external_h2 = _external_exchange_all(zdata, zones, hours, cfg)

    shed_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_e")
    shed_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_h")
    # Dump/curtailment slacks absorb EXCESS supply (e.g. a fixed net import that
    # exceeds absorbable load) — the counterpart of shedding, as in PLEXOS's
    # "Dumped" category. Without them the equality balance can be infeasible.
    dump_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_e")
    dump_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_h")

    # ---- balances -------------------------------------------------------- #
    # external_e / external_h2 are net imports (import +), so they add to supply.
    elec_lhs = (gen_by_zone + dis_by_zone - ch_by_zone - ely_p + net_e
                + external_e + shed_e - dump_e)
    m.add_constraints(elec_lhs == demand_e, name="elec_balance")

    h2_lhs = (ely_eff_da * ely_p + term_h2 + net_h + external_h2
              + dis_h2_by_zone - ch_h2_by_zone + shed_h
              - h2_cons_by_zone - dump_h)
    m.add_constraints(h2_lhs == demand_h, name="h2_balance")

    # ---- ramps ----------------------------------------------------------- #
    if cfg.enable_ramps and len(commit) > 0:
        rup = xr.DataArray(commit["ramp_up"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
        rdn = xr.DataArray(commit["ramp_dn"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
        gp_c = gen_p.sel({GEN: cidx})
        # delta[h] = gen[h] - gen[h-1] for h >= 1 (drop hour 0: no predecessor).
        delta = (gp_c - gp_c.shift({HOUR: 1})).isel({HOUR: slice(1, None)})
        if commit["ramp_up"].to_numpy(float).max() > 0:
            m.add_constraints(delta <= rup, name="ramp_up")
        if commit["ramp_dn"].to_numpy(float).max() > 0:
            m.add_constraints(-delta <= rdn, name="ramp_dn")

    # ---- reserves (optional) -------------------------------------------- #
    if cfg.enable_reserves:
        _add_reserves(m, zdata, zones, hours, commit, gen_p)

    # ---- objective ------------------------------------------------------- #
    mc = xr.DataArray(gens["mc"].to_numpy(float), coords={GEN: gen_index}, dims=[GEN])
    obj = (mc * gen_p).sum() + cfg.h2_terminal_price * term_h2.sum() \
        + cfg.voll_eur_per_mwh * (shed_e.sum() + shed_h.sum()) \
        + cfg.dump_penalty_eur_per_mwh * (dump_e.sum() + dump_h.sum())
    # A small per-MWh throughput cost on every storage device makes charging and
    # discharging in the same hour strictly wasteful, so the LP never does both
    # at once — this replaces the binary that would otherwise enforce mutual
    # exclusion, keeping the whole model a pure LP.
    if have_sto:
        obj = obj + cfg.storage_op_cost_eur_per_mwh * (ch.sum() + dis.sum())
    m.add_objective(obj)

    br = BuildResult(m, cfg, zones, hours, gens, commit, storage, gen_upper,
                     demand_e, demand_h, external_e, external_h2, net.elec, net.hydrogen, net)
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
    price_h = build.model.constraints["h2_balance"].dual
    return price_e, price_h


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


def _add_reserves(m, zdata, zones, hours, commit, gen_p):
    """FCR+FRR: spare headroom of thermal fleets >= total requirement per zone."""
    cidx = commit.index
    # With no commitment binary, the whole fleet capacity is available for
    # reserve: headroom = fleet capacity - output = pmax(fleet) - gen_p.
    capacity = xr.DataArray(commit["pmax"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
    headroom = capacity - gen_p.sel({GEN: cidx})
    A = _incidence(commit["zone"], zones, GEN)
    head_by_zone = (A * headroom).sum(GEN)
    req = []
    for z in zones:
        r = zdata[z].reserves
        req.append(r.get("Total (FCR) (MW/h)", 0.0) + r.get("Total (FRR) (MW/h)", 0.0))
    req_da = xr.DataArray(np.array(req), coords={ZONE: pd.Index(zones, name=ZONE)}, dims=[ZONE])
    m.add_constraints(head_by_zone >= req_da, name="reserves")
