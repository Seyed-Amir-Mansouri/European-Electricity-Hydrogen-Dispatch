"""Build the coupled electricity + hydrogen dispatch MILP with linopy.

Design
------
* One global index of "generators" (``gen``) spanning every supply resource
  across all zones, plus indices for storage, electrolysers, and network lines.
* Zone balances are assembled with incidence DataArrays (``A[gen, zone]``) via
  ``(A * var).sum("gen")`` — fully vectorised over zones and hours.
* Inherently sequential constraints (ramps, storage state-of-charge) loop over
  the 24 hours; everything else is vectorised.
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
    commit: pd.DataFrame        # subset of gens that are unit-committed
    storage: pd.DataFrame       # indexed by sto_id
    gen_upper: xr.DataArray     # (gen, hour) available capacity
    demand_e: xr.DataArray      # (zone, hour)
    demand_h: xr.DataArray
    external_e: xr.DataArray
    elines: list[Line]
    hlines: list[Line]
    net: NetworkData


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
    if cfg.fuel_per_thermal:
        e = eff if eff > 0 else cfg.default_efficiency
        return vom + fuel / e + (co2f / e) * co2_price
    return vom + fuel + co2f * co2_price


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
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=h2_fuel,
                    mc=_marginal_cost(zd, tech, h2_fuel, net.co2_price, cfg),
                    eff=max(zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0, 1e-6),
                    units=max_units, pmin_unit=pmin_unit, pmax_unit=pmax_unit,
                    ramp_up=ramp_pu * units * cfg.ramp_scale,
                    ramp_dn=ramp_dn * units * cfg.ramp_scale,
                    mustrun=mustrun, pmax=cap,
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
            # kind, pdis, pchg, ecap, inflow, eff
            ("Battery",
             zd.char_val("Battery (MWh)", "Net maximum capacity - generation perspective (MW)"),
             zd.char_val("Battery (MWh)", "Net maximum capacity - demand perspective (MW)"),
             e.get("Battery (MWh)", 0.0), zero,
             max(zd.char_val("Battery (MWh)", "Efficiency (%)", 92.0) / 100.0, 0.1)),
            ("Hydro reservoir", cap.get("Hydro (reservoir) (MW)", 0.0), 0.0,
             e.get("Hydro (reservoir) (MWh)", 0.0), col("Reservoir Flow Energy"),
             cfg.default_hydro_efficiency),
            ("Hydro pondage", cap.get("Hydro (pondage) (MW)", 0.0), 0.0,
             e.get("Hydro (pondage) (MWh)", 0.0), col("Pondage Flow Energy"),
             cfg.default_hydro_efficiency),
            ("Hydro open_ps", cap.get("Hydro (open_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (open_ps_pump) (MW)", 0.0)),
             e.get("Hydro (open_ps) (MWh)", 0.0), col("Open_PS Flow Energy"),
             cfg.default_pump_efficiency),
            ("Hydro closed_ps", cap.get("Hydro (closed_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (closed_ps_pump) (MW)", 0.0)),
             e.get("Hydro (closed_ps) (MWh)", 0.0), col("Closed_PS Flow Energy"),
             cfg.default_pump_efficiency),
        ]
        for kind, pdis, pchg, ecap, inf, eff in specs:
            if ecap <= 0 or pdis <= 0:
                continue
            sid = f"{z}|{kind}"
            rows.append(dict(sto=sid, zone=z, kind=kind, pdis=float(pdis),
                             pchg=float(pchg), ecap=float(ecap), eff=float(eff)))
            inflow[sid] = inf

    storage = pd.DataFrame(rows).set_index("sto") if rows else pd.DataFrame(
        columns=["zone", "kind", "pdis", "pchg", "ecap", "eff"]).rename_axis("sto")
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


def build_model(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig) -> BuildResult:
    zones = cfg.zones
    H = len(zdata[zones[0]].profiles)
    hours = pd.Index(range(H), name=HOUR)
    zidx = pd.Index(zones, name=ZONE)

    gens, gupper = _build_generators(zdata, net, cfg)
    storage, sinflow = _build_storage(zdata, cfg)

    m = linopy.Model()

    # ---- generation ------------------------------------------------------ #
    gen_index = gens.index
    upper_mat = np.vstack([gupper[g] for g in gen_index])
    gen_upper = xr.DataArray(upper_mat, coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR])
    gen_p = m.add_variables(lower=0.0, upper=gen_upper, name="gen_p")

    A_gen = _incidence(gens["zone"], zones, GEN)
    gen_by_zone = (A_gen * gen_p).sum(GEN)

    # ---- unit commitment (integer) -------------------------------------- #
    commit = gens[gens["category"] == dl.CAT_COMMIT].copy()
    cidx = commit.index
    n_upper = xr.DataArray(commit["units"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
    n = m.add_variables(lower=0.0, upper=n_upper, integer=True, coords=[cidx], name="n_units")

    pmin = xr.DataArray(commit["pmin_unit"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
    pmax = xr.DataArray(commit["pmax_unit"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
    gp_c = gen_p.sel({GEN: cidx})
    m.add_constraints(gp_c - n * pmax <= 0, name="commit_max")
    m.add_constraints(gp_c - n * pmin >= 0, name="commit_min")
    mustrun = commit["mustrun"].to_numpy(float)
    if mustrun.max() > 0:
        mr = xr.DataArray(mustrun, coords={GEN: cidx}, dims=[GEN])
        m.add_constraints(n >= mr, name="must_run")

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
        dis_by_zone = (A_sto * dis).sum(STO)
        ch_by_zone = (A_sto * ch).sum(STO)

        soc_init = cfg.initial_soc_fraction * storage["ecap"].to_numpy(float)
        inflow_mat = np.vstack([sinflow[s] for s in sidx])
        for h in range(H):
            inf_h = xr.DataArray(inflow_mat[:, h], coords={STO: sidx}, dims=[STO])
            eff_da = xr.DataArray(eff, coords={STO: sidx}, dims=[STO])
            prev = soc.sel({HOUR: h - 1}) if h > 0 else \
                xr.DataArray(soc_init, coords={STO: sidx}, dims=[STO])
            lhs = soc.sel({HOUR: h}) - eff_da * ch.sel({HOUR: h}) + dis.sel({HOUR: h}) \
                + spill.sel({HOUR: h})
            rhs = prev + inf_h
            m.add_constraints(lhs == rhs, name=f"soc_balance_{h}")
        if cfg.cyclic_storage:
            end = xr.DataArray(soc_init, coords={STO: sidx}, dims=[STO])
            m.add_constraints(soc.sel({HOUR: H - 1}) == end, name="soc_cyclic")
    else:
        dis_by_zone = ch_by_zone = 0.0

    # ---- electrolysers & H2 terminals ----------------------------------- #
    ely_cap = np.array([zdata[z].capacities.get("Electrolyser (MW)", 0.0) for z in zones])
    ely_eff = np.array([max(zdata[z].char_val("Electrolyser (MW)", "Efficiency (%)", 68.0) / 100.0, 1e-6)
                        for z in zones])
    ely_p = m.add_variables(lower=0.0, upper=_bc_z(ely_cap, zidx, hours), name="ely_p")
    ely_eff_da = xr.DataArray(ely_eff, coords={ZONE: zidx}, dims=[ZONE])

    if cfg.enable_h2_terminal:
        term_cap = np.array([zdata[z].gas_h2.get("Terminal (Hydrogen) (MW)", 0.0) for z in zones])
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
    external_e = _external_exchange(zdata, zones, hours)

    shed_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_e")
    shed_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_h")

    # ---- balances -------------------------------------------------------- #
    elec_lhs = gen_by_zone + dis_by_zone - ch_by_zone - ely_p + net_e + shed_e
    m.add_constraints(elec_lhs == demand_e - external_e, name="elec_balance")

    h2_lhs = ely_eff_da * ely_p + term_h2 + net_h + shed_h - h2_cons_by_zone
    m.add_constraints(h2_lhs == demand_h, name="h2_balance")

    # ---- ramps ----------------------------------------------------------- #
    if cfg.enable_ramps and len(commit) > 0:
        rup = xr.DataArray(commit["ramp_up"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
        rdn = xr.DataArray(commit["ramp_dn"].to_numpy(float), coords={GEN: cidx}, dims=[GEN])
        active_up = commit["ramp_up"].to_numpy(float).max() > 0
        active_dn = commit["ramp_dn"].to_numpy(float).max() > 0
        for h in range(1, H):
            delta = gen_p.sel({GEN: cidx, HOUR: h}) - gen_p.sel({GEN: cidx, HOUR: h - 1})
            if active_up:
                m.add_constraints(delta <= rup, name=f"ramp_up_{h}")
            if active_dn:
                m.add_constraints(-delta <= rdn, name=f"ramp_dn_{h}")

    # ---- reserves (optional) -------------------------------------------- #
    if cfg.enable_reserves:
        _add_reserves(m, zdata, zones, hours, gens, commit, gen_p, n, pmax)

    # ---- objective ------------------------------------------------------- #
    mc = xr.DataArray(gens["mc"].to_numpy(float), coords={GEN: gen_index}, dims=[GEN])
    obj = (mc * gen_p).sum() + cfg.h2_terminal_price * term_h2.sum() \
        + cfg.voll_eur_per_mwh * (shed_e.sum() + shed_h.sum())
    m.add_objective(obj)

    br = BuildResult(m, cfg, zones, hours, gens, commit, storage, gen_upper,
                     demand_e, demand_h, external_e, net.elec, net.hydrogen, net)
    br._ely_eff = pd.Series(ely_eff, index=zones)  # for exact H2-balance validation
    br._ely_cap = pd.Series(ely_cap, index=zones)      # electrolyser power capacity (MW)
    br._term_cap = pd.Series(term_cap, index=zones)    # H2 terminal import capacity (MW, as used)
    return br


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


def _external_exchange(zdata, zones, hours) -> xr.DataArray:
    """Net fixed exchange with non-modelled neighbours (native sign: <0 = export)."""
    rows = []
    for z in zones:
        prof = zdata[z].profiles
        cols = [c for c in prof.columns if isinstance(c, str) and c.startswith("Exports")]
        s = np.zeros(len(hours))
        for c in cols:
            s = s + _num(prof[c].to_numpy())
        rows.append(s)
    return xr.DataArray(np.vstack(rows), coords={ZONE: pd.Index(zones, name=ZONE), HOUR: hours},
                        dims=[ZONE, HOUR])


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


def _add_reserves(m, zdata, zones, hours, gens, commit, gen_p, n, pmax):
    """FCR+FRR: committed headroom of thermal units >= total requirement per zone."""
    cidx = commit.index
    # headroom = committed capacity - output = n*pmax_unit - gen_p (per committable gen)
    headroom = n * pmax - gen_p.sel({GEN: cidx})
    A = _incidence(commit["zone"], zones, GEN)
    head_by_zone = (A * headroom).sum(GEN)
    req = []
    for z in zones:
        r = zdata[z].reserves
        req.append(r.get("Total (FCR) (MW/h)", 0.0) + r.get("Total (FRR) (MW/h)", 0.0))
    req_da = xr.DataArray(np.array(req), coords={ZONE: pd.Index(zones, name=ZONE)}, dims=[ZONE])
    m.add_constraints(head_by_zone >= req_da, name="reserves")
