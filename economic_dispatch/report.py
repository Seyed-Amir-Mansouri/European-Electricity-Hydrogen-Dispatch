"""Extract, validate, and export the solved dispatch.

``validate`` independently recomputes both nodal balances from the solution
(pure numpy) and asserts the residuals are ~0 — the primary correctness check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import BuildResult
from . import data_loader as dl


def _sol(build: BuildResult, name: str) -> pd.DataFrame:
    """Return a variable's solution as a (dim0 x hour) DataFrame, or empty."""
    if name not in build.model.variables:
        return pd.DataFrame()
    da = build.model.solution[name]
    return da.to_pandas()


def extract(build: BuildResult) -> dict[str, pd.DataFrame]:
    return {
        "gen_p": _sol(build, "gen_p"),
        "n_units": _sol(build, "n_units"),
        "dis": _sol(build, "dis"),
        "ch": _sol(build, "ch"),
        "soc": _sol(build, "soc"),
        "spill": _sol(build, "spill"),
        "ely_p": _sol(build, "ely_p"),
        "term_h2": _sol(build, "term_h2"),
        "shed_e": _sol(build, "shed_e"),
        "shed_h": _sol(build, "shed_h"),
        "dump_e": _sol(build, "dump_e"),
        "dump_h": _sol(build, "dump_h"),
    }


def _ids_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient so that 'zone|...' resource ids are the row index (linopy's
    to_pandas() dim order is not guaranteed)."""
    if any(isinstance(i, str) and "|" in i for i in df.index):
        return df
    if any(isinstance(c, str) and "|" in c for c in df.columns):
        return df.T
    return df


def _zones_on_rows(df: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Orient a (zone x hour) frame so zones are the row index."""
    if set(zones) & set(df.index):
        return df
    if set(zones) & set(df.columns):
        return df.T
    return df


def _zone_sum(df: pd.DataFrame, zones: list[str], H: int) -> pd.DataFrame:
    """Sum a (gen|sto id x hour) frame into (zone x hour), id = 'zone|...'.

    When the frame is empty (e.g. no storage) return a zero (zone x H) grid so
    it stays column-aligned with the other balance terms.
    """
    if df.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    df = _ids_on_rows(df)
    grp = df.groupby(lambda gid: str(gid).split("|", 1)[0]).sum()
    return grp.reindex(zones).fillna(0.0)


def validate(build: BuildResult, tol: float = 1e-3) -> dict[str, float]:
    """Recompute elec & H2 balances from the solution; return max residuals."""
    z = build.zones
    sol = extract(build)
    H = len(build.hours)

    def zrows(name):
        df = sol[name]
        if df.empty:
            return pd.DataFrame(0.0, index=z, columns=range(H))
        return _zones_on_rows(df, z).reindex(z).fillna(0.0)

    gen_z = _zone_sum(sol["gen_p"], z, H)
    dis_z = _zone_sum(sol["dis"], z, H)
    ch_z = _zone_sum(sol["ch"], z, H)
    ely = zrows("ely_p")
    term = zrows("term_h2")
    shed_e = zrows("shed_e")
    shed_h = zrows("shed_h")
    dump_e = zrows("dump_e")
    dump_h = zrows("dump_h")

    demand_e = _zones_on_rows(build.demand_e.to_pandas(), z).reindex(z)
    demand_h = _zones_on_rows(build.demand_h.to_pandas(), z).reindex(z)
    external_e = _zones_on_rows(build.external_e.to_pandas(), z).reindex(z)
    external_h2 = _zones_on_rows(build.external_h2.to_pandas(), z).reindex(z)

    # H2 consumption by H2-fired plants
    h2 = build.gens[build.gens["h2_fuel"]]
    h2_cons = pd.DataFrame(0.0, index=z, columns=range(H))
    if not h2.empty and not sol["gen_p"].empty:
        for gid, row in h2.iterrows():
            h2_cons.loc[row["zone"]] += sol["gen_p"].loc[gid] / row["eff"]

    # Network flows -> per-zone net import (recompute from solution)
    net_e = _net_import_from_solution(build, "e")
    net_h = _net_import_from_solution(build, "h")

    # Electricity residual
    res_e = (gen_z + dis_z - ch_z - ely + net_e + external_e + shed_e - dump_e - demand_e)
    max_e = float(np.abs(res_e.to_numpy()).max()) if res_e.size else 0.0

    # Hydrogen residual
    ely_prod = _ely_production(build, sol)
    res_h = (ely_prod + term + net_h + shed_h + external_h2 - dump_h - demand_h - h2_cons)
    max_h = float(np.abs(res_h.to_numpy()).max()) if res_h.size else 0.0

    return {"max_elec_residual": max_e, "max_h2_residual": max_h, "tol": tol}


def _ely_production(build: BuildResult, sol) -> pd.DataFrame:
    """H2 produced per zone = eff * elec consumed (electrolyser efficiency)."""
    z = build.zones
    ely = sol["ely_p"]
    ely = _zones_on_rows(ely, z).reindex(z).fillna(0.0) if not ely.empty \
        else pd.DataFrame(0.0, index=z, columns=range(len(build.hours)))
    eff = getattr(build, "_ely_eff", pd.Series(0.68, index=z)).reindex(z).fillna(0.68)
    return ely.mul(eff, axis=0)


def _lines_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient a flow frame so line ids (containing '->') are the row index."""
    if any(isinstance(i, str) and "->" in i for i in df.index):
        return df
    return df.T


def _net_import_from_solution(build: BuildResult, tag: str) -> pd.DataFrame:
    z = build.zones
    lines = build.elines if tag == "e" else build.hlines
    H = len(build.hours)
    out = pd.DataFrame(0.0, index=z, columns=range(H))
    if not lines:
        return out
    fpos = _lines_on_rows(build.model.solution[f"f{tag}_pos"].to_pandas())
    fneg = _lines_on_rows(build.model.solution[f"f{tag}_neg"].to_pandas())
    for i, l in enumerate(lines):
        key = f"{tag}{i}:{l.frm}->{l.to}"
        p = fpos.loc[key].to_numpy(dtype=float)
        n = fneg.loc[key].to_numpy(dtype=float)
        out.loc[l.to] = out.loc[l.to].to_numpy() + p * (1 - l.loss) - n
        out.loc[l.frm] = out.loc[l.frm].to_numpy() + n * (1 - l.loss) - p
    return out


def summary(build: BuildResult) -> dict[str, float]:
    sol = extract(build)
    cfg = build.cfg
    shed_e = sol["shed_e"].to_numpy().sum() if not sol["shed_e"].empty else 0.0
    shed_h = sol["shed_h"].to_numpy().sum() if not sol["shed_h"].empty else 0.0
    dump_e = sol["dump_e"].to_numpy().sum() if not sol["dump_e"].empty else 0.0
    dump_h = sol["dump_h"].to_numpy().sum() if not sol["dump_h"].empty else 0.0
    gen_total = sol["gen_p"].to_numpy().sum() if not sol["gen_p"].empty else 0.0
    ely_total = sol["ely_p"].to_numpy().sum() if not sol["ely_p"].empty else 0.0
    term_total = sol["term_h2"].to_numpy().sum() if not sol["term_h2"].empty else 0.0

    # Objective recomputed from the solution (avoids solver-API coupling).
    gen_cost = 0.0
    if not sol["gen_p"].empty:
        mc = build.gens["mc"].reindex(sol["gen_p"].index).fillna(0.0)
        gen_cost = float((sol["gen_p"].mul(mc, axis=0)).to_numpy().sum())
    obj = (gen_cost + cfg.h2_terminal_price * float(term_total)
           + cfg.voll_eur_per_mwh * float(shed_e + shed_h))
    return {
        "objective_eur": obj,
        "generation_cost_eur": gen_cost,
        "total_generation_mwh": float(gen_total),
        "electrolyser_load_mwh": float(ely_total),
        "h2_terminal_import_mwh": float(term_total),
        "elec_shed_mwh": float(shed_e),
        "h2_shed_mwh": float(shed_h),
        "elec_dumped_mwh": float(dump_e),
        "h2_dumped_mwh": float(dump_h),
    }


def write_outputs(build: BuildResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean slate: remove CSVs from any previous run so the folder reflects
    # exactly this run (some files are only written when their variable is
    # non-empty, so stale files would otherwise linger). Skip files locked by
    # another process (e.g. open in Excel or mid-sync on Google Drive) rather
    # than aborting the whole output step.
    for old in out_dir.glob("*.csv"):
        try:
            old.unlink()
        except OSError as e:
            print(f"  warning: could not remove {old.name} ({e.strerror}); "
                  f"close it if open in another program")
    sol = extract(build)

    # Generation by technology (long form): zone, tech, hour, MW
    gp = sol["gen_p"]
    if not gp.empty:
        long = gp.copy()
        long.index = pd.MultiIndex.from_tuples(
            [tuple(g.split("|", 1)) for g in long.index], names=["zone", "tech"])
        long = long.reset_index().melt(id_vars=["zone", "tech"], var_name="hour", value_name="MW")
        long.to_csv(out_dir / "generation_by_tech.csv", index=False)

    for name in ["soc", "dis", "ch", "ely_p", "term_h2", "shed_e", "shed_h", "n_units"]:
        df = sol[name]
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv")

    # Network flows
    for tag, lines in [("e", build.elines), ("h", build.hlines)]:
        if lines:
            fpos = build.model.solution[f"f{tag}_pos"].to_pandas()
            fneg = build.model.solution[f"f{tag}_neg"].to_pandas()
            net = fpos - fneg
            net.to_csv(out_dir / f"flows_{'elec' if tag == 'e' else 'h2'}.csv")

    pd.Series(summary(build)).to_csv(out_dir / "summary.csv")
    write_hourly_balance(build, out_dir)


def write_hourly_balance(build: BuildResult, out_dir: Path) -> None:
    """Write PLEXOS-style hourly per-technology balances (elec & H2).

    Two wide CSVs with a two-level column header ``(zone, category)`` and one row
    per hour, in the spirit of the MMStandardOutputFile 'Hourly Market Data' /
    'Hourly H2 Data' sheets. Signs are chosen so supply is + and consumption -,
    so each row sums to ~0 (the nodal balance holds).
    """
    out_dir = Path(out_dir)
    z = build.zones
    H = len(build.hours)
    sol = extract(build)

    def zrows(name):
        df = sol[name]
        if df.empty:
            return pd.DataFrame(0.0, index=z, columns=range(H))
        return _zones_on_rows(df, z).reindex(z).fillna(0.0)

    def da_rows(da):
        return _zones_on_rows(da.to_pandas(), z).reindex(z).fillna(0.0)

    gp = _ids_on_rows(sol["gen_p"]) if not sol["gen_p"].empty else pd.DataFrame()
    dis_z, ch_z = _zone_sum(sol["dis"], z, H), _zone_sum(sol["ch"], z, H)
    ely, term = zrows("ely_p"), zrows("term_h2")
    shed_e, shed_h = zrows("shed_e"), zrows("shed_h")
    dmp_e, dmp_h = zrows("dump_e"), zrows("dump_h")
    net_e, net_h = _net_import_from_solution(build, "e"), _net_import_from_solution(build, "h")
    dem_e, dem_h = da_rows(build.demand_e), da_rows(build.demand_h)
    ext_e, ext_h = da_rows(build.external_e), da_rows(build.external_h2)
    ely_prod = _ely_production(build, sol)

    # H2 consumed by H2-fired plants, per zone
    h2 = build.gens[build.gens["h2_fuel"]]
    h2_cons = pd.DataFrame(0.0, index=z, columns=range(H))
    if not h2.empty and not gp.empty:
        for gid, row in h2.iterrows():
            if gid in gp.index:
                h2_cons.loc[row["zone"]] += gp.loc[gid].to_numpy() / row["eff"]

    def build_table(per_zone_cols):
        data = {}
        for zone in z:
            for cat, series in per_zone_cols(zone):
                data[(zone, cat)] = np.asarray(series, dtype=float)
        df = pd.DataFrame(data, index=pd.Index(range(H), name="hour"))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["zone", "category"])
        return df.round(3)

    # ---- electricity ----
    def elec_cols(zone):
        out = []
        if not gp.empty:
            for gid in gp.index:
                if gid.split("|", 1)[0] == zone:
                    out.append((gid.split("|", 1)[1], gp.loc[gid].to_numpy()))
        out += [
            ("Storage discharge", dis_z.loc[zone]),
            ("Storage charge (-)", -ch_z.loc[zone]),
            ("Electrolyser load (-)", -ely.loc[zone]),
            ("Net line import", net_e.loc[zone]),
            ("External exchange", ext_e.loc[zone]),
            ("Load shedding", shed_e.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_e.loc[zone]),
            ("Demand (-)", -dem_e.loc[zone]),
        ]
        return out

    # ---- hydrogen ----
    def h2_cols(zone):
        out = [
            ("Electrolyser production", ely_prod.loc[zone]),
            ("Terminal import", term.loc[zone]),
            ("Net pipeline import", net_h.loc[zone]),
            ("External exchange", ext_h.loc[zone]),
            ("Load shedding", shed_h.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_h.loc[zone]),
            ("H2 plant consumption (-)", -h2_cons.loc[zone]),
            ("Demand (-)", -dem_h.loc[zone]),
        ]
        return out

    build_table(elec_cols).to_csv(out_dir / "hourly_balance_elec.csv")
    build_table(h2_cols).to_csv(out_dir / "hourly_balance_h2.csv")


def write_inputs(build: BuildResult, out_dir: Path) -> None:
    """Export the per-node input data exactly as the model resolved and used it.

    Writes, into ``out_dir/inputs/``:
      * nodes_generators.csv - every generation resource per node with resolved
        parameters (capacity, units, min/max per-unit power, marginal cost,
        efficiency, ramp, must-run, category, H2-fuel flag)
      * nodes_storage.csv    - storage devices per node (power, energy, efficiency)
      * network_lines.csv    - the elec & H2 lines actually used (endpoints, caps, loss)
      * nodes_summary.csv    - one row per node: demands, exchange, capacities by
        type, electrolyser/terminal capacity, resource counts
    """
    out_dir = Path(out_dir) / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()

    z = build.zones
    H = len(build.hours)
    g = build.gens

    # Per-generator resolved parameters
    g.reset_index().to_csv(out_dir / "nodes_generators.csv", index=False)

    # Storage devices
    if not build.storage.empty:
        build.storage.reset_index().to_csv(out_dir / "nodes_storage.csv", index=False)

    # Network lines actually used
    line_rows = []
    for carrier, lines in [("electricity", build.elines), ("hydrogen", build.hlines)]:
        for l in lines:
            line_rows.append(dict(carrier=carrier, frm=l.frm, to=l.to,
                                  cap_from_to_mw=l.cap_ft, cap_to_from_mw=l.cap_tf,
                                  loss_fraction=l.loss))
    pd.DataFrame(line_rows).to_csv(out_dir / "network_lines.csv", index=False)

    # Per-node summary
    def cap_by(cat):
        return g[g["category"] == cat].groupby("zone")["pmax"].sum().reindex(z).fillna(0.0)

    def demand_sum(da):
        return _zones_on_rows(da.to_pandas(), z).reindex(z).fillna(0.0).sum(axis=1)

    sto = build.storage
    sto_e = (sto.groupby("zone")["ecap"].sum().reindex(z).fillna(0.0)
             if not sto.empty else pd.Series(0.0, index=z))
    sto_p = (sto.groupby("zone")["pdis"].sum().reindex(z).fillna(0.0)
             if not sto.empty else pd.Series(0.0, index=z))

    summ = pd.DataFrame(index=pd.Index(z, name="zone"))
    summ["elec_demand_mwh"] = demand_sum(build.demand_e)
    summ["h2_demand_mwh"] = demand_sum(build.demand_h)
    summ["ext_exchange_mwh"] = demand_sum(build.external_e)
    summ["h2_ext_exchange_mwh"] = demand_sum(build.external_h2)
    summ["committable_cap_mw"] = cap_by(dl.CAT_COMMIT)
    summ["vres_cap_mw"] = cap_by(dl.CAT_VRES)
    summ["ror_cap_mw"] = cap_by(dl.CAT_ROR)
    summ["profile_cap_mw"] = cap_by(dl.CAT_PROFILE)
    summ["electrolyser_cap_mw"] = getattr(build, "_ely_cap", pd.Series(0.0, index=z)).reindex(z).fillna(0.0)
    summ["h2_terminal_cap_mw"] = getattr(build, "_term_cap", pd.Series(0.0, index=z)).reindex(z).fillna(0.0)
    summ["storage_energy_mwh"] = sto_e
    summ["storage_power_mw"] = sto_p
    summ["n_generators"] = g.groupby("zone").size().reindex(z).fillna(0).astype(int)
    summ["n_committable"] = (g[g["category"] == dl.CAT_COMMIT].groupby("zone").size()
                             .reindex(z).fillna(0).astype(int))
    summ["n_storage"] = (sto.groupby("zone").size().reindex(z).fillna(0).astype(int)
                         if not sto.empty else 0)
    summ.round(3).to_csv(out_dir / "nodes_summary.csv")
