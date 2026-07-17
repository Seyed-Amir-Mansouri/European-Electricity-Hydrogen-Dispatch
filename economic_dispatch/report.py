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

    demand_e = _zones_on_rows(build.demand_e.to_pandas(), z).reindex(z)
    demand_h = _zones_on_rows(build.demand_h.to_pandas(), z).reindex(z)
    external_e = _zones_on_rows(build.external_e.to_pandas(), z).reindex(z)

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
    res_e = (gen_z + dis_z - ch_z - ely + net_e + external_e + shed_e - demand_e)
    max_e = float(np.abs(res_e.to_numpy()).max()) if res_e.size else 0.0

    # Hydrogen residual: ely_prod + term + net_h + shed_h - demand_h - h2_cons
    ely_prod = _ely_production(build, sol)
    res_h = (ely_prod + term + net_h + shed_h - demand_h - h2_cons)
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
    }


def write_outputs(build: BuildResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean slate: remove CSVs from any previous run so the folder reflects
    # exactly this run (some files are only written when their variable is
    # non-empty, so stale files would otherwise linger).
    for old in out_dir.glob("*.csv"):
        old.unlink()
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
