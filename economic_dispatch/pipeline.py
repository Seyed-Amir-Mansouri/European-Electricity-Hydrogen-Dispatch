"""End-to-end scenario runners shared by the CLI and the web UI.

``solve_scenario`` solves the whole horizon as one LP and returns the
:class:`BuildResult`. ``solve_rolling`` splits a long horizon into day-blocks,
solving each in turn and carrying storage state forward — the practical way to
run a full year, since a monolithic 8736-hour model is too large to solve
directly. It returns the concatenated hourly-balance tables.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .config import RunConfig
from . import data_loader, network_loader, model, solve, report
from .model import BuildResult


def solve_scenario(cfg: RunConfig) -> BuildResult:
    """Load data, build and solve the dispatch, attach prices; return the result."""
    h0, h1 = cfg.hour_slice()
    zdata = data_loader.load_zones_from_db(cfg.zones, cfg.zones_db, h0, h1)
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    build = model.build_model(zdata, net, cfg)
    solve.solve(build)
    if cfg.compute_prices:
        build.price_e, build.price_h = model.marginal_prices(build)
    return build


def solve_rolling(cfg: RunConfig, block_days: int, verbose: bool = True) -> dict:
    """Rolling-horizon solve: blocks of ``block_days`` days, storage carried over.

    Returns ``{"elec": df, "h2": df, "ok": bool}`` with the two hourly-balance
    tables concatenated over the whole horizon (global hour index). Storage is
    not cyclic within blocks; each block starts from the previous block's
    end-of-block state of charge.
    """
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    soc_carry = None
    elec_parts, h2_parts, ok = [], [], True
    day = cfg.start_day
    while day <= cfg.end_day:
        bend = min(day + block_days - 1, cfg.end_day)
        bcfg = replace(cfg, start_day=day, end_day=bend)
        h0, h1 = bcfg.hour_slice()
        zdata = data_loader.load_zones_from_db(bcfg.zones, bcfg.zones_db, h0, h1)
        build = model.build_model(zdata, net, bcfg, soc_init=soc_carry, cyclic=False)
        solve.solve(build)
        if cfg.compute_prices:
            build.price_e, build.price_h = model.marginal_prices(build)

        val = report.validate(build)
        ok = ok and val["max_elec_residual"] < val["tol"] and val["max_h2_residual"] < val["tol"]
        tables = report.hourly_balance_tables(build)
        for key, parts in (("elec", elec_parts), ("h2", h2_parts)):
            df = tables[key]
            df.index = pd.Index(range(h0, h1), name="hour")   # global absolute hours
            parts.append(df)
        if len(build.storage):
            soc_carry = build.model.solution["soc"].isel(hour=-1).to_pandas()
        if verbose:
            print(f"  block days {day}-{bend} ({h1 - h0}h): done", flush=True)
        day = bend + 1

    return {"elec": pd.concat(elec_parts), "h2": pd.concat(h2_parts), "ok": ok}
