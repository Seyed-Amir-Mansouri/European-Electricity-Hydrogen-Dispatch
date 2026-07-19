"""End-to-end scenario runner shared by the CLI and the web UI.

``solve_scenario(cfg)`` loads the databases, builds the MILP, solves it, and
(optionally) computes marginal prices — returning the :class:`BuildResult` in
memory, without writing any files.
"""
from __future__ import annotations

from .config import RunConfig
from . import data_loader, network_loader, model, solve
from .model import BuildResult


def solve_scenario(cfg: RunConfig) -> BuildResult:
    """Load data, build and solve the dispatch, attach prices; return the result."""
    h0, h1 = cfg.hour_slice()
    zdata = data_loader.load_zones_from_db(cfg.zones, cfg.zones_db, h0, h1)
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    build = model.build_model(zdata, net, cfg)
    solve.solve(build)
    if cfg.compute_prices:
        build.price_e, build.price_h = model.marginal_prices(zdata, net, cfg, build)
    return build
