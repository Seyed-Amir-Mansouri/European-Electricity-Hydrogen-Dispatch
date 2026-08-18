"""End-to-end scenario runner shared by the CLI and the web UI.

``solve_scenario`` loads the data, builds and solves the dispatch LP for the
whole horizon, attaches marginal prices, and returns the :class:`BuildResult`.
"""
from __future__ import annotations

from .config import RunConfig
from . import data_loader, network_loader, model, solve
from .model import BuildResult


def solve_scenario(cfg: RunConfig,
                   capacity_overrides: dict[str, dict[str, float]] | None = None) -> BuildResult:
    """Load data, build and solve the dispatch, attach prices; return the result.

    ``capacity_overrides``, if given, is ``{zone: {tech: value}}`` -- replaces
    the zone workbook's own "Technology Capacities" (MW), "Storage
    Capacities" (MWh energy), or (for a ``"<tech>||<attr>"`` key, e.g.
    Battery's charge/discharge MW power, which lives on the "Technology
    Characteristics" sheet instead) characteristics entries for the given
    (zone, tech) pairs before the model is built (used by the web UI's
    per-zone capacity editor). Only techs already present in one of those
    three places are overridden; unknown techs are ignored.
    """
    h0, h1 = cfg.hour_slice()
    zdata = data_loader.load_zones_from_db(cfg.zones, cfg.zones_db, h0, h1)
    if capacity_overrides:
        for z, techs in capacity_overrides.items():
            zd = zdata.get(z)
            if zd is None:
                continue
            for tech, val in techs.items():
                if tech in zd.capacities:
                    zd.capacities[tech] = val
                elif tech in zd.storage_energy:
                    zd.storage_energy[tech] = val
                elif "||" in tech:
                    char_tech, attr = tech.split("||", 1)
                    if char_tech in zd.char.index and attr in zd.char.columns:
                        zd.char.at[char_tech, attr] = val
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    build = model.build_model(zdata, net, cfg)
    solve.solve(build)
    startup_cost_eur = 0.0
    if build.uc_gens:
        fixed_profile, startup_cost_eur = model.uc_fixed_profile_and_cost(build)
        build = model.build_model(zdata, net, cfg, fixed_uc_profile=fixed_profile)
        solve.solve(build)
    build.price_e, build.price_h = model.marginal_prices(build)
    build.startup_cost_eur = startup_cost_eur
    return build
