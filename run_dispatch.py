"""CLI: solve the coupled electricity + hydrogen economic dispatch for one day.

Examples
--------
    python run_dispatch.py                          # all 23 zones, day 1
    python run_dispatch.py --zones DE00,FR00 --day 10        # a single day (10th)
    python run_dispatch.py --start-day 10 --end-day 16       # a 7-day horizon (168 h)
    python run_dispatch.py --no-ramps --reserves
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from economic_dispatch.config import RunConfig
from economic_dispatch import data_loader, network_loader, model, solve, report


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zones", default=None,
                   help="comma-separated zone codes (default: all 23)")
    p.add_argument("--day", type=int, default=None,
                   help="single day of year (1-364); shorthand for --start-day D --end-day D")
    p.add_argument("--start-day", type=int, default=None,
                   help="first day of a multi-day horizon (1-364)")
    p.add_argument("--end-day", type=int, default=None,
                   help="last day of a multi-day horizon, inclusive (1-364)")
    p.add_argument("--no-storage", action="store_true")
    p.add_argument("--no-ramps", action="store_true")
    p.add_argument("--reserves", action="store_true", help="enable FCR/FRR constraints")
    p.add_argument("--no-h2-terminal", action="store_true")
    p.add_argument("--time-limit", type=float, default=600.0, help="solver time limit (s)")
    p.add_argument("--out-tag", default=None,
                   help="write results to outputs/<TAG>/ instead of outputs/ (keep runs side by side)")
    a = p.parse_args()

    # Resolve the day horizon: --start-day/--end-day take precedence over --day.
    if a.start_day is not None or a.end_day is not None:
        start = a.start_day if a.start_day is not None else a.end_day
        end = a.end_day if a.end_day is not None else a.start_day
    elif a.day is not None:
        start = end = a.day
    else:
        start = end = 1
    if not (1 <= start <= end <= 364):
        p.error(f"invalid day range: start={start}, end={end}. Require 1 <= start <= end <= 364.")

    cfg = RunConfig(start_day=start, end_day=end)
    # Zones come from the consolidated database (build it with build_zones_db.py).
    if not Path(cfg.zones_db).exists():
        p.error(f"zone database not found: {cfg.zones_db}. Run `python build_zones_db.py` first.")
    available = data_loader.zones_in_db(cfg.zones_db)
    if a.zones:
        cfg.zones = [z.strip() for z in a.zones.split(",") if z.strip()]
    else:
        cfg.zones = available
    missing = [z for z in cfg.zones if z not in available]
    if missing:
        p.error(f"zone(s) not in {Path(cfg.zones_db).name}: {missing}. Available: {available}")
    cfg.enable_storage = not a.no_storage
    cfg.enable_ramps = not a.no_ramps
    cfg.enable_reserves = a.reserves
    cfg.enable_h2_terminal = not a.no_h2_terminal
    cfg.time_limit_s = a.time_limit
    cfg.out_tag = a.out_tag
    return cfg


def run(cfg: RunConfig) -> model.BuildResult:
    h0, h1 = cfg.hour_slice()
    span = (f"day {cfg.start_day}" if cfg.num_days() == 1
            else f"days {cfg.start_day}-{cfg.end_day}")
    print(f"Zones: {len(cfg.zones)}  {span}  "
          f"({cfg.num_days()} day(s), hours {h0}-{h1 - 1}, {h1 - h0}h)")

    t = time.time()
    zdata = data_loader.load_zones_from_db(cfg.zones, cfg.zones_db, h0, h1)
    net = network_loader.load_networks(cfg.zones, cfg.networks_db)
    print(f"Loaded {len(zdata)} zones, {len(net.elec)} elec lines, "
          f"{len(net.hydrogen)} H2 lines  ({time.time() - t:.1f}s)  "
          f"CO2={net.co2_price} EUR/t")

    t = time.time()
    build = model.build_model(zdata, net, cfg)
    nv = len(build.model.variables.labels) if hasattr(build.model.variables, "labels") else "?"
    print(f"Built model: {len(build.gens)} generators "
          f"({len(build.commit)} committable), {len(build.storage)} storage units "
          f"({time.time() - t:.1f}s)")

    t = time.time()
    status = solve.solve(build)
    print(f"Solved in {time.time() - t:.1f}s")
    if status != "ok":
        print(f"WARNING: solver returned status={status}")
        return build

    s = report.summary(build)
    print("\n=== Summary ===")
    for k, v in s.items():
        print(f"  {k:28s}: {v:,.1f}")

    val = report.validate(build)
    print("\n=== Balance validation ===")
    ok = val["max_elec_residual"] < val["tol"] and val["max_h2_residual"] < val["tol"]
    print(f"  max elec residual: {val['max_elec_residual']:.2e} MW")
    print(f"  max H2   residual: {val['max_h2_residual']:.2e} MW")
    print(f"  -> {'PASS' if ok else 'FAIL'} (tol={val['tol']})")

    out_dir = cfg.resolved_output_dir()
    report.write_outputs(build, out_dir)
    print(f"\nOutputs written to {out_dir}")
    return build


if __name__ == "__main__":
    run(parse_args())
