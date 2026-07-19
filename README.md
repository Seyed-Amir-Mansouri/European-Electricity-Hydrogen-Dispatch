# Coupled Electricity + Hydrogen Economic Dispatch

A **MILP unit-commitment economic dispatch** over the **Central-European CORE
region** — the database covers the 23 ENTSO-E bidding zones of the 13 CORE
Capacity-Calculation-Region countries (AT, BE, CZ, DE, FR, HR, HU, LU, NL, PL,
RO, SI, SK) — coupling two energy carriers, **electricity** and **hydrogen**,
using [linopy](https://linopy.readthedocs.io) + the open-source **HiGHS** solver.
The horizon is a whole number of days (default one day = 24 hours; use
`--start-day/--end-day` for multi-day runs). All data is the **ENTSO-E TYNDP
National Trends 2030 (NT2030)** scenario, supplied as the parquet databases in
`inputs/`.

## Quick start

Dependencies live in the shared `projects-venv`; code lives here in `Project 1/`.

```bash
# from Project 1/
"../projects-venv/Scripts/pip.exe" install -r requirements.txt   # first time only
"../projects-venv/Scripts/python.exe" run_dispatch.py                          # all zones, day 1
"../projects-venv/Scripts/python.exe" run_dispatch.py --zones DE00,FR00 --day 10       # a single day
"../projects-venv/Scripts/python.exe" run_dispatch.py --start-day 10 --end-day 16      # a 7-day horizon
"../projects-venv/Scripts/python.exe" run_dispatch.py --no-ramps --reserves
```

Everything the model needs is in the **`inputs/` NT2030 databases** — zone data,
network topology + prices, and cross-border flows — so a fresh clone runs
straight away.

CLI flags:

| Flag | Meaning |
|------|---------|
| `--zones DE00,FR00,…` | subset of zones (default: all zones in the database) |
| `--day D` | single day of year `D` (1-364); shorthand for `--start-day D --end-day D` |
| `--start-day S --end-day E` | multi-day horizon covering days `S..E` inclusive (`(E-S+1)·24` hours) |
| `--no-storage` | drop storage & state-of-charge |
| `--no-ramps` | drop generator ramp limits |
| `--reserves` | enable FCR/FRR head-room constraints |
| `--no-h2-terminal` | forbid hydrogen terminal imports |
| `--no-prices` | skip the marginal-price computation (the extra LP re-solve) |
| `--time-limit S` | solver time limit in seconds |
| `--out-tag NAME` | write results to `outputs/NAME/` instead of `outputs/` (keep runs side by side) |

Results are written to `outputs/` and a balance-validation check prints at the
end. Each run **wipes the existing `*.csv` in its output folder first** (clean
slate), so the folder always reflects exactly the last run — no stale files
linger from a previous run with different options. By default all runs share the
one `outputs/` folder, so a run overwrites the previous one's results. To keep
runs side by side, pass **`--out-tag NAME`**, which writes to `outputs/NAME/`
(the clean-slate wipe is scoped to that subfolder, so tagged and untagged runs
don't clobber each other). Example:

```bash
"../projects-venv/Scripts/python.exe" run_dispatch.py --day 1  --out-tag winter_day
"../projects-venv/Scripts/python.exe" run_dispatch.py --day 200 --out-tag summer_day
# -> outputs/winter_day/  and  outputs/summer_day/  side by side
```

Multi-day runs build a larger MILP (constraints scale with the number of hours);
storage state-of-charge is cyclic over the **whole** horizon and must-run uses
the first day's month.

**Zones default to every zone in the database.** Selecting a subset with
`--zones` automatically reclassifies each border: a line between two selected
zones stays an internal (optimised) link, while a line to a non-selected zone
becomes a fixed cross-border exchange. Requesting a zone absent from the
database is a clear error.

## Hourly per-technology balance (PLEXOS-style)

Each run writes exactly two CSVs to `outputs/` — wide tables in the style of the
market model's hourly per-technology output (electricity and hydrogen), with a
two-level column header `(zone, category)` and one row per hour:

| File | Per-zone categories |
|------|---------------------|
| `hourly_balance_elec.csv` | each generation technology (MW), plus Storage discharge / charge, Electrolyser load, Net line import, External exchange, Load shedding, Dumped/curtailed, Demand, **Marginal Price (EUR/MWh)** |
| `hourly_balance_h2.csv` | Electrolyser production, Terminal import, Net pipeline import, External exchange, H2 storage discharge / charge, Load shedding, Dumped/curtailed, H2 plant consumption, Demand, **Marginal Price (EUR/MWh)** |

Signs are chosen so supply is `+` and consumption `-`, so the energy (MW)
categories of each row sum to ~0 (the nodal balance holds).

**Marginal Price (EUR/MWh)** is the zonal price — the dual of the nodal balance.
Because duals need an LP, the model fixes the integer commitment to its MILP
optimum and re-solves as an LP to read the balance shadow prices (skip this extra
solve with `--no-prices`).
