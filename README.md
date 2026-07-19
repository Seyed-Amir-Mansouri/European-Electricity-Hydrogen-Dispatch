# Coupled Electricity + Hydrogen Economic Dispatch

A **MILP unit-commitment economic dispatch** over ENTSO-E-style bidding zones,
coupling two energy carriers — **electricity** and **hydrogen** — using
[linopy](https://linopy.readthedocs.io) + the open-source **HiGHS** solver. The
horizon is a whole number of days (default one day = 24 hours; use
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
becomes a fixed cross-border exchange (see below). Requesting a zone absent from
the database is a clear error.

## What the model does

Over the chosen horizon (one or more days) it minimises total variable
generation cost while meeting electricity and hydrogen demand in every zone and
hour, subject to unit commitment, storage, ramps, and inter-zone transport.

### Decisions
- **Integer unit commitment** — each dispatchable technology is a *fleet*
  (`Number of Units`); the model commits an integer count of units, each
  operating between its minimum stable power and per-unit capacity.
- **Continuous dispatch** of variable renewables (curtailable), run-of-river
  hydro, and `Other RES` / `Other Non-RES` / `DSR` (capped by hourly profile).
- **Storage** charge / discharge / state-of-charge for batteries and hydro
  reservoir & pumped-storage on the electricity side (with natural inflows), and
  **hydrogen storage** on the H2 side.
- **Electrolysers** convert electricity → hydrogen; **hydrogen-fired plants**
  convert hydrogen → electricity — the two couplers between the carriers.
- **Network flows** on the electricity and hydrogen transport grids.
- **Load-shedding slacks** (electricity & hydrogen) priced at VOLL, so the
  problem is always feasible and any supply shortfall is reported, not hidden.

### The two coupled balances (per zone, per hour)
```
ELEC:  committable gen + RES + hydro + storage discharge + H2-plant gen
       + net line imports (after losses) + net external elec import + shed_e
     = demand + storage charge + electrolyser draw
HYDROGEN:
       electrolyser output + terminal imports + net H2 line imports
       + net external H2 import + shed_h
     = H2 demand + H2-plant fuel use (gen / efficiency)

(Net external import is the fixed exchange with non-modelled neighbours, positive
= import; derived from the cross-border databases — see below.)
```

## Data model

Per-zone data is in **`inputs/zones_2030.parquet`** — for every zone:

| Category | Contents |
|----------|----------|
| Technology capacities | installed MW per technology |
| Storage capacities | storage energy (MWh) |
| Reserve requirements | FCR/FRR (only with `--reserves`) |
| Hourly profiles | demand, RES capacity factors / MW, hydro inflows |
| Technology characteristics | units, min stable power, ramps, efficiency, CO2 factor, prices, must-run |
| Gas & hydrogen assets | hydrogen terminal / storage capacities |

**`inputs/networks_2030.parquet`** holds the electricity & hydrogen line topology
(directional MW limits + loss fractions, hydrogen capacities per the NT2030
reference grid) and the global CO₂ & gas prices; the model filters these lines to
the selected zones at runtime.

### Cross-border exchange from the result databases (`inputs/`)
The fixed exchange with non-modelled neighbours is **computed on the fly** from
the crossborder-flow databases in `inputs/` (`crossborder_electricity_2030.parquet`,
`crossborder_hydrogen_2030.parquet`, `smr_production_2030.parquet`), so the
in/out-of-scope split follows the **actual zone selection** — change `--zones`
and each border is reclassified as an internal optimised line (both endpoints
selected) or a fixed external exchange (neighbour outside the selection).
Hydrogen is resolved at country level, with `IB*` interconnector hubs and Steam-
Methane-Reformer output folded in, attached to each country's main zone. The
method is specified in [`inputs/EXPORTS_CALCULATION.md`](inputs/EXPORTS_CALCULATION.md).

### Zone database format
`inputs/zones_2030.parquet` is a long, lossless table
(`zone, section, item, hour, value_num, value_str`): scalar data and technology
characteristics at `hour = -1`, hourly profiles at `hour = 0..8735`.
`data_loader` reconstructs each zone's data from it with predicate-pushdown by
zone.

## Key assumptions (all tunable in `economic_dispatch/config.py`)

- **Marginal cost** = `VOM Price + Fuel + CO2Factor·CO2Price`.
  In this dataset both the **Fuel** and **CO2Factor** columns are already per
  MWh-elec (per power generation), so neither is divided by efficiency
  (`fuel_per_thermal=False`, `co2_per_thermal=False`). Each division is
  independently switchable in `config.py` if a column is per MWh-thermal instead.
- **Hydrogen-fired plants pay no exogenous fuel cost** — their hydrogen is
  supplied by the hydrogen balance (electrolysers/terminals), which already
  carries the cost. Charging the 63.5 EUR/MWh H2 fuel price too would double count.
- **Gas network is out of scope** (electricity + hydrogen only); gas-fired plants
  simply pay the global gas price as fuel.
- **Ramp rates** (`Ramp-Up/Down Rate (MW/h)`) are treated as *per-unit* and scaled
  by the fleet size. If they turn out to be per-minute or already fleet-level,
  adjust `ramp_scale` or set `--no-ramps`.
- **Must-run** uses the calendar-month value (12-value list) for the first day
  of the horizon.
- **Maintenance** (`Maximum Number of Units in Maintenance`) is a scheduling
  ceiling, not a forced outage, so the full fleet is committable for one day.
- **Storage** starts at 50% state of charge and returns to it by day's end
  (cyclic); round-trip efficiency applied on charging.
- **Hydrogen terminal imports** are allowed at `h2_terminal_price` (default
  150 EUR/MWh, an assumption) up to the `Terminal (Hydrogen)` capacity.
- **Hydrogen storage** (`enable_h2_storage`) uses `Withdraw`/`Injection
  (Hydrogen)` as discharge/charge power. The data gives no energy value, so its
  capacity is an **assumption**: `h2_storage_hours × Withdraw (Hydrogen)`
  (default 168 h), round-trip efficiency `h2_storage_efficiency` (default 1.0).

> **Note on supply shortfalls:** load-shedding and dumping enter both balances as
> slacks (priced at VOLL / a small dump penalty), so the problem is always
> feasible and any unmet demand or curtailed surplus is reported, not hidden. With
> the full NT2030 data, hydrogen demand is served in full (zero shedding).

> **Note on a suppressed linopy warning:** `solve.py` silences the
> `"Coordinates across variables not equal. Perform outer join."` `UserWarning`.
> This is intrinsic to linopy, not a bug: it bundles all variables into one
> dataset via an *exact* coordinate align, which necessarily fails when variables
> live on different dimensions (`gen` / `zone` / `sto` / `line`), so it falls back
> to a harmless no-op outer join. `report.validate()` confirms every balance
> still closes to ~1e-13 MW. Truly removing it would require collapsing all
> variables onto a single flat index (a large rewrite with no correctness gain),
> so we suppress that one specific, benign message instead.

## Layout

```
economic_dispatch/
  config.py          run settings & tunable assumptions
  data_loader.py     load zone data from the database + technology classification
  network_loader.py  load the networks database
  model.py           build the linopy MILP
  exports_loader.py  compute cross-border exchange from the inputs/ databases
  solve.py           run HiGHS
  report.py          extract, validate balances, write CSVs
run_dispatch.py      CLI entry point
inputs/              NT2030 databases (zones, networks, cross-border) + EXPORTS_CALCULATION.md
outputs/             results CSVs (generation, flows, storage, shedding, summary,
                     hourly per-tech balance)
outputs/inputs/      per-node input data as the model resolved it (see below)
```

## Hourly per-technology balance (PLEXOS-style)

Every run writes two wide CSVs in the style of the market model's hourly
per-technology output (electricity and hydrogen) — a two-level column header
`(zone, category)` with one row per hour:

| File | Per-zone categories |
|------|---------------------|
| `hourly_balance_elec.csv` | each generation technology (MW), plus Storage discharge / charge, Electrolyser load, Net line import, External exchange, Load shedding, Dumped/curtailed, Demand |
| `hourly_balance_h2.csv` | Electrolyser production, Terminal import, Net pipeline import, External exchange, H2 storage discharge / charge, Load shedding, Dumped/curtailed, H2 plant consumption, Demand |

Signs are chosen so supply is `+` and consumption `-`, so each row sums to ~0
(the nodal balance holds).

## Exported node data

Every run also writes, to `outputs/inputs/` (or `outputs/<tag>/inputs/`), the
per-node input data exactly as the model built and used it — an audit trail of
what each modelled zone contributed:

| File | Contents |
|------|----------|
| `nodes_generators.csv` | every generation resource per node with resolved parameters: capacity, number of units, per-unit min/max power, marginal cost, efficiency, ramp, must-run, category, H2-fuel flag |
| `nodes_storage.csv` | storage devices per node (discharge/charge power, energy, efficiency) |
| `network_lines.csv` | the electricity & hydrogen lines actually used (endpoints, directional capacities, loss fraction) |
| `nodes_summary.csv` | one row per node: elec/H2 demand, external exchange, capacity by resource type, electrolyser & terminal capacity, storage, resource counts |

## Verification

`report.validate()` independently recomputes both nodal balances from the
solution and asserts residuals < 1e-3 MW. `run_dispatch.py` prints this check
(`PASS`/`FAIL`) after every solve.
```
