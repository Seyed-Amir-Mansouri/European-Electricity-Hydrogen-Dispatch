<p align="center">
  <img src="Power-Hydrogen%20Co-Dispatch%20Overview.png" alt="Power-Hydrogen Co-Dispatch Overview" width="760">
</p>

A linear-programming economic dispatch model for the Central-European CORE
region — 20 ENTSO-E bidding zones across the 13 CORE Capacity-Calculation-Region
countries (AT, BE, CZ, DE, FR, HR, HU, LU, NL, PL, RO, SI, SK). It couples two
energy carriers, electricity and hydrogen, and solves with
[linopy](https://linopy.readthedocs.io) on top of the open-source HiGHS solver.
A run covers a whole number of days — one day (24h) by default, or a range via
`--start-day`/`--end-day`. All the input data is the ENTSO-E TYNDP National
Trends 2030 (NT2030) scenario, shipped as parquet files under `inputs/`.

## Setup

You'll need Python 3.12+. From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell   (CMD: .venv\Scripts\activate.bat)
pip install -r requirements.txt
```

With the environment activated, `python` and `pip` in that terminal point at
it, so the commands below just work. If you'd rather not activate every time,
add the environment's `Scripts` folder to your Windows PATH (Settings → *Edit
the system environment variables* → *Environment Variables* → *Path* → *New*)
and restart the terminal.

## Quick start

```bash
python run_dispatch.py                                                # all zones, day 1
python run_dispatch.py --zones DE00,FR00 --start-day 10 --end-day 10  # a single day
python run_dispatch.py --start-day 10 --end-day 16                    # a 7-day horizon
python run_dispatch.py --zones DE00 --uc                              # unit commitment
```

The `inputs/` databases already have everything the model needs — zone data,
network topology and prices, cross-border flows — so a fresh clone runs
without any extra downloads or setup.

## Web UI

There's a small Django app that wraps the model:

```bash
python webui/manage.py runserver
```

Open `http://127.0.0.1:8000/` and you'll get a form: pick zones or whole
countries (checking one zone of a country pulls in its siblings), a 2030 date
range, and hit **Run dispatch**. The results page plots our electricity and
hydrogen marginal prices against PLEXOS's own reference series, alongside
per-zone validation metrics — correlation, RMSE, mean difference, real
shedding hours. It's calling the exact same model code as the CLI underneath,
just with a form in front of it.

## Command line

| Flag | Meaning |
|------|---------|
| `--zones DE00,FR00,…` | subset of zones (default: all zones in the database) |
| `--start-day S --end-day E` | multi-day horizon covering days `S..E` inclusive (`(E-S+1)·24` hours) |
| `--uc` | enable unit commitment (a small MILP — min up/down time, currently wired up only for Gas conv_old1/ccgt_old1 in DE00) |
| `--out-tag NAME` | write results to `outputs/NAME/` instead of `outputs/`, so you can keep runs side by side |

Results land in `outputs/`, and a balance-validation check prints once the
run finishes. Each run clears the `*.csv` files already sitting in its output
folder before writing new ones, so that folder always reflects exactly the
last run — nothing stale lingers from an earlier run with different options.
By default every run shares the same `outputs/` folder, so a new run just
overwrites the last one. Pass `--out-tag NAME` if you want runs kept apart —
that scopes both the output folder and the clean-slate wipe to
`outputs/NAME/`, so tagged and untagged runs never step on each other:

```bash
python run_dispatch.py --start-day 1   --end-day 1   --out-tag winter_day
python run_dispatch.py --start-day 200 --end-day 200 --out-tag summer_day
# -> outputs/winter_day/  and  outputs/summer_day/  side by side
```

Longer horizons mean a bigger LP — the constraint count scales with the
number of hours. Every storage device has to end the horizon no lower than
where it started (`soc[last hour] ≥ soc[first hour]`, a full cycle over the
run), so a run can't look artificially good just by draining reservoirs it
started full. Must-run behavior is keyed off the first day's month.

Zones default to everything in the database. Pass `--zones` and each border
gets reclassified on the fly: a line between two zones you kept stays an
internal, optimized link, while a line to a zone you dropped becomes a fixed
cross-border exchange instead. Asking for a zone that isn't in the database
just fails loudly rather than silently ignoring it.

## Hourly per-technology balance (PLEXOS-style)

Every run writes exactly two CSVs to `outputs/` — wide tables in the same
style as the market model's own hourly per-technology output, one for each
carrier, with a two-level `(zone, category)` column header and one row per
hour:

| File | Per-zone categories |
|------|---------------------|
| `hourly_balance_elec.csv` | each generation technology (MW), plus Storage discharge / charge, Electrolyser load, Net line import, External exchange, Load shedding, Dumped/curtailed, Demand, Marginal Price (EUR/MWh) |
| `hourly_balance_h2.csv` | Electrolyser production, Terminal import, Net pipeline import, External exchange, H2 storage discharge / charge, Load shedding, Dumped/curtailed, H2 plant consumption, Demand, Marginal Price (EUR/MWh) |

Supply is positive and consumption is negative, so the energy (MW) categories
in each row sum to roughly zero — that's just the nodal balance holding.

Marginal Price (EUR/MWh) is the zonal price, taken as the dual of the nodal
balance constraint. Because the dispatch is a pure LP, that dual comes
straight out of the single solve — no re-solve needed — so a price is always
available. The one exception is a node with nothing going on (no demand,
generation, or lines): its dual is degenerate and pins at the shedding
penalty, so if that happens with no real shedding behind it, the price is
left blank rather than reported as a fake VOLL spike.
