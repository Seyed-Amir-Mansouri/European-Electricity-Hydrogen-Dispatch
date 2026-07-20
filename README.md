<p align="center">
  <img src="Power-Hydrogen%20Co-Dispatch%20Overview.png" alt="Power-Hydrogen Co-Dispatch Overview" width="760">
</p>

A **linear-programming economic dispatch** over the **Central-European CORE
region** — the database covers the 23 ENTSO-E bidding zones of the 13 CORE
Capacity-Calculation-Region countries (AT, BE, CZ, DE, FR, HR, HU, LU, NL, PL,
RO, SI, SK) — coupling two energy carriers, **electricity** and **hydrogen**,
using [linopy](https://linopy.readthedocs.io) + the open-source **HiGHS** solver.
The horizon is a whole number of days (default one day = 24 hours; use
`--start-day/--end-day` for multi-day runs). All data is the **ENTSO-E TYNDP
National Trends 2030 (NT2030)** scenario, supplied as the parquet databases in
`inputs/`.

## Setup

Requires **Python 3.12+**. From the project folder, create and activate a virtual
environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell   (CMD: .venv\Scripts\activate.bat)
pip install -r requirements.txt
```

Activating the environment puts `python`, `pip`, and `streamlit` on your PATH in
that terminal, so the commands below work as written. To use them from **any**
terminal without activating first, add the environment's `Scripts` folder to the
Windows **PATH** environment variable (Settings → *Edit the system environment
variables* → *Environment Variables* → select *Path* → *New*), then restart the
terminal.

## Quick start

```bash
python run_dispatch.py                                    # all zones, day 1
python run_dispatch.py --zones DE00,FR00 --day 10         # a single day
python run_dispatch.py --start-day 10 --end-day 16        # a 7-day horizon
python run_dispatch.py --no-ramps --reserves
```

Everything the model needs is in the **`inputs/` NT2030 databases** — zone data,
network topology + prices, and cross-border flows — so a fresh clone runs
straight away.

## Web UI

An interactive [Streamlit](https://streamlit.io) app wraps the model — run it
locally after installing the requirements:

```bash
streamlit run app.py
```

It opens in your browser (`localhost:8501`). Choose the zones, horizon, and
options in the sidebar, click **Run dispatch**, then explore the per-zone
**generation stack**, the electricity & hydrogen **marginal-price** charts, and
the **balance tables** (with CSV download). The app calls the same model code as
the CLI.

## Command line

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
| `--no-prices` | skip reading the marginal-price duals from the solved LP |
| `--rolling-days N` | solve a long horizon in `N`-day rolling blocks (see below) |
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
python run_dispatch.py --day 1  --out-tag winter_day
python run_dispatch.py --day 200 --out-tag summer_day
# -> outputs/winter_day/  and  outputs/summer_day/  side by side
```

Multi-day runs build a larger LP (constraints scale with the number of hours);
storage state-of-charge is cyclic over the **whole** horizon and must-run uses
the first day's month.

### Long horizons (a month or a full year)

A monolithic LP over hundreds or thousands of hours becomes too large to solve
directly (the full 8736-hour year has ~4 M rows, which the simplex/barrier
methods do not finish on a typical machine). For horizons beyond a couple of
weeks, use **rolling-horizon** decomposition: the run is split into consecutive
day-blocks, each solved as its own LP, with **storage state-of-charge carried
forward** from one block to the next (storage is not forced cyclic within a
block). The per-block balance tables are stitched into the same two full-horizon
CSVs.

```bash
# full year in weekly blocks (52 blocks); skip prices for speed
python run_dispatch.py --start-day 1 --end-day 364 --rolling-days 7 --no-prices
```

A weekly block solves in tens of seconds, so a full year completes in roughly
half an hour rather than hanging indefinitely. Smaller blocks are faster and use
less memory but see less of the future (e.g. a battery cannot arbitrage across a
block boundary); a week is a good default. The Streamlit UI exposes the same
control as **Rolling block (days)** in the sidebar (`0` = solve monolithically).

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
Since the dispatch is a pure LP, this dual comes straight from the single solve —
no re-solve is needed (skip computing prices at all with `--no-prices`). Empty
nodes (no demand, generation, or lines) have a degenerate dual that pins at the
shedding penalty, so a price at that penalty with no actual shedding is treated
as undefined and left blank.
