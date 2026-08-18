"""Views for the CORE electricity + hydrogen dispatch web UI.

Lets the user pick zones (grouped by country -- picking a country's zone
auto-expands to its siblings via RunConfig itself, see economic_dispatch.
config._expand_to_countries), a 2030 calendar date range, run the dispatch
model, and see the resulting marginal prices plotted against PLEXOS's own
reference prices with the same validation metrics used throughout this
project's development.
"""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.shortcuts import render

from economic_dispatch.config import (
    RunConfig, discover_zones, DEFAULT_ZONES_DB,
    DEFAULT_MARGINAL_PRICE_ELEC_DB, DEFAULT_MARGINAL_PRICE_H2_DB,
)
from economic_dispatch import data_loader, pipeline, report

SCARCITY_THRESHOLD = 2000.0

_COUNTRY_NAMES = {
    "AT": "Austria", "BE": "Belgium", "CZ": "Czechia", "DE": "Germany",
    "FR": "France", "HR": "Croatia", "HU": "Hungary", "LU": "Luxembourg",
    "NL": "Netherlands", "PL": "Poland", "RO": "Romania", "SI": "Slovenia",
    "SK": "Slovakia",
}

_TECH_LABELS = {
    "DSR": "Demand Response",
    "Electrolyser": "Electrolyser",
    "Gas (ccgt_ccs)": "Gas (Combined Cycle, Carbon Capture)",
    "Gas (ccgt_new)": "Gas (Combined Cycle, New)",
    "Gas (ccgt_old1)": "Gas (Combined Cycle, Old 1)",
    "Gas (ccgt_old2)": "Gas (Combined Cycle, Old 2)",
    "Gas (ccgt_pre1)": "Gas (Combined Cycle, Pre-Existing 1)",
    "Gas (ccgt_pre2)": "Gas (Combined Cycle, Pre-Existing 2)",
    "Gas (conv_old1)": "Gas (Conventional Steam, Old 1)",
    "Gas (conv_old2)": "Gas (Conventional Steam, Old 2)",
    "Gas (ocgt_new)": "Gas (Open Cycle, New)",
    "Gas (ocgt_old)": "Gas (Open Cycle, Old)",
    "Hard Coal (ccs)": "Hard Coal (Carbon Capture)",
    "Hard Coal (new)": "Hard Coal (New)",
    "Hard Coal (old1)": "Hard Coal (Old 1)",
    "Hard Coal (old2)": "Hard Coal (Old 2)",
    "Heavy oil (old1)": "Heavy Oil (Old 1)",
    "Heavy oil (old2)": "Heavy Oil (Old 2)",
    "Hydro (closed_ps_pump)": "Hydro (Closed-Loop Pumped Storage, Pumping)",
    "Hydro (closed_ps_turbine)": "Hydro (Closed-Loop Pumped Storage, Turbine)",
    "Hydro (open_ps_pump)": "Hydro (Open-Loop Pumped Storage, Pumping)",
    "Hydro (open_ps_turbine)": "Hydro (Open-Loop Pumped Storage, Turbine)",
    "Hydro (pondage)": "Hydro (Pondage)",
    "Hydro (reservoir)": "Hydro (Reservoir)",
    "Hydro (river)": "Hydro (Run-of-River)",
    "Hydrogen (ccgt)": "Hydrogen (Combined Cycle Turbine)",
    "Hydrogen (fc)": "Hydrogen (Fuel Cell)",
    "Light Oil": "Light Oil",
    "Lignite (ccs)": "Lignite (Carbon Capture)",
    "Lignite (new)": "Lignite (New)",
    "Lignite (old1)": "Lignite (Old 1)",
    "Lignite (old2)": "Lignite (Old 2)",
    "Nuclear": "Nuclear",
    "Oil shale (new)": "Oil Shale (New)",
    "Oil shale (old)": "Oil Shale (Old)",
    "Other Non-RES": "Other Non-Renewable",
    "Other RES (biomass)": "Other Renewable (Biomass)",
    "Other RES (geothermal)": "Other Renewable (Geothermal)",
    "Other RES (marine)": "Other Renewable (Marine)",
    "Other RES (unknown)": "Other Renewable (Unspecified)",
    "Other RES (waste)": "Other Renewable (Waste)",
    "Solar": "Solar",
    "Solar (rooftop)": "Solar (Rooftop)",
    "Solar (thermal)": "Solar (Thermal)",
    "Solar (thermal_with_storage)": "Solar (Thermal with Storage)",
    "Wind (offshore)": "Wind (Offshore)",
    "Wind (onshore)": "Wind (Onshore)",
}

_STORAGE_ENERGY_LABELS = {
    "Battery": "Battery",
    "Electrolyser": "Electrolyser",
    "Hydro (closed_ps)": "Hydro (Closed-Loop Pumped Storage)",
    "Hydro (open_ps)": "Hydro (Open-Loop Pumped Storage)",
    "Hydro (pondage)": "Hydro (Pondage)",
    "Hydro (reservoir)": "Hydro (Reservoir)",
    "Solar (thermal_with_storage)": "Solar (Thermal with Storage)",
}

_STORAGE_KIND_LABELS = {
    "Battery": "Battery",
    "Hydro reservoir": "Hydro (Reservoir)",
    "Hydro pondage": "Hydro (Pondage)",
    "Hydro open_ps": "Hydro (Open-Loop Pumped Storage)",
    "Hydro closed_ps": "Hydro (Closed-Loop Pumped Storage)",
}

BATTERY_TECH = "Battery (MWh)"
BATTERY_DISCHARGE_ATTR = "Net maximum capacity - generation perspective (MW)"
BATTERY_CHARGE_ATTR = "Net maximum capacity - demand perspective (MW)"
BATTERY_DISCHARGE_KEY = f"{BATTERY_TECH}||{BATTERY_DISCHARGE_ATTR}"
BATTERY_CHARGE_KEY = f"{BATTERY_TECH}||{BATTERY_CHARGE_ATTR}"

_CHAR_LABELS = {
    BATTERY_DISCHARGE_KEY: "Battery (Discharge)",
    BATTERY_CHARGE_KEY: "Battery (Charge)",
}


def _append_paren(name: str, extra: str) -> str:
    """"Hydro (Reservoir)" + "Energy" -> "Hydro (Reservoir, Energy)";
    "Battery" + "Energy" -> "Battery (Energy)" (no existing parenthetical
    to fold into)."""
    return f"{name[:-1]}, {extra})" if name.endswith(")") else f"{name} ({extra})"


def _tech_label(tech: str) -> str:
    """Human-readable, abbreviation-free label for a raw capacity item name.

    Power (MW) items -- "<Tech> (MW)" or "<Tech><N> (MW)" -- e.g.
    "Gas (ccgt_pre2) (MW)" -> "Gas (Combined Cycle, Pre-Existing 2)",
    "DSR7 (MW)" -> "Demand Response (Product 7)". Falls back to the raw name
    (minus the " (MW)" suffix) for anything not in `_TECH_LABELS`, so a new
    input category never disappears, just shows up unprettified.

    Storage energy (MWh) items -- "<Tech> (MWh)" -- e.g.
    "Hydro (reservoir) (MWh)" -> "Hydro (Reservoir, Energy)".

    Battery charge/discharge power items -- "<tech>||<attr>" characteristics
    keys (`BATTERY_DISCHARGE_KEY`/`BATTERY_CHARGE_KEY`) -- via `_CHAR_LABELS`.
    """
    if tech in _CHAR_LABELS:
        return _CHAR_LABELS[tech]
    if tech.endswith(" (MWh)"):
        base = tech[:-6]
        return _append_paren(_STORAGE_ENERGY_LABELS.get(base, base), "Energy")
    base = tech[:-5] if tech.endswith(" (MW)") else tech
    if base in _TECH_LABELS:
        return _TECH_LABELS[base]
    for prefix, singular in (("DSR", "Demand Response"), ("Other Non-RES", "Other Non-Renewable")):
        if base.startswith(prefix) and base[len(prefix):].isdigit():
            return f"{singular} (Product {base[len(prefix):]})"
    return base


def _tech_unit(tech: str) -> str:
    """"MW" or "MWh", read off the raw item name's own unit suffix (every
    capacity/storage-energy/battery-characteristic key ends in one)."""
    return "MWh" if tech.endswith(" (MWh)") else "MW"


def _generation_label(key: str) -> str:
    """Friendly label for a `report.zone_technology_mix` key -- a generator
    tech (routed through `_tech_label`, which already covers that raw
    "<Tech> (MW)" format) or a storage device `kind` (routed through
    `_STORAGE_KIND_LABELS`, since those bare names -- e.g. "Hydro reservoir"
    -- carry no unit suffix for `_tech_label` to key off of)."""
    return _STORAGE_KIND_LABELS.get(key, _tech_label(key))


_PRODUCT_SUFFIX = re.compile(r"^(.*) \(Product (\d+)\)$")


def _label_sort_key(label: str) -> tuple[str, int]:
    """Sorts by label text, but numbered "<Name> (Product N)" variants
    (e.g. from `_tech_label`'s DSR/Other-Non-RES handling) sort numerically
    within their shared base name -- "Product 2" right after "Product 1",
    not lexicographically after "Product 19" -- with the bare, unnumbered
    label (if any) sorting first."""
    m = _PRODUCT_SUFFIX.match(label)
    return (m.group(1), int(m.group(2))) if m else (label, -1)


def _country_groups() -> list[dict]:
    """[{"name": "Austria (AT)", "code": "AT", "zones": ["AT00"]},
    {"name": "Germany (DE)", "code": "DE", "zones": ["DE00", "DEKF"]}, ...],
    sorted by country name -- one checkbox group per CORE country."""
    groups: dict[str, list[str]] = {}
    for z in discover_zones():
        groups.setdefault(z[:2], []).append(z)
    return [
        {"name": f"{_COUNTRY_NAMES.get(c, c)} ({c})", "code": c,
         "map_path": f"dispatcher/country-maps/{c.lower()}.svg", "zones": sorted(zs)}
        for c, zs in sorted(groups.items(), key=lambda kv: _COUNTRY_NAMES.get(kv[0], kv[0]))
    ]


@lru_cache(maxsize=1)
def _zone_capacity_defaults() -> dict[str, list[tuple[str, str, float, str]]]:
    """{zone: [(tech, label, default_val, unit), ...]} for every zone's own
    "Technology Capacities" (MW) *and* "Storage Capacities" (MWh energy)
    sheet entries, plus Battery's charge/discharge MW power (read from the
    "Technology Characteristics" sheet -- Battery has no entry in either of
    those two other sheets) -- nonzero only (a zero-capacity tech has no
    installed asset in that zone to edit), `label` a human-readable
    abbreviation-free name from `_tech_label`, `default_val` rounded to 3
    decimals, `unit` from `_tech_unit`, sorted by `_label_sort_key` (numeric
    within numbered-product families, alphabetical otherwise). Independent
    of the hour window, so read once and cached for the process lifetime --
    powers the web UI's per-zone capacity editor.
    """
    zdata = data_loader.load_zones_from_db(discover_zones(), DEFAULT_ZONES_DB, 0, 1)
    out: dict[str, list[tuple[str, str, float, str]]] = {}
    for z, zd in zdata.items():
        raw = dict(zd.capacities)
        raw.update(zd.storage_energy)
        dis = zd.char_val(BATTERY_TECH, BATTERY_DISCHARGE_ATTR)
        chg = zd.char_val(BATTERY_TECH, BATTERY_CHARGE_ATTR)
        if dis:
            raw[BATTERY_DISCHARGE_KEY] = dis
        if chg:
            raw[BATTERY_CHARGE_KEY] = chg
        entries = [(tech, _tech_label(tech), round(float(val), 3), _tech_unit(tech))
                  for tech, val in raw.items() if val]
        out[z] = sorted(entries, key=lambda e: _label_sort_key(e[1]))
    return out


def _day_of_year_2030(d: date) -> int:
    """Calendar date -> 1-based day-of-year, clamped to the dataset's
    364-day modelled year (52 weeks) -- Dec 31 of a real 365-day year maps
    to day 364, the last modelled day."""
    return min(d.timetuple().tm_yday, 364)


def index(request):
    return render(request, "dispatcher/index.html", {
        "country_groups": _country_groups(), "year": 2030,
        "zone_capacities": _zone_capacity_defaults(),
    })


def run_dispatch(request):
    if request.method != "POST":
        return render(request, "dispatcher/index.html", {
            "country_groups": _country_groups(), "year": 2030,
            "zone_capacities": _zone_capacity_defaults(),
        })

    requested_zones = request.POST.getlist("zones")
    start_date_str = request.POST.get("start_date", "")
    end_date_str = request.POST.get("end_date", "")

    errors: list[str] = []
    if not requested_zones:
        errors.append("Select at least one zone or country.")

    start_day = end_day = None
    try:
        start_d = date.fromisoformat(start_date_str)
        end_d = date.fromisoformat(end_date_str)
        if start_d.year != 2030 or end_d.year != 2030:
            errors.append("Dates must be in 2030 (the modelled year).")
        elif end_d < start_d:
            errors.append("End date must be on or after the start date.")
        else:
            start_day = _day_of_year_2030(start_d)
            end_day = _day_of_year_2030(end_d)
    except (TypeError, ValueError):
        errors.append("Enter valid start/end dates.")

    if errors:
        return render(request, "dispatcher/index.html", {
            "country_groups": _country_groups(), "year": 2030, "errors": errors,
            "selected_zones": requested_zones, "start_date": start_date_str, "end_date": end_date_str,
            "zone_capacities": _zone_capacity_defaults(),
        })

    cfg = RunConfig(zones=sorted(requested_zones), start_day=start_day, end_day=end_day,
                    enable_uc=False, subtract_dsr_implicit=True)

    capacity_overrides = _parse_capacity_overrides(request.POST, set(cfg.zones))
    build = pipeline.solve_scenario(cfg, capacity_overrides=capacity_overrides)

    plexos_price_e = pd.read_parquet(DEFAULT_MARGINAL_PRICE_ELEC_DB)
    plexos_price_h = pd.read_parquet(DEFAULT_MARGINAL_PRICE_H2_DB)
    h0, h1 = cfg.hour_slice()
    hours = np.arange(h0, h1)

    auto_added = sorted(set(cfg.zones) - set(requested_zones))
    tech_mix = report.zone_technology_mix(build)

    zone_results = []
    for z in cfg.zones:
        ours_e = build.price_e.sel(zone=z).to_numpy()
        px_e = plexos_price_e[z].to_numpy()[h0:h1] if z in plexos_price_e.columns else None
        shed_e = build.model.solution["shed_e"].sel(zone=z).to_numpy()

        hz = f"{z[:2]}_H2"
        ours_h = build.price_h.sel(zone=z).to_numpy()
        px_h = plexos_price_h[hz].to_numpy()[h0:h1] if hz in plexos_price_h.columns else None
        shed_h = build.model.solution["shed_h"].sel(zone=z).to_numpy()

        zone_results.append(dict(
            zone=z,
            metrics_e=_metrics(ours_e, px_e, shed_e),
            metrics_h=_metrics(ours_h, px_h, shed_h),
            plot_e=_plot_prices(hours, ours_e, px_e, f"{z} Electricity Price: Ours vs. PLEXOS", "EUR/MWh"),
            plot_h=_plot_prices(hours, ours_h, px_h, f"{z} Hydrogen Price: Ours vs. PLEXOS", "EUR/MWhH2"),
            plot_mix=_plot_tech_share(
                f"{z} — Technology share, {start_date_str} to {end_date_str}",
                tech_mix.get(z, pd.Series(dtype=float))),
        ))

    return render(request, "dispatcher/results.html", {
        "country_groups": _country_groups(),
        "year": 2030,
        "requested_zones": requested_zones,
        "auto_added": auto_added,
        "zones": cfg.zones,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "start_day": start_day,
        "end_day": end_day,
        "zone_results": zone_results,
    })


def _parse_capacity_overrides(post, zones: set[str]) -> dict[str, dict[str, float]]:
    """Pull ``cap__<zone>__<tech>`` fields from the submitted form into
    ``{zone: {tech: value}}`` (MW power capacities and MWh storage energy
    capacities alike -- `tech` carries its own unit suffix). Only fields for
    zones actually in this run are kept; unparseable values are dropped (the
    run falls back to that tech's own workbook default rather than failing
    outright)."""
    overrides: dict[str, dict[str, float]] = {}
    for key, value in post.items():
        if not key.startswith("cap__"):
            continue
        _, _, rest = key.partition("__")
        zone, _, tech = rest.partition("__")
        if zone not in zones:
            continue
        try:
            cap = float(value)
        except (TypeError, ValueError):
            continue
        overrides.setdefault(zone, {})[tech] = cap
    return overrides


def _metrics(ours: np.ndarray, plexos: np.ndarray | None, shed: np.ndarray) -> dict | None:
    """corr_excl/rmse_excl/mean_diff/corr_all (scarcity-hour filter at
    SCARCITY_THRESHOLD) plus real shedding hours -- the same battery of
    validation metrics used throughout this project's PLEXOS comparisons."""
    real_shed_hours = int((shed > 1e-9).sum())
    if plexos is None:
        return dict(corr_excl=None, rmse_excl=None, mean_diff=None, corr_all=None,
                   n_hi_ours=None, n_hi_plexos=None, real_shed_hours=real_shed_hours)
    ns = (ours < SCARCITY_THRESHOLD) & (plexos < SCARCITY_THRESHOLD)
    n = int(ns.sum())
    return dict(
        corr_excl=float(np.corrcoef(ours[ns], plexos[ns])[0, 1]) if n > 1 else None,
        rmse_excl=float(np.sqrt(np.mean((ours[ns] - plexos[ns]) ** 2))) if n > 1 else None,
        mean_diff=float(np.mean(ours[ns] - plexos[ns])) if n > 1 else None,
        corr_all=float(np.corrcoef(ours, plexos)[0, 1]) if len(ours) > 1 else None,
        n_hi_ours=int((ours >= SCARCITY_THRESHOLD).sum()),
        n_hi_plexos=int((plexos >= SCARCITY_THRESHOLD).sum()),
        real_shed_hours=real_shed_hours,
    )


def _plot_prices(hours: np.ndarray, ours: np.ndarray, plexos: np.ndarray | None, title: str,
                 yaxis_title: str = "EUR/MWh") -> str:
    """Interactive two-line Plotly chart: blue = ours, red = PLEXOS. Returns
    an HTML <div> fragment for direct embedding (plotly.js itself is loaded
    once via CDN in the base template, not repeated per chart)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=ours, mode="lines", name="Ours",
                             line=dict(color="#2563eb", width=1.8)))
    if plexos is not None:
        fig.add_trace(go.Scatter(x=hours, y=plexos, mode="lines", name="PLEXOS",
                                 line=dict(color="#dc2626", width=1.5, dash="dot")))
    fig.update_layout(
        title=title, xaxis_title="Hour of year", yaxis_title=yaxis_title,
        template="plotly_white", height=320, autosize=True,
        margin=dict(l=55, r=20, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})


def _plot_tech_share(title: str, mix: pd.Series) -> str | None:
    """Donut chart of each technology's share of a zone's total electricity
    supplied over the run's horizon (a `report.zone_technology_mix` entry,
    raw keys already sorted largest-first). Returns None if the zone
    generated nothing over the horizon (e.g. fully import-supplied), so the
    caller can skip the panel instead of rendering an empty chart."""
    if mix.empty:
        return None
    labels = [_generation_label(k) for k in mix.index]
    fig = go.Figure(data=[go.Pie(labels=labels, values=mix.to_numpy(), hole=0.35, sort=False)])
    fig.update_layout(
        title=title, template="plotly_white", height=380, autosize=True,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="v", yanchor="middle", y=0.5),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})
