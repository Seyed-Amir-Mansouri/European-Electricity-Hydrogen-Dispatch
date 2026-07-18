"""Compute fixed cross-border exchange from the NT2030 result databases.

Instead of reading the pre-baked ``Exports_*`` / ``H2Exports_*`` columns from the
per-zone Excel files (which assume a FIXED zone selection), this derives the
external exchange on the fly from the crossborder-flow parquet databases in
``inputs/`` for the ACTUAL selected zones: a border is treated as an internal
(optimised) line when both endpoints are in the selection, or as a fixed
external exchange when the neighbour is outside it.

Full specification: ``inputs/EXPORTS_CALCULATION.md``.

Sign convention (as in the guide): a produced series is a NET EXPORT, positive =
export out of the zone. The model wants a net INJECTION (import positive), so the
public entry point returns ``-net_export``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ZONE = "zone"
HOUR = "hour"


def country(zone: str) -> str:
    """Country code = first two characters of a zone code (AT00->AT, BEOF->BE)."""
    return zone[:2]


def _flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and "->" in c]


def _num(series: pd.Series) -> np.ndarray:
    return np.nan_to_num(series.to_numpy(dtype=float))


def elec_net_export(selected: list[str], edf: pd.DataFrame) -> dict[str, np.ndarray]:
    """Net electricity export (MW, + = export) per selected zone, full-year arrays.

    For each directed flow ``A->B``: if A is selected and B is not, it is an
    export of A (+); if B is selected and A is not, an import of B (-). Flows
    between two selected zones are internal (skipped).
    """
    sel = set(selected)
    out = {z: np.zeros(len(edf)) for z in selected}
    for f in _flow_cols(edf):
        a, b = f.split("->")
        if a in sel and b not in sel:
            out[a] = out[a] + _num(edf[f])
        elif b in sel and a not in sel:
            out[b] = out[b] - _num(edf[f])
    return out


def _cc(node: str) -> str:
    """Country code of an H2 node header (strip the ``_H2`` suffix)."""
    return node[:-3] if node.endswith("_H2") else node


def h2_net_export(selected: list[str], main_map: dict[str, str],
                  hdf: pd.DataFrame, smr: pd.DataFrame) -> dict[str, np.ndarray]:
    """Net hydrogen export (MW, + = export) per country's main zone, full-year.

    Resolves ``IB*`` interconnector hubs, classifies each edge by the §2 sign
    rule keeping only edges whose other endpoint's country is outside the
    selection, then folds Steam-Methane-Reformer output in as imported H2.
    """
    sel_c = {country(z) for z in selected}
    flows = _flow_cols(hdf)

    # 4a. Resolve IB* hubs: A->hub + hub->B  =>  A->B (carrying the A->hub value).
    hub_sinks: dict[str, list[str]] = {}
    for f in flows:
        l, r = f.split("->")
        if l.startswith("IB") and l.endswith("_H2") and not (r.startswith("IB") and r.endswith("_H2")):
            hub_sinks.setdefault(l, []).append(r)
    edges: list[tuple[str, str, np.ndarray]] = []
    for f in flows:
        l, r = f.split("->")
        if l.startswith("IB") and l.endswith("_H2"):
            continue  # drop aggregate hub->sink edges
        arr = _num(hdf[f])
        if r.startswith("IB") and r.endswith("_H2"):
            for sink in hub_sinks.get(r, []):
                edges.append((l, sink, arr))
        else:
            edges.append((l, r, arr))

    out = {z: np.zeros(len(hdf)) for z in main_map.values()}
    for l, r, arr in edges:
        xc, yc = _cc(l), _cc(r)
        if xc in sel_c and yc not in sel_c:
            C, sign = xc, 1.0
        elif yc in sel_c and xc not in sel_c:
            C, sign = yc, -1.0
        else:
            continue
        M = main_map.get(C)
        if M is not None:
            out[M] = out[M] + sign * arr

    # 4c. SMR is domestic H2 supply -> a negative export at the main zone.
    for C in sel_c:
        M = main_map.get(C)
        if M is not None and C in smr.columns:
            out[M] = out[M] - _num(smr[C])
    return out


def load_external_injection(cfg, zones: list[str], hours: pd.Index,
                            main_map: dict[str, str]) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (external_e, external_h2) net-injection DataArrays (import +).

    dims ``(zone, hour)`` sliced to the model's hour window. Reads the three
    parquet databases from ``cfg.exports_dir``; raises ``FileNotFoundError`` if
    any is missing so the caller can fall back to the Excel columns.
    """
    d = Path(cfg.exports_dir)
    edf = pd.read_parquet(d / "crossborder_electricity_2030.parquet")
    hdf = pd.read_parquet(d / "crossborder_hydrogen_2030.parquet")
    smr = pd.read_parquet(d / "smr_production_2030.parquet")

    h0, h1 = cfg.hour_slice()
    e_exp = elec_net_export(zones, edf)
    h_exp = h2_net_export(zones, main_map, hdf, smr)
    zidx = pd.Index(zones, name=ZONE)

    def to_injection(exp: dict[str, np.ndarray], n: int) -> xr.DataArray:
        rows = [-exp.get(z, np.zeros(n))[h0:h1] for z in zones]  # injection = -export
        return xr.DataArray(np.vstack(rows), coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])

    return to_injection(e_exp, len(edf)), to_injection(h_exp, len(hdf))
