"""Consolidate every zone workbook into a single Parquet database.

Reads all zone Excel files (via ``data_loader``) and writes
``inputs/zones_2030.parquet`` in a long, lossless format:

    zone | section | item | hour | value_num | value_str

* scalar sheets  -> section in {capacities, storage_energy, reserves, gas_h2},
                    item = parameter,  hour = -1
* characteristics -> section = characteristics, item = "<Technology>||<attribute>",
                    hour = -1   (numeric -> value_num, text -> value_str)
* hourly profiles -> section = profiles, item = <column>, hour = 0..8735

Usage:  python build_zones_db.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from economic_dispatch import data_loader as dl
from economic_dispatch.config import DEFAULT_DATA_DIR, DEFAULT_EXPORTS_DIR, discover_zones, HOURS_PER_YEAR

COLS = ["zone", "section", "item", "hour", "value_num", "value_str"]


def _coerce(v):
    """Return (value_num, value_str): numbers in value_num, text in value_str."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.nan, None
    try:
        return float(v), None
    except (TypeError, ValueError):
        return np.nan, str(v)


def _scalar_frame(zone, section, d):
    return pd.DataFrame({
        "zone": zone, "section": section, "item": list(d.keys()),
        "hour": -1, "value_num": list(d.values()), "value_str": None,
    })


def build(data_dir=DEFAULT_DATA_DIR, out=None):
    out = out or (DEFAULT_EXPORTS_DIR / "zones_2030.parquet")
    zones = discover_zones(data_dir)
    frames = []
    for z in zones:
        zd = dl.load_zone(z, data_dir, 0, HOURS_PER_YEAR)
        for section, d in [("capacities", zd.capacities), ("storage_energy", zd.storage_energy),
                           ("reserves", zd.reserves), ("gas_h2", zd.gas_h2)]:
            if d:
                frames.append(_scalar_frame(z, section, d))

        ch = zd.char.reset_index()
        idcol = ch.columns[0]
        m = ch.melt(id_vars=idcol, var_name="attr", value_name="val")
        nums, strs = zip(*(_coerce(v) for v in m["val"])) if len(m) else ((), ())
        frames.append(pd.DataFrame({
            "zone": z, "section": "characteristics",
            "item": m[idcol].astype(str) + "||" + m["attr"].astype(str),
            "hour": -1, "value_num": list(nums), "value_str": list(strs),
        }))

        prof = zd.profiles.drop(columns=[c for c in zd.profiles.columns if c == "Hour"], errors="ignore")
        prof.index.name = "hour"
        pm = prof.reset_index().melt(id_vars="hour", var_name="item", value_name="value_num")
        pm["value_num"] = pd.to_numeric(pm["value_num"], errors="coerce")
        pm["zone"] = z
        pm["section"] = "profiles"
        pm["value_str"] = None
        frames.append(pm)
        print(f"  {z}: {len(zd.capacities)} caps, {zd.char.shape[0]} techs, "
              f"{zd.profiles.shape[0]}x{zd.profiles.shape[1]} profiles")

    db = pd.concat(frames, ignore_index=True)[COLS]
    db["hour"] = db["hour"].astype("int32")
    out.parent.mkdir(parents=True, exist_ok=True)
    db.to_parquet(out, index=False, compression="zstd")
    print(f"\nWrote {out}  ({len(db):,} rows, {out.stat().st_size/1e6:.1f} MB, {len(zones)} zones)")
    return out


if __name__ == "__main__":
    build()
