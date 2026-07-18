"""Parse Networks.xlsx into directed transport lines + global price scalars.

Each network sheet has two side-by-side blocks:
  cols A-D : From, To, Length (km), Loss Fraction   -> distance/loss per pair
  cols F-I : From, To, From-To Capacity, To-From Cap -> directional MW limits
The two blocks cover different (and larger) sets of pairs, so we treat the
capacity block as the authoritative line list and look up losses by unordered
{From, To} pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl

SHEET_ELEC = "Electricity Lines"
SHEET_H2 = "Hydrogen Pipelines"
SHEET_DATA = "Data"


@dataclass
class Line:
    frm: str
    to: str
    cap_ft: float   # MW capacity in the from->to direction
    cap_tf: float   # MW capacity in the to->from direction
    loss: float     # fractional loss on the line (0-1)


@dataclass
class NetworkData:
    elec: list[Line]
    hydrogen: list[Line]
    co2_price: float      # EUR/ton
    gas_price: float      # EUR/MWh


def _loss_lookup(ws) -> dict[frozenset, float]:
    """Build {frozenset({From, To}) -> loss fraction} from the left block."""
    out: dict[frozenset, float] = {}
    for r in range(2, ws.max_row + 1):
        frm, to, _length, loss = (ws.cell(r, c).value for c in (1, 2, 3, 4))
        if isinstance(frm, str) and isinstance(to, str):
            try:
                out[frozenset({frm, to})] = float(loss) if loss is not None else 0.0
            except (TypeError, ValueError):
                out[frozenset({frm, to})] = 0.0
    return out


def _read_lines(ws, zones: set[str]) -> list[Line]:
    losses = _loss_lookup(ws)
    lines: list[Line] = []
    for r in range(2, ws.max_row + 1):
        frm, to, cap_ft, cap_tf = (ws.cell(r, c).value for c in (6, 7, 8, 9))
        if not (isinstance(frm, str) and isinstance(to, str)):
            continue
        if frm not in zones or to not in zones or frm == to:
            continue  # keep only internal lines between active modelled zones
        try:
            ft = float(cap_ft) if cap_ft is not None else 0.0
            tf = float(cap_tf) if cap_tf is not None else 0.0
        except (TypeError, ValueError):
            ft = tf = 0.0
        loss = losses.get(frozenset({frm, to}), 0.0)
        lines.append(Line(frm, to, ft, tf, loss))
    return lines


def load_networks(data_dir: Path, zones: list[str]) -> NetworkData:
    path = Path(data_dir) / "Networks.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Networks workbook not found: {path}")
    zset = set(zones)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    elec = _read_lines(wb[SHEET_ELEC], zset)
    hydrogen = _read_lines(wb[SHEET_H2], zset)

    ws = wb[SHEET_DATA]
    co2_price = float(ws.cell(2, 1).value or 0.0)
    gas_price = float(ws.cell(2, 2).value or 0.0)

    wb.close()
    return NetworkData(elec, hydrogen, co2_price, gas_price)
