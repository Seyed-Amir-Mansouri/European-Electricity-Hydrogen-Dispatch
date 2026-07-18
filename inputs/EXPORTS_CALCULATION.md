# Cross-border Exports — Data & Calculation Guide (NT2030)

This folder contains the export-relevant data from the market-model result file
`MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx`, converted to Parquet,
plus this note explaining **how electricity and hydrogen exports are derived**
from it. It is meant to let another project reproduce the export figures without
the original Excel file or the pipeline code.

All flows are hourly, `8736` rows (one climate year, CY2009), aligned by row order
(`Hour` 1..8736 / `Date`).

---

## 1. Files

| File | Rows × Cols | Contents |
|---|---|---|
| `crossborder_electricity_2030.parquet` | 8736 × 158 | Raw electricity exchange flows |
| `crossborder_hydrogen_2030.parquet` | 8736 × 50 | Raw hydrogen exchange flows |
| `smr_production_2030.parquet` | 8736 × 33 | Steam-methane-reformer H2 production per country |

### Schemas
- **electricity**: `Hour` (int), `Date` (str, e.g. `01JAN00:00`), then one column per
  directed flow named `A->B` where `A`, `B` are electricity zone codes (e.g.
  `DE00->PL00`). Value = MW flowing from `A` to `B` (can be negative = reverse).
- **hydrogen**: `Hour`, `Date`, then one column per directed flow named `A_H2->B_H2`
  (country H2 nodes, e.g. `DE_H2->PL_H2`), plus interconnector-hub nodes `IB*_H2`
  and external source nodes (`XDZ`, `XMA`, `XNO`, `XUA`, `XAmmonia`). Value = MW of H2.
- **smr**: `Date`, then one column per country code (e.g. `DE`, `FR`) = SMR H2
  production in MW (`Steam methane reformer [MWH2]`).

> Column headers in the raw files use the market model's node codes. The
> calculation below maps them to study-zone codes.

---

## 2. Sign convention (both carriers)

For a **zone/country of interest** `Z`:

- A flow **`Z->N`** (Z is the *start* node) is an **export**, kept **positive**.
- A flow **`N->Z`** (Z is the *end* node) is an **import**, kept **negated** (multiply by −1).

So every produced column is an *export* series: **positive = net export out of Z**,
**negative = net import into Z**. Only flows to a **non-selected neighbour** are kept
(a flow between two in-scope zones is an internal line, not an export).

---

## 3. Electricity exports

Source: `crossborder_electricity_2030.parquet` (sheet `Crossborder exchanges`).

For each selected zone `Z` and each flow header `A->B`:

1. If `A == Z` → export, `node = B`, sign `+`.
2. If `B == Z` → import, `node = A`, sign `−`.
3. Otherwise skip.
4. Only keep it if `node` is **outside** the selection.

**Output column naming** `Exports_<Z>_<node> (MW/h)`, with one special rule:

- **External source nodes** — codes **starting with `X`** (e.g. `XRU00`, `XSA00`,
  `XTN00`, `XMD00`, `XBACE`) — are **summed** into a single column
  **`Exports_<Z>_XX (MW/h)`**.

**Net export of a zone** (what an energy model puts on a `<Z>_Exp` node) =
sum of *all* `Exports_<Z>_*` columns (including `_XX`) for that hour.

```python
import pandas as pd
e = pd.read_parquet("crossborder_electricity_2030.parquet")
Z = "DE00"
cols = {}
for h in [c for c in e.columns if "->" in c]:
    a, b = h.split("->")
    if a == Z:   node, sign = b, +1
    elif b == Z: node, sign = a, -1
    else:        continue
    name = "XX" if node.startswith("X") else node        # aggregate X sources
    cols[name] = cols.get(name, 0) + sign * e[h].fillna(0)
exports = pd.DataFrame(cols)                              # Exports_DE00_<node>
net_export = exports.sum(axis=1)                          # -> DE00_Exp
```

---

## 4. Hydrogen exports

Source: `crossborder_hydrogen_2030.parquet` (sheet `Crossborder H2 exchanges`) plus
`smr_production_2030.parquet` (sheet `Hourly H2 Data`).

Headers are **country** H2 nodes `XX_H2` (e.g. `DE_H2`), **interconnector hubs**
`IB*_H2` (e.g. `IBIT_H2`, `IBFI_H2`), and **external sources** (`XDZ`, `XMA`, `XNO`,
`XUA`, `XAmmonia`). Country code = header without the `_H2` suffix.

The hydrogen network has **one node per country** ("main zone"); all of a country's
H2 flows attach to that main zone (e.g. `DE_H2` → `DE00`, `IT_H2` → `ITCA`). Supply a
country→main-zone map (in this pipeline it comes from the `Lines_H` network sheet).

Processing order:

### 4a. Resolve `IB*` interconnector hubs (do this first)
An `IB*` hub passes H2 between a source side (`A->hub`) and a sink side (`hub->B`).
Replace each `A->hub` segment with a direct **`A->B`** edge **carrying the `A->hub`
value**, for every sink `B`; **drop** the aggregate `hub->B` edges.

> Example: `AT_H2->IBIT_H2` together with `IBIT_H2->IT_H2` ⇒ treat as `AT_H2->IT_H2`.

### 4b. Classify each (resolved) edge for a selected country `C` (main zone `M`)
Using the sign convention of §2 (`C` start → `+`, `C` end → `−`), and keeping only
edges whose other endpoint is outside the selection. Output naming:

- **Country neighbour** (`YY_H2`) → **`H2Exports_<M>_<YY>00 (MW/h)`**.
- **External source** (code starts with `X`) → **summed** into **`H2Exports_<M>_XX (MW/h)`**.

### 4c. Fold in Steam Methane Reformer as imported hydrogen
For each country, take its SMR series from `smr_production_2030.parquet` and
**subtract** it from `H2Exports_<M>_XX` (a supply is a negative export). If a country
has SMR but no external `X` sources, create `H2Exports_<M>_XX = −SMR`.

### 4d. Net H2 export
`<M>_Exp` node value = sum of *all* `H2Exports_<M>_*` columns for that hour.

```python
import pandas as pd
h   = pd.read_parquet("crossborder_hydrogen_2030.parquet")
smr = pd.read_parquet("smr_production_2030.parquet")
sel = {"DE"}                       # selected country codes
main = {"DE": "DE00"}              # country -> main H2 zone (from Lines_H)

def cc(n): return n[:-3] if n.endswith("_H2") else n
flows = [c for c in h.columns if "->" in c]

# 4a: hub sinks
hub_sinks = {}
for f in flows:
    l, r = f.split("->")
    if l.startswith("IB") and l.endswith("_H2") and not (r.startswith("IB") and r.endswith("_H2")):
        hub_sinks.setdefault(l, []).append(r)

edges = []                          # (left, right, series)
for f in flows:
    l, r = f.split("->")
    if l.startswith("IB") and l.endswith("_H2"):
        continue                    # drop aggregate hub->sink
    if r.startswith("IB") and r.endswith("_H2"):
        for sink in hub_sinks.get(r, []):
            edges.append((l, sink, h[f]))
    else:
        edges.append((l, r, h[f]))

out = {}
for l, r, series in edges:
    xc, yc = cc(l), cc(r)
    if xc in sel and yc not in sel:   C, neigh, sign = xc, r, +1
    elif yc in sel and xc not in sel: C, neigh, sign = yc, l, -1
    else: continue
    M = main[C]
    key = f"H2Exports_{M}_XX" if cc(neigh).startswith("X") or neigh.startswith("X") \
          else f"H2Exports_{M}_{cc(neigh)}00"
    out[key] = out.get(key, 0) + sign * series.fillna(0)

# 4c: subtract SMR into _XX
for C in sel:
    M = main[C]
    if C in smr.columns:
        key = f"H2Exports_{M}_XX"
        out[key] = out.get(key, 0) - smr[C].fillna(0)

h2 = pd.DataFrame(out)
net_h2_export = h2.sum(axis=1)      # -> <M>_Exp
```

---

## 5. Notes & caveats

- **Units** are MW (of electricity, or of H2) as reported by the market model.
  Downstream energy-system models may rescale (e.g. this pipeline divides H2 by
  1000 to GW when writing openTEPES files) — the Parquet keeps the raw MW.
- **Main-zone map** is required for hydrogen and is model-specific; here it is the
  country's node in the `Lines_H` network (the selected endpoint, else the
  country's first selected zone).
- **`_XX`** always means "aggregate of all external (non-modelled) sources", and for
  hydrogen additionally includes the subtracted SMR production.
- Rows align by position (`Hour`/`Date`); there is no timezone handling — treat the
  index as the model's native hour ordering for CY2009.
- Scenario is **NT2030**; the same layout applies to other `MMStandardOutputFile_NT<year>`
  files (only the values, and the SMR magnitudes, differ).
