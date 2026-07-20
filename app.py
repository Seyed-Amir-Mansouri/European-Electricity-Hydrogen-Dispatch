"""Web UI for the CORE electricity + hydrogen dispatch model.

Run locally:
    "../projects-venv/Scripts/streamlit.exe" run app.py
"""
from __future__ import annotations

import altair as alt
import streamlit as st

from economic_dispatch.config import RunConfig
from economic_dispatch import data_loader, pipeline, report

st.set_page_config(page_title="CORE Electricity + Hydrogen Dispatch", layout="wide")

# Categories in the electricity table that are NOT generation technologies.
_NON_GEN = {"Storage discharge", "Storage charge (-)", "Electrolyser load (-)",
            "Net line import", "External exchange", "Load shedding",
            "Dumped/curtailed (-)", "Demand (-)", "Marginal Price (EUR/MWh)"}
_PRICE = "Marginal Price (EUR/MWh)"


@st.cache_resource(show_spinner=False)
def available_zones() -> list[str]:
    return data_loader.zones_in_db(RunConfig().zones_db)


@st.cache_data(show_spinner=False)
def run_scenario(zones, start, end, storage, ramps, reserves, h2term):
    """Solve one scenario and return (tables, ok, zones)."""
    cfg = RunConfig(start_day=start, end_day=end)
    cfg.zones = list(zones)
    cfg.enable_storage = storage
    cfg.enable_ramps = ramps
    cfg.enable_reserves = reserves
    cfg.enable_h2_terminal = h2term
    build = pipeline.solve_scenario(cfg)
    val = report.validate(build)
    ok = val["max_elec_residual"] < val["tol"] and val["max_h2_residual"] < val["tol"]
    return report.hourly_balance_tables(build), ok, list(cfg.zones)


# --------------------------------------------------------------------------- #
# Sidebar — the scenario controls (the CLI flags as widgets)
# --------------------------------------------------------------------------- #
st.sidebar.title("Scenario")
zones_all = available_zones()
zones = st.sidebar.multiselect("Zones", zones_all, default=zones_all)

horizon = st.sidebar.radio("Horizon", ["Single day", "Day range"], horizontal=True)
if horizon == "Single day":
    day = st.sidebar.slider("Day of year", 1, 364, 1)
    start, end = day, day
else:
    start, end = st.sidebar.slider("Day range", 1, 364, (1, 7))

st.sidebar.markdown("**Options**")
storage = st.sidebar.checkbox("Storage", True)
ramps = st.sidebar.checkbox("Ramp limits", True)
reserves = st.sidebar.checkbox("Reserves (FCR/FRR)", False)
h2term = st.sidebar.checkbox("H2 terminal imports", True)
go = st.sidebar.button("Run dispatch", type="primary", use_container_width=True)

st.title("CORE Electricity + Hydrogen Dispatch")
st.caption("ENTSO-E TYNDP NT2030 · CORE-region zones · linear dispatch (linopy + HiGHS)")

if go:
    if not zones:
        st.error("Select at least one zone.")
    else:
        with st.spinner("Building & solving the LP…"):
            st.session_state.result = run_scenario(
                tuple(sorted(zones)), start, end,
                storage, ramps, reserves, h2term)

if "result" not in st.session_state:
    st.info("Set the scenario in the sidebar and click **Run dispatch**.")
    st.stop()

tables, ok, solved_zones = st.session_state.result
elec, h2 = tables["elec"], tables["h2"]
(st.success if ok else st.error)(
    f"Solved {len(solved_zones)} zones · balances {'closed' if ok else 'FAILED'}")

tab_gen, tab_price, tab_tables = st.tabs(
    ["Generation stack", "Marginal prices", "Balance tables"])

# ---- Per-zone generation stack -------------------------------------------- #
with tab_gen:
    zone = st.selectbox("Zone", solved_zones, key="gen_zone")
    zd = elec[zone]
    gen_cols = [c for c in zd.columns if c not in _NON_GEN and zd[c].abs().sum() > 1e-6]
    if gen_cols:
        long = zd[gen_cols].reset_index().melt("hour", var_name="Technology", value_name="MW")
        area = alt.Chart(long).mark_area().encode(
            x=alt.X("hour:Q", title="Hour"),
            y=alt.Y("MW:Q", stack="zero", title="Generation (MW)"),
            color=alt.Color("Technology:N", legend=alt.Legend(columns=2)),
            tooltip=["hour:Q", "Technology:N", alt.Tooltip("MW:Q", format=",.0f")])
        dem = zd["Demand (-)"].abs().rename("MW").reset_index()
        line = alt.Chart(dem).mark_line(color="black", strokeDash=[4, 2]).encode(
            x="hour:Q", y="MW:Q", tooltip=[alt.Tooltip("MW:Q", format=",.0f", title="Demand")])
        st.altair_chart(area + line, use_container_width=True)
        st.caption("Stacked hourly generation by technology; dashed line = electricity demand.")
    else:
        st.write("No generation in this zone.")

# ---- Marginal prices ------------------------------------------------------ #
with tab_price:
    st.subheader("Electricity price (EUR/MWh)")
    st.line_chart(elec.xs(_PRICE, axis=1, level=1))
    st.subheader("Hydrogen price (EUR/MWh)")
    st.line_chart(h2.xs(_PRICE, axis=1, level=1))

# ---- Tables + downloads --------------------------------------------------- #
with tab_tables:
    st.subheader("Electricity balance")
    st.dataframe(elec, use_container_width=True, height=280)
    st.download_button("⬇ hourly_balance_elec.csv", elec.to_csv().encode(),
                       "hourly_balance_elec.csv", "text/csv")
    st.subheader("Hydrogen balance")
    st.dataframe(h2, use_container_width=True, height=280)
    st.download_button("⬇ hourly_balance_h2.csv", h2.to_csv().encode(),
                       "hourly_balance_h2.csv", "text/csv")
