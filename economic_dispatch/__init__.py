"""Coupled electricity + hydrogen economic dispatch (linear program).

A single-day (24 h) dispatch model over ENTSO-E-style bidding zones, built on
linopy + HiGHS. See README.md for the data model and assumptions.
"""

__all__ = ["config", "data_loader", "network_loader", "model", "solve", "report"]
