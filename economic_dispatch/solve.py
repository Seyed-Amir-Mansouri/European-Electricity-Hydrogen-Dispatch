"""Solve the built model with HiGHS and expose the solution."""
from __future__ import annotations

import warnings

from .model import BuildResult

warnings.filterwarnings(
    "ignore",
    message="Coordinates across variables not equal.*",
    category=UserWarning,
    module="linopy.*",
)


def solve(build: BuildResult) -> str:
    cfg = build.cfg
    status, condition = build.model.solve(
        solver_name=cfg.solver_name,
        mip_rel_gap=cfg.mip_rel_gap,
    )
    print(f"Solver status: {status} ({condition})")
    return status
