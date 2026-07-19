"""Solve the built model with HiGHS and expose the solution."""
from __future__ import annotations

import warnings

from .model import BuildResult

# linopy bundles ALL variables/constraints into one dataset when it hands the
# problem to the solver and when it reads the solution back. Because this model
# has variables on genuinely different dimensions (gen / zone / sto / line), the
# exact coordinate align inside linopy's save_join necessarily fails and it
# falls back to a (harmless, no-op) outer join, emitting this UserWarning ~11x.
# The balance check in report.validate() confirms nothing is misaligned, so we
# silence this one specific, benign message. Persisting the filter here (rather
# than a context manager) also covers the solution access done later in report.
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
