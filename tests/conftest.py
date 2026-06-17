from __future__ import annotations

import os
import sys
from contextlib import suppress

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_MAX_THREADS",
):
    os.environ.setdefault(_name, "1")


def pytest_configure() -> None:
    """Keep native numerical libraries bounded during the default test suite."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    with suppress(RuntimeError):
        torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
