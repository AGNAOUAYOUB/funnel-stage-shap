"""Funnel-Stage SHAP framework for sequential purchase prediction.

Implements the pre-registered protocol "Dynamic Explainable AI for Sequential
Purchase Prediction: A Funnel-Stage SHAP Framework for Digital Consumer
Journeys" (v1.0). Section numbers in docstrings refer to that protocol.
"""

from __future__ import annotations

import contextlib
import sys

__version__ = "0.1.0"


def _preload_lightgbm_before_sklearn() -> None:
    """Import LightGBM first, on Windows, to avoid an OpenMP runtime conflict.

    scikit-learn ships its own ``sklearn/.libs/vcomp140.dll`` for machines
    without the Visual C++ redistributable, and loads it by *absolute path*.
    LightGBM's ``lib_lightgbm.dll`` links against the system ``vcomp140.dll``.
    When scikit-learn wins the race, the two OpenMP runtimes coexist in the
    process and the first LightGBM ``fit`` dies with

        OSError: exception: access violation reading 0x0000000000000000

    inside ``LGBM_DatasetSetField`` -- a native crash with no Python-level
    cause, which is why it is worth a note rather than a shrug. Importing
    LightGBM first makes it bind to the system runtime; scikit-learn's copy then
    loads harmlessly alongside.

    Preloading ``vcomp140.dll`` by name via ctypes does *not* work: the
    absolute-path load still creates a second module. Import order is the fix.

    Failures are swallowed. LightGBM is one of five baselines (Sec. 9.1) and the
    rest of the package must remain usable without it.
    """
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        import lightgbm  # noqa: F401


_preload_lightgbm_before_sklearn()
