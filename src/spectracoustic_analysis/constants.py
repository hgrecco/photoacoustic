from pathlib import Path
from typing import Callable, Literal, TypedDict

import numpy as np
import numpy.typing as npt

OPTIONS_TO_PRINT = [
    "plot_time_trace_rep",
    "plot_with_intercept",
    "plot_uncertainty_slope",
    "plot_uncertainty_slope0",
    "max_energy",
    "alpha_ref",
    "peak_threshold_factor",
    "savgol_window_length",
    "savgol_polyorder",
    "pa_signal",
]
_footnote_timestamp: str | None = None
__version__ = "2025.02.12"

Array = npt.NDArray[np.float64]

_footnote_timestamp: str | None = None
__version__ = "2025.02.12"


PaSignal = Literal["signal_delta", "signal_peak1", "signal_peak2", "sonic_energy"]

OnErrorFunc = Callable[[str], None]

on_error_default: OnErrorFunc = lambda e: None  # noqa


class LinearPlotOptions(TypedDict):
    on_error: OnErrorFunc
    plot_with_intercept: bool
    plot_uncertainty_slope: bool
    plot_uncertainty_slope0: bool


class Options(LinearPlotOptions):
    savgol_window_length: int
    savgol_polyorder: int
    on_progress: Callable[
        [
            str,
        ],
        None,
    ]
    plot_time_trace_rep: bool
    trace_to_include: dict[tuple[str, int], bool]
    pa_signal: PaSignal
    max_energy: float
    alpha_ref: float
    peak_threshold_factor: float
    figures_save_path: Path


def default_options() -> Options:
    return {
        "savgol_window_length": 51,
        "savgol_polyorder": 3,
        "on_progress": print,
        "on_error": print,
        "plot_time_trace_rep": True,
        "trace_to_include": {},
        "pa_signal": "signal_delta",
        "plot_with_intercept": True,
        "plot_uncertainty_slope": False,
        "plot_uncertainty_slope0": True,
        "max_energy": 20.0,
        "alpha_ref": 1.0,
        "peak_threshold_factor": 2,
        "figures_save_path": Path("_figures"),
    }


OPTIONS = default_options()
