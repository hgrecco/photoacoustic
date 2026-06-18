from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
from scipy import odr
from scipy.signal import find_peaks, savgol_filter
from uncertainties.core import Variable, ufloat

from .constants import Array, Options
from .models import (
    Experiment,
    PowerScan,
    PowerscanAnalysis,
    TraceAnalysis,
    TraceMetadata,
)

# from .constants import OPTIONS

UFLOAT0 = ufloat(0, 0)
UFLOAT_NAN = ufloat(np.nan, np.nan)


class AlphaRecords(TypedDict):
    abs_sam: float
    abs_ref: float
    ref: str
    sam: str
    exc_wavelength: float
    alpha: float
    alpha_unc: float
    alpha0: float
    alpha0_unc: float


def ufloat_nanmean(*variables: Variable) -> Variable:
    """Calculate new uncertainty by averaging an iterable of uncertainties."""
    els = [
        el
        for el in variables
        if np.isfinite(el.nominal_value) and np.isfinite(el.std_dev)
    ]

    N = len(els)
    if N == 0:
        return UFLOAT_NAN

    mx = np.asarray([el.nominal_value for el in els])
    sx = np.asarray([el.std_dev for el in els])

    if np.all(sx == 0):
        return ufloat(mx.mean(), mx.std())

    wx = 1 / sx / sx

    return ufloat((mx * wx).sum() / wx.sum(), np.sqrt(1 / wx.sum()))


def to_unc_tuple(
    nominal_values: Iterable[float], std_devs: Iterable[float]
) -> tuple[Variable, ...]:
    """Zip iterable of nominal values and std devs into a tuple of uncertainties."""
    return tuple(ufloat(v, u) for v, u in zip(nominal_values, std_devs))


def _fix_fit_unc(unc: Array):
    if not np.any(unc == 0):
        return unc

    if np.all(unc == 0):
        return None
    else:
        unc = np.copy(unc)
        unc[unc == 0] = unc[unc > 0].min() / 10
        return unc


def fit_linear(
    x: Iterable[float],
    y: Iterable[float],
    x_unc: Iterable[float],
    y_unc: Iterable[float],
    intercept0: bool = False,
) -> tuple[tuple[Variable, Variable], odr.Output]:
    """Fit linear and return the slope and intercept (as value with uncertainty)."""

    x = np.asarray(x)
    y = np.asarray(y)
    x_unc = np.asarray(x_unc)
    y_unc = np.asarray(y_unc)

    data = odr.RealData(
        x,
        y,
        sx=_fix_fit_unc(x_unc),
        sy=_fix_fit_unc(y_unc),
    )

    if intercept0:
        beta0 = [np.median(y_unc[x_unc > 0] / x_unc[x_unc > 0]), 0]
        result = odr.ODR(data, odr.unilinear, beta0=beta0, ifixb=[1, 0]).run()
    else:
        result = odr.ODR(data, odr.unilinear).run()

    return to_unc_tuple(result.beta, result.sd_beta), result


def _argmedian_at_t0(signals: list[tuple[Array, Array]]) -> int:
    """Returns the indices of the median value."""
    peak_signal: list[float] = [
        signal[np.searchsorted(time, 0) + 1] for time, signal in signals
    ]
    el: float = np.percentile(peak_signal, 50, method="closest_observation")  # type: ignore
    return peak_signal.index(el)


def split_unc_tuple(
    *variables: Variable, container: type = tuple
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Unzip iterable of uncertainties into a tuple of nominal value and a tuple of std dev"""
    return container(v.nominal_value for v in variables), container(
        v.std_dev for v in variables
    )


def analyze_powerscan(powerscan: PowerScan, options: Options) -> PowerscanAnalysis:
    """Analyze a powerscan folder to the slope and intercept
    of the delta signal vs energy.
    """

    measurement_files = powerscan.measurement_files
    folder = powerscan.path
    parts = folder.name.split("_", 2)
    if len(parts) == 3:
        sam_ref, wl, desc = parts  # TODO: maybe change sam_ref parameter
        try:
            wl = float(wl)
        except Exception:
            print(f"{folder}: wavelength must be a number")
            wl = np.nan
    else:
        print(f"{folder}: invalid")
        sam_ref, wl, desc = "N/A", np.nan, folder.stem

    energies: list[Variable] = []
    pa_signals: list[Variable] = []

    for filepath, measurement_file in measurement_files.items():
        _signals = []
        _pa_signals = []
        _energies = []
        for trace in measurement_file.traces:
            _signals.append((trace.time, trace.signal))
            _pa_signals.append(trace.get_analysis(options)[options["pa_signal"]])
            _energies.append(trace.get_analysis(options)["energy"])

        if len(_signals) == 0:
            continue
        elif len(_signals) > 1:
            _ndx = _argmedian_at_t0(_signals)
        else:
            _ndx = 0

        energies.append(ufloat_nanmean(*_energies))
        pa_signals.append(ufloat_nanmean(*_pa_signals))

    try:
        x, x_unc = split_unc_tuple(
            *energies, container=lambda el: np.fromiter(el, dtype=float)
        )
        y, y_unc = split_unc_tuple(
            *pa_signals, container=lambda el: np.fromiter(el, dtype=float)
        )

        valid = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
        if np.sum(valid) >= 2:
            (slope, intercept), result = fit_linear(
                x[valid], y[valid], x_unc[valid], y_unc[valid]
            )
            (slope0, _intercept0), result0 = fit_linear(
                x[valid], y[valid], x_unc[valid], y_unc[valid], intercept0=True
            )
        else:
            options["on_error"](
                f"Could not fit for {folder.stem}: not enough valid points"
            )
            slope = intercept = UFLOAT_NAN
            slope0 = _intercept0 = UFLOAT_NAN
            result, result0 = None, None

    except Exception as ex:
        options["on_error"](f"Could not fit for {folder.stem}: {str(ex)}")
        slope = intercept = UFLOAT_NAN
        slope0 = _intercept0 = UFLOAT_NAN
        result, result0 = None, None

    return PowerscanAnalysis(
        sam_ref=sam_ref,
        exc_wavelength=wl,
        description=desc,
        slope=slope,
        intercept=intercept,
        slope0=slope0,
        result=result,
        result0=result0,
        energies=energies,
        pa_signals=pa_signals,
    )


def find_first_two_peaks(
    time: Array,
    signal: Array,
    signal_smooth: Array | None,
    options: Options,
) -> list[tuple[Variable, Variable]]:
    """Find upto first two peaks.

    Iterable of Time, Signal
    """

    bg = signal[:500].mean()
    std = signal[:500].std()

    # MHz
    acq_frequency = 1 / np.diff(time)[0]
    # us
    time_distance = 1 / 2
    time_width = 1 / 4

    if signal_smooth is None:
        signal_smooth = savgol_filter(
            signal,
            options["savgol_window_length"],
            options["savgol_polyorder"],
        )

    out = []

    # Find positive peaks

    peak_threshold_factor = options["peak_threshold_factor"]

    ndxs, _props = find_peaks(
        -signal_smooth,
        height=-bg + peak_threshold_factor * std,
        prominence=peak_threshold_factor * std,
        distance=time_distance * acq_frequency,
        width=time_width * acq_frequency,
    )

    out.append(
        pd.DataFrame(
            dict(
                time=time[ndxs],
                time_unc=np.zeros_like(time[ndxs]),
                signal=signal_smooth[ndxs],
                signal_unc=np.zeros_like(time[ndxs]),
            )
        )
    )

    # Find negative peaks peaks

    ndxs, _props = find_peaks(
        signal_smooth,
        height=bg + peak_threshold_factor * std,
        prominence=peak_threshold_factor * std,
        distance=time_distance * acq_frequency,
        width=time_width * acq_frequency,
    )

    out.append(
        pd.DataFrame(
            dict(
                time=time[ndxs],
                time_unc=np.zeros_like(time[ndxs]),
                signal=signal_smooth[ndxs],
                signal_unc=np.zeros_like(time[ndxs]),
            )
        )
    )

    out = pd.concat(out).sort_values(by="time").reset_index(drop=True)

    if len(out) == 0:
        return []

    # Keep only the first two peaks: positive, followed by negative.

    delta = np.abs(out["signal"].values - bg)
    sel = delta > np.max(delta) / 20 + std
    out["sel1"] = sel
    out = out[sel].reset_index(drop=True)

    sign = np.sign(out["signal"].values - bg)
    sel = np.zeros(len(out), dtype=bool)
    best = np.where((sign == 1) & (np.roll(sign, -1) == -1))[0]

    if len(best) == 0:
        return []

    signal_best = out.iloc[best]["signal"].to_numpy()
    signal_best /= np.max(signal_best)
    # we consider that all peaks 30% smaller than the maximum
    signal_best[signal_best > 0.7] = 1

    best = best[np.argmax(signal_best)]

    sel[best] = True
    sel[best + 1] = True

    return [
        (
            ufloat(out.iloc[ndx]["time"], out.iloc[ndx]["time_unc"]),
            ufloat(out.iloc[ndx]["signal"], out.iloc[ndx]["signal_unc"]),
        )
        for ndx in (best, best + 1)
    ]


def analyze_time_trace(
    time: Array, signal: Array, options, alldf_attrs: TraceMetadata | None = None
) -> TraceAnalysis:
    """Find the first two peaks to obtain the time and signal delta."""

    if alldf_attrs is None:
        description = None
        wavelength = np.nan
        exc_wavelength = np.nan
    else:
        path = Path(alldf_attrs.PATH)
        sample, wl, *metadata = path.stem.split("_")
        description = alldf_attrs.Desc
        try:
            energy = alldf_attrs.Laser_energy_before
            wavelength = float(alldf_attrs.Wavelength)
            exc_wavelength = float(wl)
        except ValueError:
            options["on_error"]("Couldn't convert wavelength value to float")
            wavelength = np.nan
            exc_wavelength = np.nan

    signal_smooth: Array = savgol_filter(
        signal, options["savgol_window_length"], options["savgol_polyorder"]
    )

    peaks = find_first_two_peaks(time, signal, signal_smooth, options)

    # TODO: Just in case
    peaks.append((UFLOAT_NAN, UFLOAT_NAN))
    peaks.append((UFLOAT_NAN, UFLOAT_NAN))

    first_peak = peaks[0]
    second_peak = peaks[1]

    time_delta = first_peak[0] - second_peak[0]
    signal_delta = first_peak[1] - second_peak[1]

    ti = first_peak[0].nominal_value - 3
    tf = second_peak[0].nominal_value + 3
    dt = time[1] - time[0]
    time_filter = np.logical_and(time > ti, time < tf)
    # TBD: Add proper error propagation
    sonic_energy = ufloat(
        np.sum(np.abs(signal_smooth[time_filter]) * dt) / (tf - ti), 0
    )

    return TraceAnalysis(
        path="",
        repeat=None,
        include=all(
            map(
                np.isfinite,
                (
                    time_delta.nominal_value,
                    time_delta.std_dev,
                    signal_delta.nominal_value,
                    signal_delta.std_dev,
                ),
            )
        ),
        energy=energy,
        time_peak1=peaks[0][0],
        signal_peak1=peaks[0][1],
        time_peak2=peaks[1][0],
        signal_peak2=peaks[1][1],
        time_delta=time_delta,
        signal_delta=signal_delta,
        sonic_energy=sonic_energy,
        signal_smooth=signal_smooth,
        description=description,
        wavelength=wavelength,
        exc_wavelength=exc_wavelength,
    )


def compute_alpha(exp: Experiment, force: bool = False) -> pd.DataFrame | None:
    if not exp.done and not force:
        condition = not exp.done and not force
        print(f"{condition=}, {force=}, {exp.done=}")
        return
    elif force:
        # TODO: program an "on_warning" function on the options
        print("Forcing alpha calculation, this might fail.")
    if exp.absorbance is None:
        exp.options["on_error"]("samples absorbance not yet defined")
        print("samples absorbance not yet defined")
        return
    if exp.sam_powerscan is None:
        exp.options["on_error"]("no sample powerscan yet computed")
        print("no sample powerscan yet computed")
        return

    # (m_sam / m_ref) = alpha * (1- 10^-(A_sam)) / (1- 10^-(A_ref))
    abs_sam = exp.absorbance["sam"]
    abs_ref = exp.absorbance["ref"]

    alpha_ref = exp.options["alpha_ref"]

    slope0_sam = exp.sam_powerscan.get_analysis(exp.options)["slope0"]
    slope_sam = exp.sam_powerscan.get_analysis(exp.options)["slope"]

    factor = (1 - 10 ** (-abs_ref)) / (1 - 10 ** (-abs_sam))

    sam_path = ""
    for p, pwsc in exp.powerscans.items():
        if pwsc.get_analysis(exp.options)["sam_ref"] == "sam":
            sam_path = p.name

    alpha_records: list[AlphaRecords] = []

    alphas = []
    alpha0s = []
    for path, powerscan in exp.powerscans.items():
        if powerscan.get_analysis(exp.options)["sam_ref"] == "sam":
            continue
        slope_ref = powerscan.get_analysis(exp.options)["slope"]
        slope0_ref = powerscan.get_analysis(exp.options)["slope0"]
        alpha = alpha_ref * slope_sam / slope_ref * factor
        alpha0 = alpha_ref * slope0_sam / slope0_ref * factor
        alphas.append(alpha)
        alpha0s.append(alpha0)
        alpha_records.append(
            AlphaRecords(
                abs_sam=abs_sam,
                abs_ref=abs_ref,
                ref=path.name,
                sam=sam_path,
                exc_wavelength=powerscan.get_analysis(exp.options)["exc_wavelength"],
                alpha=alpha.nominal_value,
                alpha_unc=alpha.std_dev,
                alpha0=alpha0.nominal_value,
                alpha0_unc=alpha0.std_dev,
            )
        )

    alpha_records.append(
        AlphaRecords(
            abs_sam=abs_sam,
            abs_ref=abs_ref,
            ref="avg",
            sam=sam_path,
            exc_wavelength=powerscan.get_analysis(exp.options)["exc_wavelength"],
            alpha=ufloat_nanmean(*alphas).nominal_value,
            alpha_unc=ufloat_nanmean(*alphas).std_dev,
            alpha0=ufloat_nanmean(*alpha0s).nominal_value,
            alpha0_unc=ufloat_nanmean(*alpha0s).std_dev,
        )
    )
    return pd.DataFrame.from_records(alpha_records)
