import datetime

from matplotlib import colors, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from models import Experiment, Trace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from constants import OPTIONS, _footnote_timestamp, __version__


def get_time_trace_name(filename: str, repeat: int) -> str:
    sample, wl, *metadata = filename.split("_")
    metadata_str = "-".join(metadata)
    return f"_time-trace_{sample}_{wl}_{str(repeat).zfill(3)}_{metadata_str}.png"


def save_all_figures(
    experiment: Experiment,
    overwrite: bool = False,
):

    for folderpath, powerscan in experiment.powerscans.items():
        for (
            measurement_filepath,
            measurement_file,
        ) in powerscan.measurement_files.items():
            for repeat, trace in enumerate(measurement_file.traces):
                filename = get_time_trace_name(str(measurement_filepath.name), repeat)

                OPTIONS["on_progress"](
                    f"building time trace figure {filename} repeat {repeat}"
                )

                figure_filepath = (
                    experiment.root / OPTIONS["figures_save_path"] / filename
                )
                if not figure_filepath.parent.exists():
                    figure_filepath.parent.mkdir()
                if not figure_filepath.exists() or overwrite:
                    fig = build_time_trace_figure(trace)
                    fig.savefig(
                        experiment.root / OPTIONS["figures_save_path"] / filename,
                        dpi=200,
                    )


def plot_signal_and_peaks(ax: Axes, trace: Trace):
    """Plot the signal and peaks (if given).

    Parameters
    ----------
    ax
        Matplotlib axes to draw.
    signal_df
        Dataframe with the signal.
    peak_df
        Dataframe with found peaks.
    """

    ax.plot(trace.time, trace.signal, c="tab:gray")

    signal_smooth = trace.analysis["signal_smooth"]
    if signal_smooth is not None:
        ax.plot(trace.time, signal_smooth, c="tab:blue")

    mx = np.max(np.abs((ax.get_ylim())))
    ax.set_ylim(-mx, mx)

    bg = trace.signal[:500].mean()
    std = trace.signal[:500].std()

    ax.axhline(y=bg, ls=":", c="black")
    ax.axhline(y=bg - std, ls=":", c="black")
    ax.axhline(y=bg + std, ls=":", c="black")

    ax.set_xlabel(r"time / $\mu s$")
    ax.set_ylabel("signal / V")

    for n in (1, 2):
        x = (
            trace.analysis["time_peak1"].nominal_value
            if n == 1
            else trace.analysis["time_peak2"].nominal_value
        )
        y = (
            trace.analysis["signal_peak1"].nominal_value
            if n == 1
            else trace.analysis["signal_peak2"].nominal_value
        )
        if np.isnan(x) or np.isnan(y):
            continue
        ax.axvline(x=x, ls="--", c="tab:green")
        ax.axhline(y=y, ls="--", c="tab:green")
        ax.plot([x], [y], "x", c="tab:red")


def build_time_trace_figure(trace: Trace) -> Figure:
    """Plot a figure

    Parameters
    ----------
    sample
        name of the sample.
    signal_df
        Dataframe with the signal.
    peak_df
        Dataframe with the found peaks.
    title, optional

    """

    fig = plt.figure()
    fig.set_figwidth(297 / 40)
    fig.set_figheight(210 / 40)

    gs = GridSpec(3, 3, figure=fig)  # , bottom=.05)
    ax_plot = fig.add_subplot(gs[:2, :])
    ax_meta = fig.add_subplot(gs[2, :2])
    ax_peak = fig.add_subplot(gs[2, -1])

    ax_plot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax_inset: Axes = inset_axes(ax_plot, width="30%", height="20%")
    ax_inset.plot(trace.time, trace.signal)
    ax_inset.get_xaxis().set_ticks([])
    ax_inset.get_yaxis().set_ticks([])

    plot_signal_and_peaks(ax_plot, trace)
    ax_meta.axis(False)
    ax_peak.axis(False)

    cellText = [
        ("Description", trace.analysis["description"]),
        ("Wavelength", f"{trace.analysis['wavelength']} nm"),
        (
            "Laser energy",
            r"$({:.2uL})~\mu J$".format(trace.analysis["energy"]),
        ),
        ("Exc. Wavelength", f"{trace.analysis.get('exc_wavelength', 'N/A')} nm"),
    ]

    table = ax_meta.table(
        cellText=cellText, colLabels=("Parameter", "Value"), loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5)

    ax_peak.set_title(f"Peaks (include={trace.analysis['include']})")
    table = ax_peak.table(
        cellText=[
            (
                f"{trace.analysis['time_peak1'].nominal_value:.3f}",
                f"{trace.analysis['signal_peak1'].nominal_value:.3f}",
            ),
            (
                f"{trace.analysis['time_peak2'].nominal_value:.3f}",
                f"{trace.analysis['signal_peak2'].nominal_value:.3f}",
            ),
            (
                f"{trace.analysis['time_delta'].nominal_value:.3f}",
                f"{trace.analysis['signal_delta'].nominal_value:.3f}",
            ),
        ],
        colLabels=(
            # "Peak #",
            r"Time / $\mu s$",
            "Signal / V",
        ),
        rowLabels=(" #1 ", " #2 ", r" $\Delta$ "),
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    # table.scale(1, 4)

    t0 = trace.analysis["time_peak1"]
    t1 = trace.analysis["time_peak2"]
    if np.isfinite(t0.nominal_value) and np.isfinite(t1.nominal_value):
        lb = t0.nominal_value - 3 * (t1.nominal_value - t0.nominal_value)
        ub = t0.nominal_value + 6 * (t1.nominal_value - t0.nominal_value)
        ax_plot.set_xlim(lb, ub)
        ax_inset.axvline(x=lb, ls="-", c="black")
        ax_inset.axvline(x=ub, ls="-", c="black")

    fig.tight_layout()

    return fig
