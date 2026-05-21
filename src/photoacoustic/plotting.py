import datetime
from pathlib import Path
import pickle
from typing import Iterable

from matplotlib import colors, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.typing import ColorType
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from uncertainties.core import Variable

from photoacoustic.models import Experiment, PowerScan, Trace
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from photoacoustic.constants import OPTIONS, Array, _footnote_timestamp, __version__


def get_powerscan_overview_name(foldername: str) -> str:
    sample, wl, *metadata = foldername.split("_")
    metadata_str = "-".join(metadata)
    return f"__powerscan-overview__{sample}__{wl}__{metadata_str}"


def get_time_trace_name(filepath: Path, repeat: int) -> str:
    sample = filepath.parent.name.split("_")[0]
    _, wl, *metadata = str(filepath.stem).split("_")
    metadata_str = "-".join(metadata)
    return f"__time-trace__{sample}__{wl}__{str(repeat).zfill(3)}__{metadata_str}"


def footnote(fig: Figure, *, left_footer: str = "", right_footer: str = ""):
    """Add footnote to page."""

    if left_footer:
        fig.text(0.02, 0.01, left_footer, ha="left", fontsize=6, wrap=True)  # type: ignore
    if right_footer:
        fig.text(0.98, 0.01, right_footer, ha="right", fontsize=6, wrap=True)  # type: ignore


def default_footnote(fig: Figure | None):
    """Add default footnote to page, which includes the analysis
    datetime and the script version.

    To initialize the analysis datetime to current time,
    call this function with None value.
    """
    global _footnote_timestamp
    if fig is None:
        _footnote_timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    else:
        footnote(
            fig,
            left_footer=f"Analysis datetime: {_footnote_timestamp}",
            right_footer=f"Photoacoustic analysis version: {__version__}",
        )


def build_powerscan_overview_figure(
    signals: list[tuple[Array, Array]], energy: Array
) -> Figure:
    fig, ax = plt.subplots(1, 1)

    fig.set_figwidth(297 / 40)
    fig.set_figheight(210 / 40)

    ax.set_xlabel(r"$\Delta$time / $\mu s$")
    ax.set_ylabel("signal / V")

    try:
        norm = colors.Normalize(vmin=energy.min(), vmax=energy.max())
    except Exception:
        norm = colors.Normalize(vmin=0, vmax=1)

    for power, (time, signal) in zip(energy, signals):
        ax.plot(time, signal, c=plt.cm.jet(norm(power)))

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet),
        orientation="vertical",
        ax=ax,
        label=r"Laser power / $\mu J$",
    )
    default_footnote(fig)
    plt.tight_layout()

    return fig


def save_fig_to_pickle(fig: Figure, path: Path):
    with open(path, "wb") as f:
        pickle.dump(fig, f)


def save_all_figures(
    experiment: Experiment,
    overwrite: bool = False,
):

    new_trace_figure = False
    for folderpath, powerscan in experiment.powerscans.items():
        # TODO: program a way of getting a list of times and signasl from powerscan
        signals: list[tuple[Array, Array]] = []
        energies = []
        for (
            measurement_filepath,
            measurement_file,
        ) in powerscan.measurement_files.items():
            for repeat, trace in enumerate(measurement_file.traces):
                filename = get_time_trace_name(measurement_filepath, repeat)

                figure_filepath = (
                    experiment.root
                    / OPTIONS["figures_save_path"]
                    / f"{filename}.pickle"
                )
                if not figure_filepath.parent.exists():
                    figure_filepath.parent.mkdir()
                if not figure_filepath.exists() or overwrite:
                    OPTIONS["on_progress"](
                        f"building time trace figure {filename} repeat {repeat}"
                    )
                    fig = build_time_trace_figure(trace)
                    save_fig_to_pickle(
                        fig,
                        figure_filepath,
                    )
                    new_trace_figure = True
                signals.append((trace.time, trace.signal))
                energies.append(trace.analysis["energy"].nominal_value)

        if new_trace_figure:
            fig = build_powerscan_overview_figure(
                signals=signals, energy=np.asarray(energies)
            )
            figname = get_powerscan_overview_name(folderpath.name)
            save_fig_to_pickle(
                fig,
                experiment.root / OPTIONS["figures_save_path"] / f"{figname}.pickle",
            )
    if new_trace_figure:
        fig = build_linear_fit_figure(list(experiment.powerscans.values()))
        save_fig_to_pickle(
            fig, experiment.root / OPTIONS["figures_save_path"] / "__linear_fit.pickle"
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

    ax.set_title(
        "/".join(trace.metadata.PATH.split("/")[-2:])
        + f"\nrep ({trace.analysis['repeat']}/{trace.metadata.__PA_REPEATS__})"
    )

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


def sample_repeat_metadata_from_name(fname: str) -> tuple[str, int, str]:
    samp, wl, repeat, metadata = fname.split("__")[2:]
    repeat = int(repeat)
    return samp, repeat, metadata


def order_images(root: Path):
    figs_path = root / OPTIONS["figures_save_path"]
    time_trace_paths: dict[str, dict[str, dict[int, Path]]] = {}
    powerscan_paths: dict[str, Path] = {}
    for fp in figs_path.glob("*.pickle"):
        if fp.stem.startswith("__time-trace"):
            sample, repeat, metadata = sample_repeat_metadata_from_name(fp.stem)
            if sample not in time_trace_paths.keys():
                time_trace_paths[sample] = {}
            if metadata not in time_trace_paths[sample].keys():
                time_trace_paths[sample][metadata] = {}
            time_trace_paths[sample][metadata][repeat] = fp
        elif fp.stem.startswith("__powerscan-overview"):
            sample = fp.stem.split("__")[2]
            powerscan_paths[sample] = fp

    assert sorted(list(time_trace_paths.keys())) == sorted(list(powerscan_paths.keys()))

    ordered_paths = []
    for sample, metadata_dicts in time_trace_paths.items():
        for metadata, repeats_dict in metadata_dicts.items():
            for i in range(len(repeats_dict.keys())):
                ordered_paths.append(repeats_dict[i])
        ordered_paths.append(powerscan_paths[sample])
    return [*ordered_paths, figs_path / "__linear_fit.pickle"]


def build_pdf(root: Path):
    with PdfPages(root / "summary.pdf") as pdf:
        for image_path in order_images(root):
            with open(image_path, "rb") as f:
                fig = pickle.load(f)
                pdf.savefig(fig, dpi=200)
                plt.close(fig)


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


def get_line_colors() -> dict:
    return {
        "sam": ["C0", "C4", "C6", "C9"],
        "ref0": ["C1", "C3", "C5", "C7"],
        "ref1": ["C2", "C8", "khaki", "olivedrab"],
    }


def split_unc_tuple(
    *variables: Variable, container: type = tuple
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Unzip iterable of uncertainties into a tuple of nominal value and a tuple of std dev"""
    return container(v.nominal_value for v in variables), container(
        v.std_dev for v in variables
    )


def plot_linear_with_intercept(
    powerscan: PowerScan, ax_plot: Axes, color: ColorType | None
):
    x, x_unc = split_unc_tuple(*powerscan.analysis["energies"])
    x_fit = np.linspace(0, np.max(x) * 1.1, 10)
    y_fit = (
        powerscan.analysis["slope"].nominal_value * x_fit
        + powerscan.analysis["intercept"].nominal_value
    )

    if color is not None:
        (line,) = ax_plot.plot(x_fit, y_fit, color=color)
    else:
        (line,) = ax_plot.plot(x_fit, y_fit)

    color = line.get_color()

    if OPTIONS["plot_uncertainty_slope"]:
        xa = np.linspace(0, np.max(x) * 1.1, 100)

        # Slope
        ya_var = (
            xa**2 * powerscan.analysis["result"].cov_beta[0, 0]
            + powerscan.analysis["result"].cov_beta[1, 1]
            + 2 * xa * powerscan.analysis["result"].cov_beta[0, 1]
        )
        ya_unc = np.sqrt(ya_var)
        ya = (
            powerscan.analysis["slope"].nominal_value * xa
            + powerscan.analysis["intercept"].nominal_value
        )

        ax_plot.fill_between(
            xa,
            y1=ya - ya_unc,
            y2=ya + ya_unc,
            color=line.get_color(),
            alpha=0.2,
        )


def plot_linear(powerscan: PowerScan, ax_plot: Axes, color: ColorType | None):
    x, x_unc = split_unc_tuple(*powerscan.analysis["energies"])
    y, y_unc = split_unc_tuple(*powerscan.analysis["pa_signals"])
    x_fit = np.linspace(0, np.max(x) * 1.1, 10)
    y_fit = powerscan.analysis["slope0"].nominal_value * x_fit
    (line,) = ax_plot.plot(x_fit, y_fit, color=color, ls=":")

    ax_plot.errorbar(
        x,
        y,
        xerr=x_unc,
        yerr=y_unc,
        linestyle="None",
        marker=".",
        color=line.get_color(),
    )
    if OPTIONS["plot_uncertainty_slope0"]:
        try:
            xa = np.linspace(0, np.max(x) * 1.1, 100)

            # Slope0
            ya_unc = np.sqrt(xa**2 * powerscan.analysis["result0"].cov_beta[0, 0])
            ya = powerscan.analysis["slope0"].nominal_value * xa

            ax_plot.fill_between(
                xa,
                y1=ya - ya_unc,
                y2=ya + ya_unc,
                color=line.get_color(),
                alpha=0.2,
            )
        except Exception as e:
            OPTIONS["on_error"](f"Couldn't plot uncertainty of fit with origin 0: {e}")


def build_linear_fit_figure(powerscans: Iterable[PowerScan]) -> Figure:
    fig, (ax_plot, ax_meta) = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=(0.7, 0.3))
    )
    fig.set_figwidth(297 / 40)
    fig.set_figheight(210 / 40)

    ax_plot.set_xlabel(r"Laser power / $\mu J$")
    ax_plot.set_ylabel("PAS / V")
    ax_plot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    cellText = []
    rowLabels = []
    rowColours = []

    for powerscan in powerscans:
        powerscan.recompute_analysis()

        label = powerscan.path.name
        sample_name = label.split("_")[0]

        slope = powerscan.analysis["slope"]
        slope0 = powerscan.analysis["slope0"]
        intercept = powerscan.analysis["intercept"]

        color = None
        if sample_name in ("ref0", "ref1", "sam"):
            try:
                color = get_line_colors()[sample_name][0]
            except IndexError:
                OPTIONS["on_error"]("Ran out of line colors, changing to default")
            except Exception as ex:
                OPTIONS["on_error"](
                    f"An exception ocurred while trying to set line colors: {ex}"
                )

        if OPTIONS["plot_with_intercept"]:
            plot_linear_with_intercept(powerscan, ax_plot, color)

        plot_linear(powerscan, ax_plot, color)

        cellText.append(
            (label, f"${slope:.2uL}$", f"${intercept:.2uL}$", f"${slope0:.2uL}$"),
        )

        rowColours.append(color)
        rowLabels.append("   ")

    table = ax_meta.table(
        cellText=cellText,
        colLabels=(
            "Subfolder",
            r"Slope / $\left(V / \mu J \right)$",
            "Intercept / $V$",
            r"Slope0 / $\left(V / \mu J \right)$",
        ),
        loc="center",
        rowColours=rowColours,
        rowLabels=rowLabels,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(5)

    ax_meta.axis(False)

    default_footnote(fig)
    fig.tight_layout()

    return fig
