import argparse
import sys
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from spectracoustic_analysis.constants import OPTIONS, Options, default_options
from spectracoustic_analysis.models import Experiment, PowerscanAnalysis
from spectracoustic_analysis.plotting import (
    build_linear_fit_figure,
    build_powerscan_overview_figure,
)

# inputs:
# absorbance path for each sample
# alpha sepectrum path


def read_absorbance(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=None, skiprows=1)
        df.columns = ["wavelength", "absorbance"]
        return df
    except Exception as e:
        raise IOError(
            f"Couldn't read absorbance file with path {path} because of exception {e}"
        )


def read_experiments(root: Path, options: Options) -> dict[int, Experiment]:
    experiments: dict[int, Experiment] = {}
    for wldir in sorted(root.iterdir()):
        if not wldir.is_dir():
            print(f"Skipping {wldir.relative_to(root)}: not a directory")
            continue
        try:
            wl = int(wldir.name)
        except Exception:
            raise IOError(
                f"Wavelength powerscan directory name should be convertible to wavelength value (int), not {wldir.relative_to(root)}"
            )

        experiments[wl] = Experiment.from_path(wldir, options=default_options())
    return experiments


def sample_name_from_powerscan_path(path: Path) -> str:
    return path.name.split("_")[0]


def experiments_to_analysis_df(
    experiments: dict[int, Experiment], options: Options
) -> pd.DataFrame:
    """'
    return dataframe has the same columns as PowerscanAnalysis has fields, but with the prefix "ref0", "ref1", "sam",
    or any corresponding sample name.
    """
    sample_name_powerscans: dict[str, list[PowerscanAnalysis]] = {}
    for wl, experiment in experiments.items():
        for path, powerscan in experiment.powerscans.items():
            sample_name = sample_name_from_powerscan_path(path)
            if sample_name not in sample_name_powerscans.keys():
                sample_name_powerscans[sample_name] = []

            analysis = powerscan.get_analysis(options)
            sample_name_powerscans[sample_name].append(analysis)

    df_analysis = pd.concat(
        [
            pd.DataFrame(powerscans)
            .add_prefix(f"{sample_name}_")
            .drop(f"{sample_name}_pa_signals", axis=1)
            .drop(f"{sample_name}_energies", axis=1)
            .drop(f"{sample_name}_result", axis=1)
            .drop(f"{sample_name}_result0", axis=1)
            for sample_name, powerscans in sample_name_powerscans.items()
        ],
        axis=1,
    )

    sample_names = list(sample_name_powerscans.keys())
    amount_samples = len(sample_names)
    for i in range(amount_samples - 1):
        for j in range(i + 1, amount_samples):
            assert np.allclose(
                df_analysis[f"{sample_names[i]}_exc_wavelength"].to_numpy(),
                df_analysis[f"{sample_names[j]}_exc_wavelength"].to_numpy(),
            ), (
                f"Wavelength vector for {sample_names[i]} doesn't match wavelength vector for {sample_names[j]}"
            )
    #    slope: Variable
    #    intercept: Variable
    #    slope0: Variable

    variable_columns = []
    [
        variable_columns.extend(
            [f"{sample}_slope", f"{sample}_slope0", f"{sample}_intercept"]
        )
        for sample in sample_names
    ]
    df_analysis = unzip_unc_column(df_analysis, *variable_columns)
    df_analysis.attrs["sample_names"] = sample_names
    return df_analysis


def unzip_unc_column(
    df: pd.DataFrame, *column_names: str, drop_unc: bool = False
) -> pd.DataFrame:
    """Unzip uncertainty column (`name`) into nominal_value (`name`) and std_dev ()`name_unc`).

    If `drop_unc` is True, only the nominal value will be extracted.
    """

    original_columns = df.columns
    for column_name in column_names:
        if drop_unc:
            df[OdrResult[column_name,]] = (
                df[column_name].apply(lambda x: (x.nominal_value,)).to_list()
            )
        else:
            df[[column_name, column_name + "_unc"]] = (
                df[column_name].apply(lambda x: (x.nominal_value, x.std_dev)).to_list()
            )

    if drop_unc:
        return df

    new_columns = []
    for column_name in original_columns:
        new_columns.append(column_name)
        if column_name in column_names:
            new_columns.append(column_name + "_unc")

    return df.reindex(columns=new_columns)


def interpol_spectra(
    df_ref: pd.DataFrame, df_sam: pd.DataFrame, df_analysis: pd.DataFrame
) -> pd.DataFrame:
    """
    this interpolates assuming absorbance dfs are longer in wavelength.
    It also assumes ref0 wls == sam wls == any other
    """
    interpolable_columns = ["slope", "slope0", "intercept"]
    min_wl = df_analysis.ref0_exc_wavelength.min()
    max_wl = df_analysis.ref0_exc_wavelength.max()

    all_wls = np.arange(min_wl, max_wl + 1, 1)
    records = {"wavelength": all_wls}
    for sample_name in df_analysis.attrs["sample_names"]:
        for col in interpolable_columns:
            records[f"{sample_name}_{col}"] = np.interp(
                all_wls,
                df_analysis[f"{sample_name}_exc_wavelength"],
                df_analysis[f"{sample_name}_{col}"],
            )
    records["ref_abs"] = df_ref[
        np.logical_and(df_ref.wavelength <= max_wl, df_ref.wavelength >= min_wl)
    ].absorbance.to_numpy()
    records["sam_abs"] = df_sam[
        np.logical_and(df_sam.wavelength <= max_wl, df_sam.wavelength >= min_wl)
    ].absorbance.to_numpy()
    return pd.DataFrame.from_records(records)


def generate_report(root: Path, ref: Path, sam: Path, plot_linears: bool):
    df_ref = read_absorbance(ref)
    df_sam = read_absorbance(sam)

    options = default_options()

    experiments = read_experiments(root, options)
    photoacoustic_wls = np.array(list(experiments.keys()))

    df_analysis = experiments_to_analysis_df(experiments, options)
    df_spectra = interpol_spectra(df_ref, df_sam, df_analysis)

    plt.plot(
        df_spectra.wavelength,
        df_spectra.ref0_slope0 / df_spectra.ref0_slope0.max(),
        label="ref0 slope",
        color="C0",
    )
    plt.plot(
        df_spectra.wavelength,
        df_spectra.sam_slope0 / df_spectra.sam_slope0.max(),
        label="sam slope",
        color="C1",
    )
    plt.plot(
        df_spectra.wavelength,
        df_spectra.ref_abs / df_spectra.ref_abs.max(),
        label="ref abs",
        color="C0",
        ls="dashed",
    )
    plt.plot(
        df_spectra.wavelength,
        df_spectra.sam_abs / df_spectra.sam_abs.max(),
        label="sam abs",
        color="C1",
        ls="dashed",
    )

    df_spectra["alpha"] = (
        df_spectra["sam_slope0"]
        * (1 - 10 ** (-df_spectra["ref_abs"]))
        / (df_spectra["ref0_slope0"] * (1 - 10 ** (-df_spectra["sam_abs"])))
    )

    plt.plot(
        df_spectra.wavelength,
        df_spectra["alpha"] / df_spectra["alpha"].max(),
        label="alpha",
        color="black",
    )
    plt.legend(fontsize=15)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("normalized spectra")
    plt.show()


def get_absorbance_path(
    sam_ref: Literal["sam", "ref"], argv: argparse.Namespace
) -> Path:
    match sam_ref:
        case "sam":
            if argv.sam is None:
                return argv.root.resolve() / "sam_abs.txt"
            return argv.ref.resolve()
        case "ref":
            if argv.ref is None:
                return argv.root.resolve() / "sam_abs.txt"
            return argv.ref.resolve()
        case _:
            raise ValueError(f"sam_ref should be sam or ref, not {sam_ref}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spectral-analysis",
        description="Analyze spectral powerscan data",
    )
    parser.add_argument(
        "-e",
        "--experiment-root",
        type=Path,
        help="Path to the root of the experiment folder",
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=Path,
        help="Path to the reference absorbance.txt file",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-s",
        "--sam",
        type=Path,
        help="Path to the sample absorbance.txt file",
        default=None,
        required=False,
    )
    parser.add_argument(
        "-l",
        "--plot-linears",
        type=bool,
        help="Option to plot linear functions on the report",
        default=True,
        required=False,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parse_args(sys.argv[1:])

    root = args.experiment_root.resolve()
    sam_abs = get_absorbance_path("sam", args)
    ref_abs = get_absorbance_path("sam", args)
    plot_linears = args.plot_linears

    generate_report(
        root,
        sam_abs,
        ref_abs,
        plot_linears,
    )
