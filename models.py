from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from scipy import odr
from uncertainties.core import UFloat, Variable

from constants import OPTIONS, Array

FileDataFrame: TypeAlias = pd.DataFrame
TraceDataFrame: TypeAlias = pd.DataFrame


class TraceAnalysis(TypedDict):
    path: str
    repeat: int | None

    energy: Variable

    time_peak1: Variable
    signal_peak1: Variable

    time_peak2: Variable
    signal_peak2: Variable

    time_delta: Variable
    signal_delta: Variable

    sonic_energy: Variable

    include: bool

    signal_smooth: np.ndarray

    description: str | None
    wavelength: float
    exc_wavelength: float


@dataclass
class TraceMetadata:
    __PA_REPEATS__: int
    __PA_TIME_UNITS__: str
    __PA_SIGNAL_UNITS__: str

    Start: float
    Stop: float
    Step: float
    Wavelength: float
    Bandwidth: float

    Averages: int

    Laser_energy_before: UFloat
    Laser_energy_after: UFloat

    Desc: str
    Comment: str
    XAxis: str
    YAxis: str
    DataChannel: str
    TriggerChannel: str
    Compression: str
    PATH: str
    Type: str

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]):
        attrs2 = {}
        for key in attrs:
            attrs2[key.replace(" ", "_")] = attrs[key]
        return cls(**attrs2)


@dataclass
class Trace:
    time: Array
    signal: Array
    metadata: TraceMetadata

    @classmethod
    def from_trace_dataframe(cls, tracedf: TraceDataFrame):
        attrs: dict[str, Any] = tracedf.attrs
        return cls(
            time=tracedf.time.to_numpy(),
            signal=tracedf.signal.to_numpy(),
            metadata=TraceMetadata.from_attrs(attrs),
        )

    @cached_property
    def analysis(self) -> TraceAnalysis:
        from analysis import analyze_time_trace

        return analyze_time_trace(self.time, self.signal, self.metadata)


@dataclass
class FileMetadata:
    __PA_REPEATS__: tuple[int]
    __PA_TIME_UNITS__: tuple[str]
    __PA_SIGNAL_UNITS__: tuple[str]

    Start: tuple[float]
    Stop: tuple[float]
    Step: tuple[float]
    Wavelength: tuple[float]
    Bandwidth: tuple[float]

    Averages: tuple[int]

    Laser_energy_before: tuple[UFloat]
    Laser_energy_after: tuple[UFloat]

    Desc: tuple[str]
    Comment: tuple[str]
    XAxis: tuple[str]
    YAxis: tuple[str]
    DataChannel: tuple[str]
    TriggerChannel: tuple[str]
    Compression: tuple[str]
    Type: tuple[str]
    PATH: str

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]):
        attrs2 = {}
        for key in attrs:
            attrs2[key.replace(" ", "_")] = attrs[key]
        return cls(**attrs2)


@dataclass
class MeasurementFile:
    traces: list[Trace]
    metadata: FileMetadata

    @classmethod
    def from_path(cls, path: Path):
        from input import yield_individual_repeats, read

        filedf, metadata = read(path)
        traces = []
        for ndx, tracedf in yield_individual_repeats(filedf):
            trace = Trace.from_trace_dataframe(tracedf)
            trace.analysis["repeat"] = ndx
            traces.append(trace)
        return cls(traces, metadata)

    @property
    def energy(self) -> Variable:
        from analysis import ufloat_nanmean

        return ufloat_nanmean(
            *[
                trace.analysis["energy"]
                for trace in self.traces
                if trace.analysis["include"]
            ]
        )

    @property
    def pa_signal(self) -> Variable:
        from analysis import ufloat_nanmean

        return ufloat_nanmean(
            *[
                trace.analysis[OPTIONS["pa_signal"]]
                for trace in self.traces
                if trace.analysis["include"]
            ]
        )

    def to_pandas(self) -> pd.DataFrame:
        records = []
        for trace in self.traces:
            path = Path(trace.metadata.PATH)
            # TODO: maybe define this records dict somewhere else (like typed dict)
            records.append(
                {
                    "path": path.relative_to(path.parent.parent),
                    "repeat": trace.analysis["repeat"],
                    "include": trace.analysis["include"],
                    "energy": trace.analysis["energy"].nominal_value,
                    "time_peak1": trace.analysis["time_peak1"].nominal_value,
                    "signal_peak1": trace.analysis["signal_peak1"].nominal_value,
                    "time_peak2": trace.analysis["time_peak2"].nominal_value,
                    "signal_peak2": trace.analysis["signal_peak2"].nominal_value,
                    "time_delta": trace.analysis["time_delta"].nominal_value,
                    "signal_delta": trace.analysis["signal_delta"].nominal_value,
                }
            )
        return pd.DataFrame.from_records(records)


class PowerscanAnalysis(TypedDict):
    sam_ref: str
    exc_wavelength: float
    description: str
    slope: Variable
    intercept: Variable
    slope0: Variable
    result: odr.Output | None
    result0: odr.Output | None

    energies: list[Variable]
    pa_signals: list[Variable]


@dataclass
class PowerScan:
    measurement_files: dict[Path, MeasurementFile]
    path: Path

    _analysis: PowerscanAnalysis | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_path(cls, path: Path):
        measurement_files = {}
        for filepath in path.glob("*.txt"):
            if filepath.name.startswith("_"):
                print(f"{filepath.name}: skipping, user skip prefix.")
                continue

            measurement_files[filepath] = MeasurementFile.from_path(filepath)
        return cls(measurement_files=measurement_files, path=path)

    @property
    def analysis(self) -> PowerscanAnalysis:
        if self._analysis is None:
            from analysis import analyze_powerscan

            self._analysis = analyze_powerscan(self)
        return self._analysis

    def recompute_analysis(self):
        from analysis import analyze_powerscan

        self._analysis = analyze_powerscan(self)

    def to_pandas(self) -> pd.DataFrame:
        records = []
        for fp, measurement_file in self.measurement_files.items():
            records.append(
                {
                    "path": fp.relative_to(self.path.parent),
                    "description": measurement_file.metadata.Desc[0],
                    "comment": measurement_file.metadata.Comment[0],
                    "wavelength": measurement_file.metadata.Wavelength[0],
                    "bandwidth": measurement_file.metadata.Bandwidth[0],
                    "averages": measurement_file.metadata.Averages[0],
                    "repeats": measurement_file.metadata.__PA_REPEATS__,
                    "energy": measurement_file.energy.nominal_value,
                    "energy_unc": measurement_file.energy.std_dev,
                    "pa_signal": measurement_file.pa_signal.nominal_value,
                    "pa_signal_unc": measurement_file.pa_signal.std_dev,
                }
            )
        return pd.DataFrame.from_records(records)


@dataclass
class Experiment:
    root: Path
    powerscans: dict[Path, PowerScan]
    _absorbance: dict[Literal["sam", "ref"], float] | None = field(
        default=None,
    )
    done: bool = False

    @classmethod
    def from_path(cls, p: Path):
        from input import read_absorbance

        root = p
        powerscans = {}
        for subfolder in root.iterdir():
            if not subfolder.is_dir():
                print(f"{subfolder}: skipping, not a folder.")
                continue
            if subfolder.stem.startswith("_"):
                print(f"{subfolder}: skipping, user skip prefix.")
                continue
            if not any(subfolder.glob("*.txt")):
                print(f"{subfolder}: skipping, folder without text files")
                continue

            powerscans[subfolder] = PowerScan.from_path(subfolder)

        _absorbance = read_absorbance(root / "abs.txt")
        done = (root / "done.txt").exists()

        return cls(root, powerscans, _absorbance, done)

    @property
    def absorbance(self) -> dict[Literal["sam", "ref"], float] | None:
        from input import read_absorbance

        if self._absorbance is None:
            self._absorbance = read_absorbance(self.root / "abs.txt")
        return self._absorbance

    def set_absorbance(self, p: Path):
        from input import read_absorbance

        self._absorbance = read_absorbance(p)

    @cached_property
    def sam_powerscan(self) -> PowerScan | None:
        for pwsc in self.powerscans.values():
            if pwsc.analysis["sam_ref"] == "sam":
                return pwsc

    def __str__(self):
        return f"""
root: {self.root}
absorbance: {self.absorbance}
done: {self.done}
folders:\t
""" + "\n".join(
            [
                f"\t{path.relative_to(self.root)}:\n\t\t"
                + "\n\t\t".join(
                    f"{filepath.name}" for filepath in powerscan.measurement_files
                )
                for path, powerscan in self.powerscans.items()
            ]
        )
