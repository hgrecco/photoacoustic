from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from scipy import odr
from uncertainties.core import UFloat, Variable

from constants import Array

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
            traces.append(trace)
        return cls(traces, metadata)


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

    @classmethod
    def from_path(cls, path: Path):
        measurement_files = {}
        for filepath in path.glob("*.txt"):
            if filepath.name.startswith("_"):
                print(f"{filepath.name}: skipping, user skip prefix.")
                continue

            measurement_files[filepath] = MeasurementFile.from_path(filepath)
        return cls(measurement_files=measurement_files, path=path)

    @cached_property
    def analysis(self) -> PowerscanAnalysis:
        from analysis import analyze_powerscan

        return analyze_powerscan(self)


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

            powerscans[subfolder] = PowerScan.from_path(subfolder)

        _absorbance = read_absorbance(root / "abs.txt")
        done = (root / "done.txt").exists()

        return cls(root, powerscans, _absorbance, done)

    def update(self, path: Path) -> None:
        from input import read_absorbance

        if not path.is_relative_to(self.root):
            print(f"{path} not within {self.root}")
            return

        relative_path = path.relative_to(self.root)
        depth = len(relative_path.parts)

        if depth == 1:
            if path.is_dir() and not path.name.startswith("_"):
                print("New directory created")
                self.powerscans[path] = PowerScan.from_path(path)
            elif not path.is_dir() and path.name not in (
                "abs.txt",
                "done.txt",
            ):
                print(f"Ignoring file {path}")
            elif path.name == "abs.txt":
                print("Updating abs.txt path")
                self._absorbance = read_absorbance(path)
            elif path.name == "done.txt":
                print("All experiment files are in the directory")
                self.done = True
            else:
                raise RuntimeError("Unexpected error updating the directory structure")
        elif depth == 2:
            if path.is_dir():
                print("Ignoring directory too deep in the experiment file structure")
            elif path.name.startswith("_"):
                print("skipping for user prefix")
            elif path.name.endswith(".txt"):
                print(f"Adding file {relative_path.name} to {path.parent}")
                self.powerscans[path.parent].measurement_files[path] = (
                    MeasurementFile.from_path(path)
                )
            else:
                print(f"ignoring file {path.name} for not being txt")
        else:
            print(
                f"Ignoring file or directory with depth {depth} in the experiment file structure"
            )

    @property
    def absorbance(self) -> dict[Literal["sam", "ref"], float] | None:
        from input import read_absorbance

        if self._absorbance is None:
            self._absorbance = read_absorbance(self.root / "abs.txt")
        return self._absorbance

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
