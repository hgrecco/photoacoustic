from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Literal, Protocol

import numpy as np
import pandas as pd
import photoacoustic.constants as constants
from photoacoustic.constants import Array, OnErrorFunc

# from photoacoustic.constants import OPTIONS
from photoacoustic.models import FileDataFrame, FileMetadata, TraceDataFrame
from uncertainties.core import ufloat, ufloat_fromstr

# Repeats is not included as metadata in the file.
ATTR_REPEATS = "__PA_REPEATS__"

ATTR_TIME_UNITS = "__PA_TIME_UNITS__"
ATTR_SIGNAL_UNITS = "__PA_SIGNAL_UNITS__"

# Metadata float attributes
ATTRS_FLOAT = ("Start", "Stop", "Step", "Wavelength", "Bandwidth")
# Metadata integer attributes
ATTRS_INT = ("Averages",)
# Metadata uncertainty attributes
ATTRS_UNC = ("Laser energy before", "Laser energy after")

UFLOAT0 = ufloat(0, 0)
UFLOAT_NAN = ufloat(np.nan, np.nan)


def parse_metadata_content(
    metadata: dict[str, str], repeats: int | None = None
) -> Generator[tuple[Any, Any], None, None]:
    """Parse metadata content."""

    if repeats is None:
        fun = lambda conv, x: conv(x)  # noqa
    else:
        fun = lambda conv, x: tuple(conv(el) for el in x.split(","))  # noqa

    for k, v in metadata.items():
        k.replace(" ", "_")
        if k in ATTRS_FLOAT:
            yield k, fun(float, v)
        elif k in ATTRS_INT:
            yield k, fun(int, v)
        elif k in ATTRS_UNC:
            yield k, fun(ufloat_fromstr, v)
        else:
            yield k, fun(str, v)


class ReadLiner(Protocol):
    def readline(self) -> str:
        ...


def read_metadata(fi: ReadLiner) -> dict[str, str]:
    """Consume metadata from an open file."""
    metadata: dict[str, str] = {}
    cnt = 0
    while True:
        line = str.strip(fi.readline())
        if not line:
            cnt += 1
            if cnt == 2:
                # Two empty lines
                break
            continue
        key, value = line.split(",", 1)
        metadata[key] = value

    return metadata


def read_without_repeats(p: Path | str) -> tuple[FileDataFrame, FileMetadata]:
    """Read Edinburgh Instruments ascii file (with no repeats)."""
    if isinstance(p, str):
        p = Path(p)

    with p.open("r", encoding="ascii") as fi:
        metadata = read_metadata(fi)

        # TODO: is the time always in ms? is the signal always in Volts?
        df = pd.read_csv(  # type: ignore
            fi, sep=",", header=0, names=("time", "signal")
        )
        df["time"] = df["time"] / 1_000

        for k, v in parse_metadata_content(metadata, None):
            df.attrs[k] = v

    df.attrs[ATTR_TIME_UNITS] = "microseconds"
    df.attrs[ATTR_SIGNAL_UNITS] = "volts"
    df.attrs[ATTR_REPEATS] = None
    metadata = FileMetadata.from_attrs(df.attrs)
    return df, metadata


def read_with_repeats(p: Path | str) -> tuple[FileDataFrame, FileMetadata]:
    """Read Edinburgh Instruments ascii file (with repeats)."""

    if isinstance(p, str):
        p = Path(p)

    with p.open("r", encoding="ascii") as fi:
        fi.readline()  # In files with repeats, the first two lines are like a header
        fi.readline()  # In files with repeats, the first two lines are like a header

        metadata = read_metadata(fi)

        # As the number of repeats is not stored in the metadata,
        # we infer this value by counting the number of values in
        # one the metadata keys.

        estimated_length = len(metadata["Averages"].split(","))

        # TODO: is the time always in milliseconds? is the signal always in Volts?
        df = pd.read_csv(  # type: ignore
            fi,
            sep=",",
            header=0,
            names=("time",) + tuple("signal%d" % n for n in range(estimated_length)),
        )
        df["time"] = df["time"] / 1_000

        for k, v in parse_metadata_content(metadata, estimated_length):
            assert len(v) == estimated_length
            df.attrs[k] = v

    df.attrs["PATH"] = str(p)
    df.attrs[ATTR_TIME_UNITS] = "microseconds"
    df.attrs[ATTR_SIGNAL_UNITS] = "volts"
    df.attrs[ATTR_REPEATS] = estimated_length
    metadata = FileMetadata.from_attrs(df.attrs)
    return df, metadata


def read(p: Path | str) -> tuple[FileDataFrame, FileMetadata]:
    """Read Edinburgh Instruments ascii file (with or without repeats).

    returns a DataFrame with attrs referring to metadata and traces.
    """

    if isinstance(p, str):
        p = Path(p)

    # is there a better way?
    with p.open("r", encoding="ascii") as fi:
        # In a file with repeats, the first line contains the filename.
        # In a file without repeats, the file line contains the
        # first metadata key value pair, comma separated.
        # This will fail is the filename contains a comma (which is rare)

        if "," in fi.readline():
            return read_without_repeats(p)
        else:
            return read_with_repeats(p)


def read_absorbance(
    path: Path, on_error: OnErrorFunc = constants.on_error_default
) -> dict[Literal["sam", "ref"], float]:
    absorbances = {}
    try:
        for line in path.read_text().splitlines():
            k, v = line.split("=")
            k = k.strip().lower()
            if k in ("sam", "ref"):
                absorbances[k] = float(v.strip())
            else:
                on_error(f"absorbance name should be sam or ref, not {k}")
        if sorted(list(absorbances.keys())) != ["ref", "sam"]:
            on_error("absorption abs.txt file incomplete")
    except Exception as ex:
        on_error(f"Absorption values could not be loaded: {str(ex)}")
    return absorbances


def yield_individual_repeats(
    raw_df: FileDataFrame,
) -> Generator[tuple[int | None, TraceDataFrame], None, None]:
    """Yield number of repetition and dataframe."""

    time: Array = raw_df["time"].to_numpy()

    if ATTR_REPEATS in raw_df.attrs and raw_df.attrs[ATTR_REPEATS] is not None:
        for ndx in range(raw_df.attrs[ATTR_REPEATS]):
            signal: Array = raw_df["signal%d" % ndx].to_numpy()
            tmpdf = pd.DataFrame(dict(time=time, signal=signal))
            for k, v in raw_df.attrs.items():
                if isinstance(v, (tuple, np.ndarray)):
                    tmpdf.attrs[k] = v[ndx]
                else:
                    tmpdf.attrs[k] = v
            yield ndx, tmpdf
    else:
        yield None, raw_df
