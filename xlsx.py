from pathlib import Path

from models import Experiment, MeasurementFile, PowerScan
import pandas as pd


def write_excel(root: Path, exp: Experiment):
    with pd.ExcelWriter(root / "summary.xlsx") as xlsx:
        for _, powerscan in exp.powerscans.items():
            for _, measurement_file in powerscan.measurement_files.items():
                write_file_sheet(xlsx, measurement_file)
            write_powerscan_sheet(xlsx, powerscan)


def write_file_sheet(xlsx: pd.ExcelWriter, measurement_file: MeasurementFile) -> None:
    path = Path(measurement_file.metadata.PATH)
    try:
        prefix = "(%s)" % path.parent.stem.split("_")[0]
    except Exception:
        prefix = "(?)"
    measurement_file.to_pandas().to_excel(
        xlsx, sheet_name=prefix + " " + path.stem, startrow=0, index=False, header=True
    )


def write_powerscan_sheet(xlsx: pd.ExcelWriter, powerscan: PowerScan) -> None:
    powerscan.to_pandas().to_excel(xlsx, sheet_name=powerscan.path.stem, index=False)


# folder	sam_ref	exc_wavelength	description	slope	slope_unc	intercept	intercept_unc	slope0	slope0_unc	result	result0	p-value ref0 slope	p-value ref0 slope0
