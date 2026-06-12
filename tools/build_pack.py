from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WHEELHOUSE = DIST / "pack-wheelhouse"
TARGET_PLATFORM = "win-64"


@dataclass(frozen=True)
class Project:
    name: str
    path: Path


PROJECTS = (Project("analysis-rt", ROOT),)


def load_project_metadata(project: Project) -> tuple[str, str]:
    pyproject = project.path / "pyproject.toml"
    if not pyproject.is_file():
        raise SystemExit(f"Missing {pyproject} for {project.name}")

    with pyproject.open("rb") as file:
        data = tomllib.load(file)

    metadata = data.get("project", {})
    name = metadata.get("name")
    version = metadata.get("version")
    if name != project.name:
        raise SystemExit(
            f"Expected project {project.name!r} at {project.path}, found {name!r}"
        )
    if not version:
        raise SystemExit(f"Project {project.name!r} does not define a static version")

    return name, version


def run_command(command: list[str], cwd: Path) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_wheelhouse() -> None:
    if WHEELHOUSE.exists():
        shutil.rmtree(WHEELHOUSE)
    WHEELHOUSE.mkdir(parents=True)


def build_wheel(project: Project) -> Path:
    load_project_metadata(project)

    before = set(WHEELHOUSE.glob("*.whl"))
    run_command(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(WHEELHOUSE),
            str(project.path),
        ],
        cwd=ROOT,
    )
    after = set(WHEELHOUSE.glob("*.whl"))
    built = sorted(after - before)

    if len(built) != 1:
        raise SystemExit(
            f"Expected exactly one wheel for {project.name}, found {len(built)}"
        )

    return built[0]


def output_file() -> Path:
    name, version = load_project_metadata(PROJECTS[0])
    return DIST / f"{name}-{version}-{TARGET_PLATFORM}.ps1"


def pack_environment(wheels: list[Path]) -> Path:
    output = output_file()
    if output.exists():
        output.unlink()

    command = [
        "pixi-pack",
        "--environment",
        "default",
        "--platform",
        TARGET_PLATFORM,
        "--output-file",
        str(output),
        "--ignore-pypi-non-wheel",
        "--create-executable",
    ]
    for wheel in wheels:
        command.extend(["--inject", str(wheel)])

    run_command(command, cwd=ROOT)

    if not output.is_file():
        raise SystemExit(f"pixi-pack did not create {output}")

    return output


def main() -> int:
    prepare_wheelhouse()
    wheels = [build_wheel(project) for project in PROJECTS]
    output = pack_environment(wheels)
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
