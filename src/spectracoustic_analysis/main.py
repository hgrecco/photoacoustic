import pathlib
import queue
import sys
import threading
import time
from enum import Enum, auto

import matplotlib
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileSystemEventHandler,
)
from watchdog.observers.polling import PollingObserver

from .constants import Options, default_options
from .models import Experiment, MeasurementFile, PowerScan
from .plotting import build_pdf, save_all_figures
from .xlsx import write_excel


class Action(Enum):
    READ_ABSORBANCE = auto()
    READ_MEASUREMENT_FILE = auto()
    STOP_ANALYSIS = auto()
    SKIP = auto()


class ExperimentEventHandler(FileSystemEventHandler):
    def __init__(self, experiment: Experiment, queue: queue.Queue) -> None:
        super().__init__()

        self.experiment = experiment
        self.queue = queue
        save_all_figures(self.experiment)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        time.sleep(0.01)
        p = pathlib.Path(str(event.src_path))
        self.queue.put(p)


def determine_action(experiment: Experiment, path: pathlib.Path) -> Action:
    action = Action.SKIP

    if not path.is_relative_to(experiment.root):
        print(f"{path.name} not within {experiment.root}")
        return action

    if not path.name.endswith(".txt"):
        print(f"{path.name}: skipping, not a .txt file")
        return action

    if path.name.startswith("_"):
        print(f"{path.name}: skipping for user prefix")
        return action

    relative_path = path.relative_to(experiment.root)
    depth = len(relative_path.parts)

    if depth == 1:
        if path.name == "abs.txt":
            action = Action.READ_ABSORBANCE
        elif path.name == "done.txt":
            action = Action.STOP_ANALYSIS
        else:
            print(
                f"{path}: skipping. It does not comply with the experiment file structure format."
            )
    elif depth == 2:
        action = Action.READ_MEASUREMENT_FILE
    else:
        print(
            f"Ignoring file or directory with depth {depth} in the experiment file structure"
        )
    return action


def process_path(experiment: Experiment, q: queue.Queue, stop_event: threading.Event):
    if q.empty():
        return
    p = q.get()
    relative_path = p.relative_to(experiment.root)
    print(f"creation event for: {p.name}")
    match determine_action(experiment, p):
        case Action.READ_ABSORBANCE:
            print("ABSORBANCE")
            experiment.set_absorbance(p)
            save_all_figures(experiment)
            print(experiment)
        case Action.READ_MEASUREMENT_FILE:
            print("READ MEAS FILE")
            if p.parent not in experiment.powerscans.keys():
                experiment.powerscans[p.parent] = PowerScan.from_path(
                    p.parent, experiment.options
                )
            print(f"Adding file {relative_path.name} to {p.parent}")
            experiment.powerscans[p.parent].measurement_files[p] = (
                MeasurementFile.from_path(p, experiment.options)
            )
            save_all_figures(experiment)
            print(experiment)
        case Action.STOP_ANALYSIS:
            print("STOP ANALYSIS")
            experiment.done = True
            stop_event.set()
        case Action.SKIP:
            print("SKIP")
            pass


def main(root: pathlib.Path | str, options: Options | None = None):
    if not options:
        options = default_options()

    matplotlib.use("Agg")

    root = pathlib.Path(root)
    exp = Experiment.from_path(root, options)

    print(exp)

    q = queue.Queue()
    stop_event = threading.Event()
    event_handler = ExperimentEventHandler(exp, q)
    # observer = Observer()
    observer = PollingObserver(timeout=0.01)
    observer.schedule(
        event_handler,
        str(root),
        recursive=True,
    )

    observer.start()

    try:
        while not stop_event.is_set():
            process_path(experiment=exp, q=q, stop_event=stop_event)
            time.sleep(0.1)
        observer.stop()
        observer.join()
        print("finishing up the analysis")
        print("building excel summary")
        write_excel(root, exp)
        print("building pdf summary")
        build_pdf(root, exp.options["figures_save_path"])
    finally:
        observer.stop()
        observer.join()

    print("analysis done!")


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1])
    main(root)
