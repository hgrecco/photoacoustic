from enum import Enum, auto
import threading
import time
import warnings

from constants import OPTIONS
from models import Experiment, MeasurementFile, PowerScan
import pathlib
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from plotting import build_pdf, save_all_figures
from xlsx import write_excel


class Action(Enum):
    READ_ABSORBANCE = auto()
    READ_MEASUREMENT_FILE = auto()
    STOP_ANALYSIS = auto()
    SKIP = auto()


class ExperimentEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        experiment: Experiment,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()

        self.experiment = experiment
        self.stop_event = stop_event
        save_all_figures(self.experiment)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        time.sleep(0.01)
        p = pathlib.Path(str(event.src_path))
        relative_path = p.relative_to(self.experiment.root)
        # TODO: move this processing to a differen thread so that I don't have
        # queue pile up problems
        print(f"creation event for: {p.name}")
        match self.determine_action(p):
            case Action.READ_ABSORBANCE:
                self.experiment.set_absorbance(p)
                save_all_figures(self.experiment)
                print(self.experiment)
            case Action.READ_MEASUREMENT_FILE:
                if p.parent not in self.experiment.powerscans.keys():
                    self.experiment.powerscans[p.parent] = PowerScan.from_path(p.parent)
                print(f"Adding file {relative_path.name} to {p.parent}")
                self.experiment.powerscans[p.parent].measurement_files[p] = (
                    MeasurementFile.from_path(p)
                )
                save_all_figures(self.experiment)
                print(self.experiment)
            case Action.STOP_ANALYSIS:
                print("stopping analysis")
                self.experiment.done = True
                self.stop_event.set()
            case Action.SKIP:
                pass

    def determine_action(self, path: pathlib.Path) -> Action:
        action = Action.SKIP

        if not path.is_relative_to(self.experiment.root):
            print(f"{path.name} not within {self.experiment.root}")
            return action

        if not path.name.endswith(".txt"):
            print(f"{path.name}: skipping, not a .txt file")
            return action

        if path.name.startswith("_"):
            print(f"{path.name}: skipping for user prefix")
            return action

        relative_path = path.relative_to(self.experiment.root)
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


def main():

    root = pathlib.Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    exp = Experiment.from_path(root)

    print(exp)

    stop_event = threading.Event()
    event_handler = ExperimentEventHandler(exp, stop_event)
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
            time.sleep(1)
        observer.stop()
        observer.join()
        print("building pdf summary")
        build_pdf(root)
        print("building excel summary")
        write_excel(root, exp)
        print("finishing up the analysis")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
