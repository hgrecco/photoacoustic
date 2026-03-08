import time
import warnings

from models import Experiment
import pathlib
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from plotting import save_all_figures


class ExperimentEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        experiment: Experiment,
    ) -> None:
        super().__init__()

        self.experiment = experiment
        save_all_figures(self.experiment)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        time.sleep(0.01)
        p = pathlib.Path(str(event.src_path))
        # TODO: move this processing to a differen thread so that I don't have
        # queue pile up problems
        if not p.name.startswith("_") and not p.parent.name.startswith("_"):
            print(f"creation event for: {p.name}")
            self.experiment.update(p)
            save_all_figures(self.experiment)
            print(self.experiment)
        else:
            print(f"skipping for user prefix: {p.name}")


def main():
    warnings.catch_warnings(action="ignore")
    root = pathlib.Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    exp = Experiment.from_path(root)

    print(exp)

    event_handler = ExperimentEventHandler(exp)
    # observer = Observer()
    observer = PollingObserver(timeout=0.1)
    observer.schedule(
        event_handler,
        str(root),
        recursive=True,
    )

    observer.start()

    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
