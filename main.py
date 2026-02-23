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
        time.sleep(0.1)
        p = pathlib.Path(str(event.src_path))
        self.experiment.update(p)
        save_all_figures(self.experiment)
        print(self.experiment)


def main():
    warnings.catch_warnings(action="ignore")
    root = pathlib.Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    exp = Experiment.from_path(root)

    print(exp)

    event_handler = ExperimentEventHandler(exp)
    observer = Observer()
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
