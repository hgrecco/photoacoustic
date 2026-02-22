import time

from models import Experiment
import pathlib
from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


class ExperimentEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        experiment: Experiment,
    ) -> None:
        super().__init__()

        self.experiment = experiment

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        time.sleep(0.1)
        p = pathlib.Path(str(event.src_path))
        self.experiment.update(p)
        print(self.experiment)


def main():
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
