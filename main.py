from models import Experiment
import pathlib


def main():
    root = pathlib.Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    exp = Experiment.from_path(root)

    print(exp)
    p = input("\nenter path when updated: ")
    while p != "\n":
        p = pathlib.Path(p)
        exp.update(p)
        print(exp)
        p = input("\nenter path when updated: ")


if __name__ == "__main__":
    main()
