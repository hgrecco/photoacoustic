import time
import shutil
import os
from pathlib import Path
import random


def initialize_watch_folder(
    source_path: Path, filepaths: list[Path], dest_path: Path, num_files: int
):
    shutil.rmtree(dest_path)
    os.mkdir(dest_path)
    assert num_files < len(filepaths)
    for i in range(num_files):
        fp = filepaths.pop(0)
        copy_path = dest_path / fp.relative_to(source_path)
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        print(f"{fp=}\n{copy_path=}\n\n")
        shutil.copy2(fp, copy_path)
    return filepaths


def sort_paths(source_path: Path, paths: list[Path]) -> list[Path]:
    sorted_paths = []
    ref0_paths = []
    ref1_paths = []
    sam_paths = []
    for path in paths:
        rpath = path.relative_to(source_path)
        match rpath.parent.name[:4]:
            case "ref0":
                ref0_paths.append(path)
            case "ref1":
                ref1_paths.append(path)
            case "sam_":
                sam_paths.append(path)
            case _:
                pass

    sorted_paths.append(source_path / "abs.txt")
    sorted_paths.extend(ref0_paths)
    sorted_paths.extend(sam_paths)
    sorted_paths.extend(ref1_paths)
    sorted_paths.append(source_path / "done.txt")
    return sorted_paths


def main():
    source_path = Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/data/70"
    )
    dest_path = Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    paths = [p for p in source_path.rglob("*") if not p.is_dir()]
    random.shuffle(paths)
    done_path = source_path / "done.txt"
    paths.remove(done_path)
    paths = sort_paths(source_path, paths)

    paths.pop(-1)

    paths = initialize_watch_folder(source_path, paths, dest_path, random.randint(1, 5))

    input("press enter to start data copying\n")

    for path in paths:
        print(f"{path.parent.name=}")
        copy_path = dest_path / path.relative_to(source_path)
        print(f"coying {path} to {copy_path}")
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        shutil.copy(path, copy_path)

        time.sleep(0.1)


if __name__ == "__main__":
    main()
