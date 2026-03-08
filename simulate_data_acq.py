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

    paths = initialize_watch_folder(source_path, paths, dest_path, random.randint(1, 5))

    input("press enter to start data copying\n")

    for path in paths:
        print(path)
        copy_path = dest_path / path.relative_to(source_path)
        print(f"coying {path} to {copy_path}")
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        shutil.copy(path, copy_path)

        time.sleep(1)
    shutil.copy2(done_path, dest_path)


if __name__ == "__main__":
    main()
