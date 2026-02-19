import time
import shutil
import os
from pathlib import Path
import random


def main():
    source_path = Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/data/cilindro"
    )
    dest_path = Path(
        "/home/tomi/Documents/academicos/doc/projects/photoacoustic/git/photoacoustic/test/watch_folder"
    )
    paths = [p for p in source_path.rglob("*") if not p.is_dir()]
    random.shuffle(paths)

    for path in paths:
        print(path)
        if path.name == "done.txt":
            done_path = path
            continue
        copy_path = dest_path / path.relative_to(source_path)
        print(f"coying {path} to {copy_path}")
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        shutil.copy(path, copy_path)

        time.sleep(1)
    shutil.copy2(done_path, dest_path)


if __name__ == "__main__":
    main()
