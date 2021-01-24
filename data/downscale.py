#!/usr/bin/env python3

import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def downscale_set(old, new, scale=4):
    new.mkdir(parents=True, exist_ok=True)
    all_missing = []

    for folder in old.iterdir():
        missing = []
        (new / folder.name).mkdir(parents=True, exist_ok=True)
        for file in tqdm(folder.iterdir(), total=len(list(folder.iterdir()))):
            new_file = new / file.parent.name / file.name
            if file.suffix == '.png':
                try:
                    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        width = int(img.shape[1] // scale)
                        height = int(img.shape[0] // scale)
                        dim = (width, height)
                        resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(str(new_file), resized)

                except cv2.error:
                    print(f"Error with file {file}. Skipping ...")
                    missing.append(file.with_suffix("").stem)
                except Exception:
                    print("Unexpected error")
                    missing.append(file.with_suffix("").stem)

            elif file.suffixes == ['.gt', '.txt'] and not file.with_suffix("").stem in missing:
                shutil.copy(file, new_file)
        all_missing += missing
    print(len(all_missing))

def main():
    root = Path('/home/xbankov/manual-splits/')
    train_root = root / 'train'
    valid_root = root / 'valid'
    test_root = root / 'test'
    downscale_set(train_root, Path(str(train_root) + '_downscaled'))
    downscale_set(valid_root, Path(str(valid_root) + '_downscaled'))
    downscale_set(test_root, Path(str(test_root) + '_downscaled'))

if __name__ == '__main__':
    main()
