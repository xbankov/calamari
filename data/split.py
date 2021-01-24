import random
import shutil
from pathlib import Path

from tqdm import tqdm

random.seed(42)



def main():
    root = Path('/home/xbankov/dta19/')
    new_train = Path('/home/xbankov/manual-splits/train')
    new_valid = Path('/home/xbankov/manual-splits/valid')
    new_test = Path('/home/xbankov/manual-splits/test')
    folders = [folder for folder in root.glob('*')]
    training_set = random.sample(folders, int(len(folders) / 100 * 75))
    valid_set = random.sample(training_set, int(len(training_set) / 100 * 25))
    test_set = set(folders) - set(training_set)

    for folder in tqdm(training_set):
        if not (new_train / folder.stem).exists():
            shutil.copytree(folder, new_train / folder.stem)

    for folder in tqdm(valid_set):
        if not (new_valid / folder.stem).exists():
            shutil.copytree(folder, new_valid / folder.stem)

    for folder in tqdm(test_set):
        if not (new_test / folder.stem).exists():
            shutil.copytree(folder, new_test/ folder.stem)





if __name__ == '__main__':
    main()
