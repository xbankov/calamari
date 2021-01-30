import random
import shutil

from tqdm import tqdm

from cfg import DATA_ROOT, MANUAL_SPLITS

random.seed(42)



def main():
    root = DATA_ROOT.parent / 'dta19'
    new_train = MANUAL_SPLITS / 'train'
    new_valid = MANUAL_SPLITS / 'valid'
    new_test = MANUAL_SPLITS / 'test'
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
