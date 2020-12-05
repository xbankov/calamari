import os
from glob import glob
import random
import json
from pathlib import Path

random.seed(42)


def save_cmd(split, name, mode):
    with open(name, 'w+') as f:
        if mode == 'train':
            f.write(f'calamari-train --folders {" ".join(split)}')
        elif mode == 'predict':
            f.write(f'calamari-predict --folders {" ".join(split)}')
        elif mode == 'eval':
            f.write(f'calamari-eval --gt eval.json')
            dict = {'gt': " ".join([f'{fold}/*.gt.txt' for fold in split])}
            with open(Path(name).with_suffix('.json'), 'w+') as js:
                json.dump(dict, js)


def print_cmd(split):
    print("--folders " + " ".join(split))


def main():
    folders = [os.path.basename(file) for file in glob('*')]
    training_set = random.sample(folders, int(len(folders) / 100 * 75))
    test_set = set(folders) - set(training_set)
    save_cmd(sorted(training_set), 'training.sh', 'train')
    save_cmd(sorted(test_set), 'predict.sh', 'predict')
    save_cmd(sorted(test_set), 'eval.sh', 'eval')


if __name__ == '__main__':
    main()
