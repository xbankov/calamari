import shutil
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from data.cfg import SRGAN_PATH


def srgan_set(lr_directory, sr_directory):
    sr_directory.mkdir(parents=True, exist_ok=True)

    # Change model input shape to accept all size inputs
    model = keras.models.load_model(SRGAN_PATH)
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    for folder in lr_directory.iterdir():
        (sr_directory / folder.name).mkdir(parents=True, exist_ok=True)
        for file in tqdm(folder.iterdir(), total=len(list(folder.iterdir()))):
            new_file = sr_directory / file.parent.name / file.name
            if file.suffix == '.png':
                # Read image
                low_res = cv2.imread(str(file), 1)

                # Convert to RGB (opencv uses BGR as default)
                low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

                # Rescale to 0-1.
                low_res = low_res / 255.0

                # Get super resolution image
                sr = model.predict(np.expand_dims(low_res, axis=0))[0]

                # Rescale values in range 0-255
                sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

                # Convert back to BGR for opencv
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                # Save the results:
                cv2.imwrite(str(new_file), sr)

            elif file.suffixes == ['.gt', '.txt']:
                shutil.copy(file, new_file)


def main():
    # Inside mounted docker image
    root = Path('/data/')
    train = root / 'train'
    valid = root / 'valid'
    test = root / 'test'
    srgan_set(Path(str(train) + '_downscaled'), Path(str(train) + '_srgan'))
    srgan_set(Path(str(valid) + '_downscaled'), Path(str(valid) + '_srgan'))
    srgan_set(Path(str(test) + '_downscaled'), Path(str(test) + '_srgan'))


if __name__ == '__main__':
    main()
