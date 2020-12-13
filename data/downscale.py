from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--directory", required=True,
                        help="Directory to save new images and folders into!")
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Folders containing images to downscale")
    parser.add_argument("--scale_factor", help="Downscale factor: 2 for twice smaller image.", default=2)
    args = parser.parse_args()

    root = Path(args.directory)
    print("Resolving folders")

    for folder in args.folders:
        folder_name = Path(folder).stem
        image_files = glob(folder + '/*.png')
        gt_files = glob(folder + '/*.txt')
        (root / folder_name).mkdir(parents=True, exist_ok=True)

        for image in tqdm(image_files, desc="Downscaling images"):
            image_name = Path(image).name
            new_filename = root / folder_name / image_name
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            width = int(img.shape[1] // args.scale_factor)
            height = int(img.shape[0] // args.scale_factor)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(new_filename), resized)

        for gt in tqdm(gt_files, desc="Copying ground truth .txt files"):
            filename = Path(gt).name
            new_file = root / folder_name / filename
            shutil.copy(filename, new_file)  #


if __name__ == '__main__':
    main()
