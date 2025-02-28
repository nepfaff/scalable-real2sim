"""
This script subtracts binary masks from two folders.
This can be useful for subtracting the object masks from the gripper masks if the
gripper masks include the object as can happen in bad lighting conditions or when the
object has colors that are similar to the gripper.
"""

import argparse
import os

import cv2
import numpy as np

from tqdm import tqdm


def subtract_masks(folder_a: str, folder_b: str, output_folder: str) -> None:
    """
    Subtracts binary masks from two folders and saves the result in an output folder.

    This function reads binary mask images from two specified folders,
    subtracts the masks, and saves the resulting masks in the output folder.
    If both masks are white (255), the result will be set to black (0).
    The function only processes files that are common to both folders.

    Args:
        folder_a (str): Path to the first folder containing binary masks.
        folder_b (str): Path to the second folder containing binary masks.
        output_folder (str): Path to the folder where the output masks will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))

    common_files = files_a.intersection(files_b)

    for filename in tqdm(common_files):
        path_a = os.path.join(folder_a, filename)
        path_b = os.path.join(folder_b, filename)

        mask_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

        if mask_a is None or mask_b is None:
            print(f"Skipping {filename}: Could not load image")
            continue

        # Ensure same size
        if mask_a.shape != mask_b.shape:
            print(
                f"Skipping {filename}: Shape mismatch {mask_a.shape} vs {mask_b.shape}"
            )
            continue

        # Subtract masks: If both are white, set to black
        result = np.where((mask_a == 255) & (mask_b == 255), 0, mask_a)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract binary masks")
    parser.add_argument("folder_a", type=str, help="Path to first folder")
    parser.add_argument("folder_b", type=str, help="Path to second folder")
    parser.add_argument(
        "output_folder", type=str, help="Path to output folder that is mask_a - mask_b."
    )

    args = parser.parse_args()
    subtract_masks(args.folder_a, args.folder_b, args.output_folder)
