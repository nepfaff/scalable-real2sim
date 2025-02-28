import concurrent.futures
import os

from PIL import Image
from tqdm import tqdm


def invert_binary_mask(input_path: str, output_path: str) -> None:
    """Inverts a binary mask.

    This function opens an image file, converts it to grayscale, inverts the pixel
    values, and saves the inverted image to a new file.
    """
    # Open the image
    img = Image.open(input_path)

    # Convert the image to grayscale (if it's not already)
    img = img.convert("L")

    # Invert the binary mask
    inverted_img = Image.eval(img, lambda x: 255 - x)

    # Save the inverted image
    inverted_img.save(output_path)


def invert_masks_in_directory(input_dir: str, output_dir: str) -> None:
    """Inverts all binary masks in a directory.

    This function processes all PNG files in the specified input directory,
    inverts their pixel values, and saves the inverted images to the specified
    output directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all PNG files in the input directory
    mask_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    def process_mask(mask_file):
        input_path = os.path.join(input_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)
        invert_binary_mask(input_path, output_path)

    # Invert each mask and save to the output directory using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_mask, mask_files),
                total=len(mask_files),
                desc="Inverting masks",
            )
        )
