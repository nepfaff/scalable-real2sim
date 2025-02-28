import concurrent.futures

from pathlib import Path

import numpy as np

from PIL import Image
from tqdm import tqdm


def load_mask(mask_path: Path) -> np.ndarray:
    """Loads a mask, whether it's in .npy or .png format."""
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    elif mask_path.suffix == ".png":
        mask = np.asarray(Image.open(mask_path).convert("L"))
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
    return mask


def process_image_with_alpha(
    img_path: Path, mask_path: Path, out_dir_path: Path
) -> None:
    img = np.asarray(Image.open(img_path).convert("RGB"))
    mask = load_mask(mask_path)

    # Ensure the mask has the correct shape for concatenation.
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    img_w_alpha = np.concatenate((img, mask), axis=-1)

    out_path = out_dir_path / img_path.name
    img_w_alpha_pil = Image.fromarray(img_w_alpha)
    img_w_alpha_pil.save(out_path)


def add_alpha_channel(img_dir: str, mask_dir: str, out_dir: str) -> None:
    """Adds an alpha channel to images using corresponding masks.

    This function processes all PNG images in the specified image directory
    and combines them with masks from the specified mask directory. The
    resulting images with an alpha channel are saved in the specified output
    directory. The function ensures that the number of images matches the
    number of masks.

    Args:
        img_dir (str): The directory containing the input images.
        mask_dir (str): The directory containing the input masks (in .npy or .png format).
        out_dir (str): The directory where the output images with alpha channels will be
            saved.

    Raises:
        AssertionError: If no images or masks are found, or if the number of images
                        does not match the number of masks.
    """
    image_dir_path = Path(img_dir)
    mask_dir_path = Path(mask_dir)
    out_dir_path = Path(out_dir)

    image_files = sorted(list(image_dir_path.glob("*.png")))
    mask_files = sorted(
        list(mask_dir_path.glob("*.npy")) + list(mask_dir_path.glob("*.png"))
    )

    assert len(image_files) > 0, f"No images found in {image_dir_path}"
    mask_files = sorted(
        [mask for mask in mask_files if mask.stem in {img.stem for img in image_files}]
    )
    assert len(mask_files) > 0, f"No matching masks found in {mask_dir_path}"
    assert len(image_files) == len(mask_files), "Number of images and masks must match."
    out_dir_path.mkdir(exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda args: process_image_with_alpha(*args),
                    zip(image_files, mask_files, [out_dir_path] * len(image_files)),
                ),
                total=len(image_files),
                desc="Adding alpha channel",
            )
        )
