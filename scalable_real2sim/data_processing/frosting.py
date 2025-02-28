import logging
import shutil
import subprocess

from pathlib import Path

from .alpha_channel import add_alpha_channel
from .colmap import convert_bundlesdf_to_colmap_format
from .image_subsampling import select_and_copy_dissimilar_images
from .masks import invert_masks_in_directory


def process_moving_obj_data_for_sugar(
    input_dir: str, output_dir: str, num_images: int = 800, use_depth: bool = False
):
    """Process moving object/static camera data for SuGAR.

    Args:
        input_dir: Path to input directory containing:
            - cam_K.txt         # Camera intrinsic parameters
            - gripper_masks/    # Masks for the gripper
            - masks/            # Masks for the object + gripper
            - ob_in_cam/        # Poses of the object in camera frame (X_CO)
            - rgb/              # RGB images
        output_dir: Path to output directory
        num_images: Number of images to sample (default: 800)
        use_depth: Whether to use depth images (default: False)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Step 0: Copy input directory to output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    rgb_alpha_dir = output_dir / "rgb_alpha"
    add_alpha_channel(
        img_dir=str(output_dir / "rgb"),
        mask_dir=str(output_dir / "masks"),
        out_dir=str(rgb_alpha_dir),
    )

    gripper_masks_inverted_dir = output_dir / "gripper_masks_inverted"
    invert_masks_in_directory(
        input_dir=str(output_dir / "gripper_masks"),
        output_dir=str(gripper_masks_inverted_dir),
    )

    images_dino_sampled_dir = output_dir / "images_dino_sampled"
    if len(list((rgb_alpha_dir).glob("*.png"))) >= num_images:
        select_and_copy_dissimilar_images(
            image_dir=str(rgb_alpha_dir),
            output_dir=str(images_dino_sampled_dir),
            K=num_images,
            model_name="dino",
        )
    else:
        logging.info(
            f"Skipping image selection: not enough images "
            f"(found: {len(list((rgb_alpha_dir).glob('*.png')))}, required: {num_images})"
        )
        shutil.copytree(rgb_alpha_dir, images_dino_sampled_dir)

    shutil.rmtree(output_dir / "gripper_masks", ignore_errors=True)
    shutil.rmtree(output_dir / "gripper_masks", ignore_errors=True)
    (output_dir / "gripper_masks_inverted").rename(output_dir / "gripper_masks")

    shutil.rmtree(output_dir / "images", ignore_errors=True)
    (output_dir / "images_dino_sampled").rename(output_dir / "images")

    shutil.rmtree(output_dir / "poses", ignore_errors=True)
    (output_dir / "ob_in_cam").rename(output_dir / "poses")

    convert_bundlesdf_to_colmap_format(
        data_dir=str(output_dir), output_dir=str(output_dir)
    )

    compute_pcd_script = (
        Path(__file__).parent.parent / "Frosting/gaussian_splatting/compute_pcd_init.sh"
    )
    subprocess.run(["bash", str(compute_pcd_script), str(output_dir)], check=True)

    if use_depth:
        convert_depth_script = (
            Path(__file__).parent.parent
            / "Frosting/gaussian_splatting/utils/make_depth_scales.py"
        )
        args = [
            "--base_dir",
            str(output_dir),
            "--depths_dir",
            str(output_dir / "depths"),
        ]
        subprocess.run(["python", str(convert_depth_script), *args], check=True)
    else:
        logging.info("Skipping depth image conversion.")

    logging.info("Done preprocessing data for Frosting.")
