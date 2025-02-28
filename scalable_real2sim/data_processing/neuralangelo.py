import logging
import os
import shutil

from pathlib import Path
from types import SimpleNamespace

from scalable_real2sim.neuralangelo.projects.neuralangelo.scripts.convert_data_to_json import (
    data_to_json,
)
from scalable_real2sim.neuralangelo.projects.neuralangelo.scripts.generate_config import (
    generate_config,
)

from .alpha_channel import add_alpha_channel
from .colmap import convert_bundlesdf_to_colmap_format
from .image_subsampling import select_and_copy_dissimilar_images
from .masks import invert_masks_in_directory


def process_moving_obj_data_for_neuralangelo(
    input_dir: str, output_dir: str, num_images: int = 800
) -> str:
    """Process moving object/static camera data for Neuralangelo.

    Args:
        input_dir: Path to input directory containing:
            - cam_K.txt         # Camera intrinsic parameters
            - gripper_masks/    # Masks for the gripper
            - masks/            # Masks for the object + gripper
            - ob_in_cam/        # Poses of the object in camera frame (X_CO)
            - rgb/              # RGB images
        output_dir: Path to output directory
        num_images: Number of images to sample (default: 800)

    Returns:
        cfg_path: Path to the generated config file.
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

    data_to_json(args=SimpleNamespace(data_dir=str(output_dir), scene_type="object"))

    cfg_path = generate_config(
        args=SimpleNamespace(
            data_dir=os.path.abspath(str(output_dir)),
            config_path=os.path.join(output_dir, "neuralangelo_recon.yaml"),
            sequence_name="neuralangelo_recon",
            scene_type="object",
            auto_exposure_wb=True,
            val_short_size=300,
        )
    )

    logging.info("Done preprocessing data for Neuralangelo.")

    return os.path.abspath(cfg_path)
