import argparse
import math
import os
import shutil

from scalable_real2sim.data_processing.image_subsampling import (
    select_and_copy_dissimilar_images,
)
from scalable_real2sim.segmentation.segment_moving_object_data import (
    segment_moving_obj_data,
)


def downsample_images(rgb_dir: str, num_images: int) -> None:
    # Check if we need to subsample images.
    rgb_files = sorted(os.listdir(rgb_dir))
    if len(rgb_files) > num_images:
        print(f"Found {len(rgb_files)} images, subsampling to {num_images}...")

        # Move original images to rgb_original.
        rgb_original_dir = os.path.join(rgb_dir, "..", "rgb_original")
        os.makedirs(rgb_original_dir, exist_ok=True)
        for f in rgb_files:
            shutil.move(os.path.join(rgb_dir, f), os.path.join(rgb_original_dir, f))

        num_uniform_frames = num_images // 2  # Big gaps cause tracking to fail
        max_frame_gap = math.floor(len(rgb_files) / num_uniform_frames)
        print(f"Using a maximum frame gap of {max_frame_gap} for image downsampling.")
        select_and_copy_dissimilar_images(
            image_dir=rgb_original_dir,
            output_dir=rgb_dir,
            K=num_images,
            N=max_frame_gap,
            model_name="dino",
        )
        print("Image subsampling complete.")
    else:
        print("No image subsampling needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rgb_dir", type=str, help="Path to the folder containing RGB frames"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the folder to save binary masks where 1 indicates the object "
        "of interest and 0 indicates the background.",
    )
    parser.add_argument(
        "--txt_prompt",
        type=str,
        help="Text prompt to use for grounding the object of interest.",
        default=None,
    )
    parser.add_argument(
        "--txt_prompt_index",
        type=int,
        help="Index of the frame to use for grounding the object of interest.",
        default=0,
    )
    parser.add_argument(
        "--neg_txt_prompt",
        type=str,
        help="Text prompt to use for grounding the object to ignore.",
        default=None,
    )
    parser.add_argument(
        "--num_neg_frames",
        type=int,
        help="Number of frames to add negatives to (uniformly spaced). Increasing this "
        "might lead to OOM.",
        default=10,
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        help="Path to the folder to save images with predicted bounding boxes and "
        "query points overlaid.",
        default=None,
    )
    parser.add_argument(
        "--gui_frames",
        type=str,
        nargs="*",
        help="List of RGB image names (without extension) to provide GUI labels for. "
        "NOTE: Segmentation will start from the first frame in the list. Hence, if you "
        "want to segment all images, you should specify the first frame.",
        default=None,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to subsample to.",
        default=None,
    )
    parser.add_argument(
        "--gripper_sam2_path",
        type=str,
        help="Path to custom SAM2 model for gripper segmentation.",
        default=None,
    )
    parser.add_argument(
        "--gripper_grounding_dino_path",
        type=str,
        help="Path to custom GroundingDINO model for gripper object detection.",
        default=None,
    )
    args = parser.parse_args()
    rgb_dir = args.rgb_dir
    output_dir = args.output_dir
    txt_prompt = args.txt_prompt
    txt_prompt_index = args.txt_prompt_index
    neg_txt_prompt = args.neg_txt_prompt
    num_neg_frames = args.num_neg_frames
    debug_dir = args.debug_dir
    gui_frames = args.gui_frames
    num_images = args.num_images
    gripper_sam2_path = args.gripper_sam2_path
    gripper_grounding_dino_path = args.gripper_grounding_dino_path

    if num_images is not None:
        downsample_images(rgb_dir, num_images)

    segment_moving_obj_data(
        rgb_dir=rgb_dir,
        output_dir=output_dir,
        txt_prompt=txt_prompt,
        txt_prompt_index=txt_prompt_index,
        neg_txt_prompt=neg_txt_prompt,
        num_neg_frames=num_neg_frames,
        debug_dir=debug_dir,
        gui_frames=gui_frames,
        gripper_sam2_path=gripper_sam2_path,
        gripper_grounding_dino_path=gripper_grounding_dino_path,
    )
