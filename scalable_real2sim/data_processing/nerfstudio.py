import concurrent.futures
import json
import logging
import os
import re
import shutil

import numpy as np

from PIL import Image
from tqdm import tqdm

from .alpha_channel import add_alpha_channel
from .image_subsampling import select_and_copy_dissimilar_images
from .masks import invert_masks_in_directory


def convert_txt_or_png_to_nerfstudio_depth(
    folder_path: str,
    output_folder: str,
    image_folder: str = None,
    max_depth_value: int = 65535,
    bit_depth: int = 16,
) -> None:
    """Converts depth data from .txt or .png files to the format expected by Nerfstudio.

    This function processes all relevant files in the specified folder, reading depth data
    from either .txt or .png files. The depth values are assumed to be in meters for .txt
    files and in millimeters for .png files. The processed depth data is saved in the
    specified output folder.
    Background depth values are set to max_depth_value.

    Args:
        folder_path (str): The path to the folder containing the input .txt or .png files.
        output_folder (str): The path to the folder where the output depth data will be saved.
        image_folder (str, optional): An optional path to an image folder (default is None).
            If provided, the function will adjust the depth values based on the transparency
            (alpha channel) of the corresponding image.
        max_depth_value (int, optional): The maximum depth value to consider (default is 65535).
        bit_depth (int, optional): The bit depth of the output data (default is 16).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Regular expression to extract numeric part from filenames
    number_pattern = re.compile(r"(\d+)")

    # List all relevant files in the folder (both .txt and .png)
    files = [f for f in os.listdir(folder_path) if f.endswith((".txt", ".png"))]

    def process_file(filename):
        file_path = os.path.join(folder_path, filename)

        # Handle both .txt and .png files
        if filename.endswith(".txt"):
            # Extract numeric part from the filename
            match = number_pattern.search(filename)
            if not match:
                logging.warning(f"No numeric part found in {filename}. Skipping.")
                return
            numeric_part = match.group(1)

            # Read depth data from the text file
            try:
                depth_data_meters = np.loadtxt(file_path)
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
                return
        elif filename.endswith(".png"):
            # Extract numeric part from the filename
            match = number_pattern.search(filename)
            if not match:
                logging.warning(f"No numeric part found in {filename}. Skipping.")
                return
            numeric_part = match.group(1)

            # Read depth data from the PNG file (assuming it contains depth values)
            try:
                depth_image = Image.open(file_path).convert(
                    "I"
                )  # 'I' mode for 32-bit pixels
                depth_data_meters = (
                    np.array(depth_image) / 1000.0
                )  # Assuming PNG stores millimeters
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
                return
        else:
            return  # Skip any files that aren't .txt or .png

        # Convert depth from meters to millimeters
        depth_data_mm = depth_data_meters * 1000.0

        # Clip depth values to the specified maximum depth value
        depth_data_mm_clipped = np.clip(depth_data_mm, 0, max_depth_value)

        # If image_folder is provided, adjust depth values based on transparency
        if image_folder:
            # Look for the corresponding image file in the image folder
            image_file_found = False
            for image_filename in os.listdir(image_folder):
                if image_filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    image_match = number_pattern.search(image_filename)
                    if image_match and numeric_part == image_match.group(1):
                        image_file_path = os.path.join(image_folder, image_filename)
                        image_file_found = True
                        break

            if not image_file_found:
                return

            # Open the corresponding image to check for transparency
            try:
                # Open the image and ensure it has an alpha channel
                image = Image.open(image_file_path).convert("RGBA")
                alpha_channel = image.split()[3]  # Extract the alpha channel
                alpha_array = np.array(alpha_channel)

                # Create a mask where alpha == 0 (transparent pixels)
                transparent_mask = alpha_array == 0

                # Ensure the mask dimensions match the depth data dimensions
                if transparent_mask.shape != depth_data_mm_clipped.shape:
                    logging.warning(
                        f"Dimension mismatch between depth data and alpha "
                        f"mask in {filename}."
                    )
                    return

                # Set depth values to max_depth_value where the image is transparent
                depth_data_mm_clipped[transparent_mask] = max_depth_value

            except Exception as e:
                logging.error(f"Error processing image file {image_file_path}: {e}")
                return

        # Convert depth data to the specified bit depth
        if bit_depth == 16:
            depth_data_uint = depth_data_mm_clipped.astype(np.uint16)
        elif bit_depth == 32:
            depth_data_uint = depth_data_mm_clipped.astype(np.uint32)
        else:
            logging.error(
                f"Unsupported bit depth: {bit_depth}. Supported values are 16 and 32."
            )
            return

        # Construct the output filename
        base_filename = os.path.splitext(filename)[0]
        if bit_depth == 16:
            output_filename = base_filename + ".png"
        else:  # For 32-bit, use TIFF format
            output_filename = base_filename + ".tiff"
        output_file_path = os.path.join(output_folder, output_filename)

        # Save the depth data as an image
        try:
            depth_image = Image.fromarray(depth_data_uint)
            if bit_depth == 16:
                depth_image.save(output_file_path, format="PNG")
            else:  # Save as TIFF for 32-bit depth
                depth_image.save(output_file_path, format="TIFF")
        except Exception as e:
            logging.error(f"Error saving {output_filename}: {e}")

    # Use ThreadPoolExecutor to speed up the processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_file, files),
                total=len(files),
                desc="Processing files",
            )
        )


def transform_pose_opengl_to_opencv(pose: np.ndarray) -> np.ndarray:
    """Converts a pose from OpenGL to OpenCV format or vice versa (inverse transform
    is identical).

    This function takes a 4x4 pose matrix in OpenGL format and converts it to OpenCV
    format. The conversion involves flipping the y and z axes of the pose matrix.
    """
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz


def convert_bundle_sdf_poses_to_nerfstudio_poses(folder_path: str) -> None:
    """Converts BundleSDF poses to the format expected by Nerfstudio. The poses are
    overwritten in place.
    """
    with os.scandir(folder_path) as paths:
        for path in paths:
            X_CW_opencv = np.loadtxt(path.path)
            X_WC_opencv = np.linalg.inv(X_CW_opencv)
            X_WC_opengl = transform_pose_opengl_to_opencv(X_WC_opencv)
            np.savetxt(path.path, X_WC_opengl)


def read_intrinsics(cam_K_path):
    """
    Reads the camera intrinsics from cam_K.txt.
    """
    K = np.loadtxt(cam_K_path)
    if K.shape != (3, 3):
        raise ValueError("Camera intrinsics should be a 3x3 matrix.")
    return K


def read_pose(pose_path):
    """
    Reads the camera pose from a text file.
    """
    pose = np.loadtxt(pose_path)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose file {pose_path} should contain a 4x4 matrix.")
    return pose


def collect_files(directory, extensions):
    """
    Collects files from a directory with specified extensions.
    Returns a dictionary mapping from numerical filenames to file paths.
    """
    files = {}
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext.lower() in extensions and name.isdigit():
            idx = int(name)
            files[idx] = os.path.join(directory, filename)
    return files


def downsample_indices(indices, num_images):
    """
    Downsamples the list of indices to the desired number of images.
    """
    if num_images >= len(indices) or num_images <= 0:
        return indices  # Return all indices if num_images is invalid
    indices = sorted(indices)
    interval = len(indices) / num_images
    selected_indices = [indices[int(i * interval)] for i in range(num_images)]
    return selected_indices


def create_nerfstudio_transforms_json(
    data_dir: str,
    output_path: str,
    use_depth: bool = False,
    depth_dir: str = None,
    use_masks: bool = False,
    masks_dir: str = None,
    num_images: int = None,
) -> None:
    """
    Creates the transforms.json file for Nerfstudio.

    Args:
        data_dir (str): The directory containing the camera intrinsics, images, and poses.
            The directory should contain the following files:
            - cam_K.txt: The camera intrinsics.
            - images: The directory containing the images.
            - poses: The directory containing the poses.
            - optional: depth: The directory containing the depth files.
            - optional: gripper_masks: The directory containing the gripper masks.
        output_path (str): The path where the transforms.json file will be saved.
        use_depth (bool, optional): Whether to include depth information. Defaults to False.
        depth_dir (str, optional): The directory containing depth files. Defaults to None.
        use_masks (bool, optional): Whether to include mask information. Defaults to False.
        masks_dir (str, optional): The directory containing mask files. Defaults to None.
        num_images (int, optional): The number of images to include in the output. Defaults to None.
    """
    # Paths to required files and directories
    cam_K_path = os.path.join(data_dir, "cam_K.txt")
    images_dir = os.path.join(data_dir, "images")
    poses_dir = os.path.join(data_dir, "poses")

    if use_depth:
        depth_dir = depth_dir if depth_dir else os.path.join(data_dir, "depth")
    if use_masks:
        masks_dir = masks_dir if masks_dir else os.path.join(data_dir, "gripper_masks")

    # Read camera intrinsics
    K = read_intrinsics(cam_K_path)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Collect image files
    image_files = collect_files(
        images_dir, extensions={".jpg", ".png", ".jpeg", ".tiff"}
    )
    pose_files = collect_files(poses_dir, extensions={".txt"})

    if use_depth:
        depth_files = collect_files(depth_dir, extensions={".png", ".jpg", ".exr"})
    if use_masks:
        mask_files = collect_files(masks_dir, extensions={".png", ".jpg", ".bmp"})

    # Ensure that we have matching images and poses
    indices = sorted(set(image_files.keys()) & set(pose_files.keys()))

    # Downsample indices if num_images is specified
    if num_images is not None:
        indices = downsample_indices(indices, num_images)

    frames = []
    for idx in indices:
        image_path = os.path.relpath(image_files[idx], data_dir)
        pose_path = pose_files[idx]

        # Read pose
        transform_matrix = read_pose(pose_path)
        transform_matrix = transform_matrix.tolist()

        frame = {
            "file_path": image_path,
            "transform_matrix": transform_matrix,
        }

        if use_depth and idx in depth_files:
            depth_path = os.path.relpath(depth_files[idx], data_dir)
            frame["depth_file_path"] = depth_path

        if use_masks and idx in mask_files:
            mask_path = os.path.relpath(mask_files[idx], data_dir)
            frame["mask_path"] = mask_path

        frames.append(frame)

    # Collect image dimensions from the first image
    if frames:
        sample_image_path = os.path.join(data_dir, frames[0]["file_path"])
        from PIL import Image

        with Image.open(sample_image_path) as img:
            w, h = img.size
    else:
        raise ValueError("No frames available to process.")

    # Construct the transforms.json data
    transforms = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_model": "OPENCV",
        "frames": frames,
    }

    # Save to output path
    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=4)

    logging.info(f"transforms.json file saved to {output_path}")


def preprocess_data_for_nerfstudio(
    data_dir: str, output_dir: str, num_images: int | None = None
) -> None:
    """Preprocesses the data for Nerfstudio.

    Args:
        data_dir (str): The data directory after BundleSDF processing.
        output_dir (str): The directory where the preprocessed data will be saved.
        num_images (int, optional): Number of images to sample. If None, uses all images.
    """
    # Step 0: Copy input directory to output directory
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"cp -r {data_dir}/* {output_dir}")

    # Create temporary directories for intermediate processing
    rgb_alpha_dir = os.path.join(output_dir, "rgb_alpha")
    depth_converted_dir = os.path.join(output_dir, "depth_converted")
    gripper_masks_inverted_dir = os.path.join(output_dir, "gripper_masks_inverted")
    images_sampled_dir = os.path.join(output_dir, "images_dino_sampled")

    # Step 1: Add alpha channel to RGB images
    add_alpha_channel(
        img_dir=os.path.join(output_dir, "rgb"),
        mask_dir=os.path.join(output_dir, "masks"),
        out_dir=rgb_alpha_dir,
    )

    # Step 2: Convert depth data to Nerfstudio format
    convert_txt_or_png_to_nerfstudio_depth(
        folder_path=os.path.join(output_dir, "depth"),
        output_folder=depth_converted_dir,
        image_folder=rgb_alpha_dir,
    )

    # Step 3: Convert poses to OpenGL format
    convert_bundle_sdf_poses_to_nerfstudio_poses(
        folder_path=os.path.join(output_dir, "ob_in_cam")
    )

    # Step 4: Invert gripper masks
    invert_masks_in_directory(
        input_dir=os.path.join(output_dir, "gripper_masks"),
        output_dir=gripper_masks_inverted_dir,
    )

    # Step 5: Sample most dissimilar images
    if num_images is not None:
        select_and_copy_dissimilar_images(
            image_dir=rgb_alpha_dir,
            output_dir=images_sampled_dir,
            K=num_images,
            model_name="dino",
        )
    else:
        logging.info("Skipping image sampling as num_images is not specified.")
        shutil.copytree(rgb_alpha_dir, images_sampled_dir)

    # Step 6: Create filtered directories based on sampled images
    depth_filtered_dir = os.path.join(output_dir, "depth_filtered")
    poses_filtered_dir = os.path.join(output_dir, "poses_filtered")
    gripper_masks_filtered_dir = os.path.join(output_dir, "gripper_masks_filtered")

    os.makedirs(depth_filtered_dir, exist_ok=True)
    os.makedirs(poses_filtered_dir, exist_ok=True)
    os.makedirs(gripper_masks_filtered_dir, exist_ok=True)

    # Get base filenames from sampled images
    sampled_files = [
        f
        for f in os.listdir(images_sampled_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    basenames = [os.path.splitext(f)[0] for f in sampled_files]

    # Copy corresponding files
    for basename in basenames:
        if os.path.exists(os.path.join(depth_converted_dir, f"{basename}.png")):
            os.system(f"cp {depth_converted_dir}/{basename}.png {depth_filtered_dir}/")
        if os.path.exists(os.path.join(output_dir, "ob_in_cam", f"{basename}.txt")):
            os.system(f"cp {output_dir}/ob_in_cam/{basename}.txt {poses_filtered_dir}/")
        if os.path.exists(os.path.join(gripper_masks_inverted_dir, f"{basename}.png")):
            os.system(
                f"cp {gripper_masks_inverted_dir}/{basename}.png {gripper_masks_filtered_dir}/"
            )

    # Step 7: Clean up and rename directories
    cleanup_dirs = [
        "rgb",
        "rgb_alpha",
        "masks",
        "gripper_masks",
        "gripper_masks_inverted",
        "depth",
        "depth_converted",
        "ob_in_cam",
    ]
    for d in cleanup_dirs:
        path = os.path.join(output_dir, d)
        if os.path.exists(path):
            os.system(f"rm -rf {path}")

    # Rename filtered directories to final names
    os.system(f"mv {gripper_masks_filtered_dir} {output_dir}/gripper_masks")
    os.system(f"mv {depth_filtered_dir} {output_dir}/depth")
    os.system(f"mv {images_sampled_dir} {output_dir}/images")
    os.system(f"mv {poses_filtered_dir} {output_dir}/poses")

    # Remove any .mp4 files
    os.system(f"rm -f {output_dir}/*.mp4")

    # Step 8: Create transforms.json
    create_nerfstudio_transforms_json(
        data_dir=output_dir,
        output_path=os.path.join(output_dir, "transforms.json"),
        use_depth=True,
        use_masks=True,
        num_images=num_images,
    )

    logging.info("Done preprocessing data for Nerfstudio.")
