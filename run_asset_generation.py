"""
Script for generating assets from the data collected by `run_data_collection.py`.
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import time

from datetime import timedelta

import numpy as np
import torch

from tqdm import tqdm

from scalable_real2sim.data_processing.frosting import process_moving_obj_data_for_sugar
from scalable_real2sim.data_processing.image_subsampling import (
    select_and_copy_dissimilar_images,
)
from scalable_real2sim.data_processing.nerfstudio import preprocess_data_for_nerfstudio
from scalable_real2sim.data_processing.neuralangelo import (
    process_moving_obj_data_for_neuralangelo,
)
from scalable_real2sim.output.canonicalize import canonicalize_mesh_from_file
from scalable_real2sim.output.sdformat import create_sdf
from scalable_real2sim.robot_payload_id.scripts.identify_grasped_object_payload import (
    identify_grasped_object_payload,
)
from scalable_real2sim.segmentation.detect_object import detect_object
from scalable_real2sim.segmentation.segment_moving_object_data import (
    segment_moving_obj_data,
)


def downsample_images(data_dir: str, num_images: int) -> None:
    start = time.perf_counter()

    # Check if we need to subsample images.
    rgb_dir = os.path.join(data_dir, "rgb")
    rgb_files = sorted(os.listdir(rgb_dir))
    if len(rgb_files) > num_images:
        logging.info(f"Found {len(rgb_files)} images, subsampling to {num_images}...")

        # Move original images to rgb_original.
        rgb_original_dir = os.path.join(data_dir, "rgb_original")
        os.makedirs(rgb_original_dir, exist_ok=True)
        for f in rgb_files:
            shutil.move(os.path.join(rgb_dir, f), os.path.join(rgb_original_dir, f))

        num_uniform_frames = num_images // 2  # Big gaps cause tracking to fail
        max_frame_gap = math.ceil(len(rgb_files) / num_uniform_frames)
        logging.info(
            f"Using a maximum frame gap of {max_frame_gap} for image downsampling."
        )
        select_and_copy_dissimilar_images(
            image_dir=rgb_original_dir,
            output_dir=rgb_dir,
            K=num_images,
            N=max_frame_gap,
            model_name="dino",
        )
        logging.info("Image subsampling complete.")
    else:
        logging.info("No image subsampling needed.")

    logging.info(
        f"Image downsampling took {timedelta(seconds=time.perf_counter() - start)}."
    )


def run_segmentation(
    data_dir: str, output_dir: str, use_finetuned_gripper_networks: bool = False
) -> None:
    start = time.perf_counter()

    # Detect the object of interest. Need to add a dot for the DINO model.
    object_of_interest = (
        detect_object(
            image_path=os.path.join(
                data_dir, "rgb", sorted(os.listdir(os.path.join(data_dir, "rgb")))[0]
            )
        )
        + "."
    )
    torch.cuda.empty_cache()
    logging.info(f"Detected object of interest: {object_of_interest}")

    gripper_txt = (
        ("gripper")
        if use_finetuned_gripper_networks
        else (
            "Blue plastic robotic gripper with two symmetrical, curved arms "
            "attached to the end of a metallic robotic arm."
        )
    )

    # Generate the object masks.
    segment_moving_obj_data(
        rgb_dir=os.path.join(data_dir, "rgb"),
        output_dir=os.path.join(output_dir, "masks"),
        txt_prompt=object_of_interest,
        txt_prompt_index=1,
        neg_txt_prompt=gripper_txt,
    )
    torch.cuda.empty_cache()

    # Generate the gripper masks.
    segment_moving_obj_data(
        rgb_dir=os.path.join(data_dir, "rgb"),
        output_dir=os.path.join(output_dir, "gripper_masks"),
        txt_prompt=gripper_txt,
        txt_prompt_index=1,
        neg_txt_prompt=object_of_interest,
    )
    torch.cuda.empty_cache()

    logging.info(f"Segmentation took {timedelta(seconds=time.perf_counter() - start)}.")


def run_bundle_sdf_tracking_and_reconstruction(
    data_dir: str, output_dir: str, interpolate_missing_vertices: bool = False
) -> None:
    start = time.perf_counter()

    # Run tracking and reconstruction. We run it in a subprocess to be able to use
    # independent dependencies for BundleSDF.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bundle_sdf_dir = os.path.join(current_dir, "scalable_real2sim", "BundleSDF")
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    os.chdir(bundle_sdf_dir)
    torch.cuda.empty_cache()
    subprocess.run(
        "source .venv/bin/activate && "
        f"python run_custom.py --video_dir {data_dir} "
        f"--out_folder {output_dir} --use_gui 1 "
        f"--interpolate_missing_vertices {int(interpolate_missing_vertices)}",
        cwd=bundle_sdf_dir,
        shell=True,
        executable="/bin/bash",
    )
    os.chdir(current_dir)

    # Extract data.
    output_dir_parent = os.path.dirname(output_dir)
    shutil.move(
        os.path.join(output_dir, "ob_in_cam"),
        os.path.join(data_dir, "ob_in_cam"),
    )
    shutil.move(
        os.path.join(output_dir, "mesh_cleaned.obj"),
        os.path.join(output_dir_parent, "bundle_sdf_mesh.obj"),
    )
    textured_mesh_dir = os.path.join(output_dir_parent, "bundle_sdf_mesh")
    os.makedirs(textured_mesh_dir, exist_ok=True)
    shutil.move(
        os.path.join(output_dir, "textured_mesh.obj"),
        os.path.join(textured_mesh_dir, "textured_mesh.obj"),
    )
    shutil.move(
        os.path.join(output_dir, "material.mtl"),
        os.path.join(textured_mesh_dir, "material.mtl"),
    )
    shutil.move(
        os.path.join(output_dir, "material_0.png"),
        os.path.join(textured_mesh_dir, "material_0.png"),
    )

    logging.info(
        "BundleSDF tracking and reconstruction took "
        f"{timedelta(seconds=time.perf_counter() - start)}."
    )


def run_nerfstudio(data_dir: str, output_dir: str, use_depth: bool = False) -> None:
    # Preprocess the data for Nerfstudio.
    start = time.perf_counter()
    preprocess_data_for_nerfstudio(data_dir, output_dir)
    logging.info(
        f"Nerfstudio preprocessing took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Run Nerfstudio training.
    logging.info(
        "Started Nerfstudio training. Monitor progress at "
        "https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007."
    )
    start = time.perf_counter()
    torch.cuda.empty_cache()
    # NOTE: Use "nerfacto-big" or "nerfacto-huge" for better results but longer
    # training times.
    method = "depth-nerfacto" if use_depth else "nerfacto"
    train_process = subprocess.run(
        "source .venv_nerfstudio/bin/activate && "
        f"ns-train {method} --output-dir {output_dir} "
        # "--max-num-iterations 30000 "  # Adjust if needed
        "--pipeline.model.background_color 'random' "
        f"--pipeline.model.disable-scene-contraction True --data {output_dir} "
        # Slower but enables larger datasets
        f"{'--pipeline.datamanager.load-from-disk True ' if not use_depth else ''}"
        "--vis viewer_legacy --viewer.quit-on-train-completion True",
        cwd=os.path.dirname(os.path.abspath(__file__)),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    logging.info(
        f"Nerfstudio training took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Extract config file path from output. This is a bit complex because the file name
    # might be split across multiple lines.
    lines = train_process.stdout.split("\n")
    config_lines = []
    capture = False
    for line in lines:
        if "Saving config to:" in line:
            capture = True  # Start capturing from the next lines
            continue
        if capture:
            if "Saving checkpoints to:" in line:  # Stop capturing
                break
            config_lines.append(line.strip())
    # Join lines to reconstruct the path.
    config_path = "".join(config_lines).strip()
    if config_path and config_path.endswith("config.yml"):
        logging.info(f"Found config file: {config_path}")
    else:
        raise RuntimeError(
            "Could not find config file path in ns-train output. "
            f"Output: {train_process.stdout}"
        )

    # Run mesh extraction.
    start = time.perf_counter()
    # NOTE: It is recommended to increase `--resolution` to the highest value that fits
    # into your GPU memory.
    mesh_output_dir = os.path.join(output_dir, "nerfstudio_mesh")
    subprocess.run(
        "source .venv_nerfstudio/bin/activate && "
        f"ns-export tsdf --load-config {config_path} "
        f"--output-dir {mesh_output_dir} --target-num-faces 100000 "
        "--downscale-factor 2 --num-pixels-per-side 2048 --resolution 250 250 250 "
        "--use-bounding-box True --bounding-box-min -0.5 -0.5 -0.5 "
        "--bounding-box-max 0.5 0.5 0.5 --refine-mesh-using-initial-aabb-estimate True",
        cwd=os.path.dirname(os.path.abspath(__file__)),
        shell=True,
        executable="/bin/bash",
    )
    logging.info(
        f"Nerfstudio mesh extraction took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Move the mesh to the output directory parent directory.
    shutil.move(mesh_output_dir, os.path.dirname(output_dir))


def run_frosting(data_dir: str, output_dir: str, use_depth: bool = False) -> None:
    # Preprocess the data for Frosting.
    logging.info("Preprocessing data for Frosting...")
    start = time.perf_counter()
    process_moving_obj_data_for_sugar(
        data_dir, output_dir, num_images=1800, use_depth=use_depth
    )
    logging.info(
        f"Frosting preprocessing took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Run reconstruction. We run it in a subprocess to be able to use independent
    # dependencies for Frosting.
    start = time.perf_counter()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frosting_dir = os.path.join(current_dir, "scalable_real2sim", "Frosting")
    output_dir = os.path.abspath(output_dir)
    os.chdir(frosting_dir)
    torch.cuda.empty_cache()
    subprocess.run(
        "source .venv/bin/activate && "
        f"python train_full_pipeline.py -s {output_dir} -r 'dn_consistency' "
        "--high_poly True --export_obj True --white_background False "
        f"--masks {output_dir}/gripper_masks/ "
        + ("--depths depth/" if use_depth else ""),
        cwd=frosting_dir,
        shell=True,
        executable="/bin/bash",
    )
    os.chdir(current_dir)

    # Move the mesh to the output directory parent directory.
    mesh_path = os.path.join(
        frosting_dir, "output", "refined_frosting_base_mesh", "frosting"
    )
    mesh_out_dir = os.path.join(os.path.dirname(output_dir), "frosting_mesh")
    shutil.move(mesh_path, mesh_out_dir)

    logging.info(
        f"Frosting reconstruction took {timedelta(seconds=time.perf_counter() - start)}."
    )


def run_neuralangelo(data_dir: str, output_dir: str, use_depth: bool = False) -> None:
    # Process the data for Neuralangelo.
    logging.info("Processing data for Neuralangelo...")
    start = time.perf_counter()
    cfg_path = process_moving_obj_data_for_neuralangelo(
        data_dir, output_dir, num_images=1800
    )
    logging.info(
        f"Neuralangelo preprocessing took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Run Neuralangelo training.
    logging.info("Running Neuralangelo training...")
    start = time.perf_counter()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    neuralangelo_dir = os.path.join(current_dir, "scalable_real2sim", "neuralangelo")
    output_dir = os.path.abspath(output_dir)
    os.chdir(neuralangelo_dir)
    torch.cuda.empty_cache()
    subprocess.run(
        "source .venv/bin/activate && "
        f"torchrun --nproc_per_node=1 train.py --config {cfg_path} "
        f"--logdir {output_dir} --show_pbar "
        "--wandb --wandb_name nicholas_neuralangelo",
        cwd=neuralangelo_dir,
        shell=True,
        executable="/bin/bash",
    )
    logging.info(
        f"Neuralangelo training took {timedelta(seconds=time.perf_counter() - start)}."
    )

    # Obtain checkpoint path.
    checkpoint_version_txt = os.path.join(output_dir, "latest_checkpoint.txt")
    with open(checkpoint_version_txt, "r") as f:
        checkpoint_name = f.read().strip()
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    # Run Neuralangelo mesh extraction.
    logging.info("Running Neuralangelo mesh extraction...")
    start = time.perf_counter()
    output_mesh_dir = os.path.join(os.path.dirname(output_dir), "neuralangelo_mesh")
    # Lower resolution to reduce mesh size, lower block_res to reduce GPU memory usage.
    subprocess.run(
        "source .venv/bin/activate && "
        "torchrun --nproc_per_node=1 projects/neuralangelo/scripts/extract_mesh.py "
        f"--config {cfg_path} --checkpoint={checkpoint_path} --textured "
        f"--resolution=2048 --block_res=160 --output_file={output_mesh_dir}/mesh.obj",
        cwd=neuralangelo_dir,
        shell=True,
        executable="/bin/bash",
    )
    logging.info(
        f"Neuralangelo mesh extraction took {timedelta(seconds=time.perf_counter() - start)}."
    )
    os.chdir(current_dir)


def replace_trimesh_mesh_material(mesh_path: str) -> None:
    """Replaces the trimesh material file values with a nicer looking one."""
    material_path = os.path.join(os.path.dirname(mesh_path), "material.mtl")

    # Replace the material values with nicer looking ones.
    with open(material_path, "w") as mtl_file:
        mtl_file.write(
            "newmtl material_0\n"
            "Ns 50.000000\n"
            "Ka 1.000000 1.000000 1.000000\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.2 0.2 0.2\n"
            "Ke 0.000000 0.000000 0.000000\n"
            "Ni 1.500000\n"
            "d 1.000000\n"
            "illum 2\n"
            "map_Kd material_0.png\n"
        )


def main(
    data_dir: str,
    robot_id_dir: str,
    output_dir: str,
    skip_segmentation: bool = False,
    bundle_sdf_interpolate_missing_vertices: bool = False,
    use_depth: bool = False,
    use_finetuned_gripper_segmentation: bool = False,
):
    logging.info("Starting asset generation...")

    # Create output dir.
    os.makedirs(output_dir, exist_ok=True)

    # Get all subdirectories in the data directory.
    object_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for object_dir in tqdm(object_dirs):
        # Set up logging for this object.
        object_name = os.path.basename(object_dir)
        object_output_dir = os.path.join(output_dir, object_name)
        os.makedirs(object_output_dir, exist_ok=True)

        # Create a file handler for this object.
        log_file = os.path.join(object_output_dir, f"{object_name}_processing.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add the file handler to the root logger.
        logging.getLogger().addHandler(file_handler)

        logging.info(f"Processing object: {object_dir}")

        try:
            # Downsample images to not run out of memory.
            downsample_images(object_dir, num_images=1800)

            # Generate object and gripper masks.
            if not skip_segmentation:
                logging.info("Running segmentation...")
                run_segmentation(
                    data_dir=object_dir,
                    output_dir=object_dir,
                    use_finetuned_gripper_networks=use_finetuned_gripper_segmentation,
                )
            else:
                logging.info("Skipping segmentation...")
                if not os.path.exists(os.path.join(object_dir, "masks")):
                    raise FileNotFoundError(
                        "Object masks not found. Please run segmentation first."
                    )
                if not os.path.exists(os.path.join(object_dir, "gripper_masks")):
                    raise FileNotFoundError(
                        "Gripper masks not found. Please run segmentation first."
                    )

            # Object tracking + BundleSDF reconstruction.
            logging.info("Running BundleSDF tracking and reconstruction...")
            bundle_sdf_output_dir = os.path.join(object_output_dir, "bundle_sdf")
            os.makedirs(bundle_sdf_output_dir, exist_ok=True)
            run_bundle_sdf_tracking_and_reconstruction(
                data_dir=object_dir,
                output_dir=bundle_sdf_output_dir,
                interpolate_missing_vertices=bundle_sdf_interpolate_missing_vertices,
            )

            # Canonicalize the BundleSDF mesh.
            bundle_sdf_mesh_path = os.path.join(
                object_output_dir, "bundle_sdf_mesh/textured_mesh.obj"
            )
            canonicalize_mesh_from_file(
                mesh_path=bundle_sdf_mesh_path,
                output_path=bundle_sdf_mesh_path,
            )
            replace_trimesh_mesh_material(bundle_sdf_mesh_path)
            logging.info(f"Canonicalized BundleSDF mesh: {bundle_sdf_mesh_path}")

            # Physical property estimation in the BundleSDF frame.
            bundle_sdf_inertia_params_path = os.path.join(
                object_dir, "bundle_sdf_inertial_params.json"
            )
            identify_grasped_object_payload(
                robot_joint_data_path=robot_id_dir,
                object_joint_data_path=os.path.join(object_dir, "system_id_data"),
                object_mesh_path=bundle_sdf_mesh_path,
                json_output_path=bundle_sdf_inertia_params_path,
            )
            with open(bundle_sdf_inertia_params_path, "r") as f:
                bundle_sdf_inertia_params = json.load(f)

            # Output the BundleSDF SDFormat file.
            bundle_sdf_sdf_output_dir = os.path.join(
                object_output_dir, f"{object_name}_bundle_sdf.sdf"
            )
            create_sdf(
                model_name=object_name,
                mesh_parts_dir_name=f"{object_name}_bundle_sdf_parts",
                output_path=bundle_sdf_sdf_output_dir,
                visual_mesh_path=bundle_sdf_mesh_path,
                collision_mesh_path=bundle_sdf_mesh_path,
                mass=bundle_sdf_inertia_params["mass"],
                center_of_mass=np.array(bundle_sdf_inertia_params["center_of_mass"]),
                moment_of_inertia=np.array(bundle_sdf_inertia_params["inertia_matrix"]),
                use_hydroelastic=False,  # Enable for more accurate but Drake-specific SDFormat
                use_coacd=True,
            )

            # Nerfacto reconstruction + SDFormat output.
            nerfstudio_output_dir = os.path.join(object_output_dir, "nerfstudio")
            run_nerfstudio(
                data_dir=object_dir,
                output_dir=nerfstudio_output_dir,
                use_depth=use_depth,
            )

            # Frosting reconstruction + SDFormat output.
            frosting_output_dir = os.path.join(object_output_dir, "frosting")
            run_frosting(
                data_dir=object_dir,
                output_dir=frosting_output_dir,
                use_depth=use_depth,
            )

            # Neuralangelo reconstruction + SDFormat output.
            neuralangelo_output_dir = os.path.join(object_output_dir, "neuralangelo")
            run_neuralangelo(data_dir=object_dir, output_dir=neuralangelo_output_dir)

            # Canonicalize the Neuralangelo mesh.
            neuralangelo_mesh_path = os.path.join(
                object_output_dir, "neuralangelo_mesh/mesh.obj"
            )
            canonicalize_mesh_from_file(
                mesh_path=neuralangelo_mesh_path,
                output_path=neuralangelo_mesh_path,
            )
            logging.info(f"Canonicalized Neuralangelo mesh: {neuralangelo_mesh_path}")

            # Physical property estimation in the Neuralangelo frame.
            neuralangelo_inertia_params_path = os.path.join(
                object_dir, "neuralangelo_inertial_params.json"
            )
            identify_grasped_object_payload(
                robot_joint_data_path=robot_id_dir,
                object_joint_data_path=os.path.join(object_dir, "system_id_data"),
                object_mesh_path=neuralangelo_mesh_path,
                json_output_path=neuralangelo_inertia_params_path,
            )
            with open(neuralangelo_inertia_params_path, "r") as f:
                neuralangelo_inertia_params = json.load(f)

            # Output the Neuralangelo SDFormat file.
            neuralangelo_sdf_output_dir = os.path.join(
                object_output_dir, f"{object_name}_neuralangelo.sdf"
            )
            create_sdf(
                model_name=object_name,
                mesh_parts_dir_name=f"{object_name}_neuralangelo_parts",
                output_path=neuralangelo_sdf_output_dir,
                visual_mesh_path=neuralangelo_mesh_path,
                collision_mesh_path=neuralangelo_mesh_path,
                mass=neuralangelo_inertia_params["mass"],
                center_of_mass=np.array(neuralangelo_inertia_params["center_of_mass"]),
                moment_of_inertia=np.array(
                    neuralangelo_inertia_params["inertia_matrix"]
                ),
                use_hydroelastic=False,  # Enable for more accurate but Drake-specific SDFormat
                use_coacd=True,
            )

            logging.info(f"Finished processing object: {object_dir}")
            torch.cuda.empty_cache()

        finally:
            # Remove the file handler after processing this object.
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory where the collected data is saved. This should be "
        "the top-level directory that contains all the object subdirectories.",
    )
    parser.add_argument(
        "--robot-id-dir",
        type=str,
        required=True,
        help="Path to the directory where the robot ID data is saved. This should be "
        "the top-level directory that contains all different gripper opening "
        "subdirectories. NOTE that the robot parameters should have already been "
        "identified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where the generated assets will be saved. A new "
        "subdirectory will be created for each object with the same name as the "
        "subdirectory in the data directory.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="If specified, skip the segmentation step. This requires segmentation "
        "data to be already present in the data directory. It might be useful if you "
        "want to run segmentation using the `segment_moving_obj_data.py` script and "
        "manually specified positive/ negative annotations which is significantly "
        "more robust than automatic annotations from DINO.",
    )
    parser.add_argument(
        "--bundle-sdf-interpolate-missing-vertices",
        action="store_true",
        help="If specified, interpolate missing vertices in the BundleSDF texture map. "
        "This results in higher quality textures but is extremely slow.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="If specified, use depth images for geometric reconstruction when "
        "supported by the reconstruction method.",
    )
    parser.add_argument(
        "--use-finetuned-gripper-segmentation",
        action="store_true",
        help="If specified, use fine tuned SAM2 and GroundingDINO models for gripper"
        "segmentation.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(
        data_dir=args.data_dir,
        robot_id_dir=args.robot_id_dir,
        output_dir=args.output_dir,
        skip_segmentation=args.skip_segmentation,
        bundle_sdf_interpolate_missing_vertices=args.bundle_sdf_interpolate_missing_vertices,
        use_depth=args.use_depth,
        use_finetuned_gripper_segmentation=args.use_finetuned_gripper_segmentation,
    )
