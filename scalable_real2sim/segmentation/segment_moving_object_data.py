"""
Adapted from
https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py
"""

import glob
import logging
import os
import shutil

from itertools import chain

import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector, init_detector
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

"""
Hyperparameters for Grounding and Tracking
"""
PROMPT_TYPE_FOR_VIDEO = "point"  # Choose from ["point", "box", "mask"]
OFFLOAD_VIDEO_TO_CPU = True  # Prevents OOM for large videos but is slower.
OFFLOAD_STATE_TO_CPU = True
DINO_CONFIDENCE_THRESHOLD = 0.6


def convert_png_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Full path to the input image
            input_path = os.path.join(input_folder, filename)

            # Open the image using PIL
            with Image.open(input_path) as img:
                # Convert the image to RGB mode (JPG doesn't support transparency)
                img = img.convert("RGB")

                # Generate the output filename with .jpg extension
                output_filename = os.path.splitext(filename)[0] + ".jpg"

                # Full path to the output image
                output_path = os.path.join(output_folder, output_filename)

                # Save the image in JPG format
                img.save(output_path, "JPEG")


def sample_points_from_masks(masks, num_points):
    """
    Sample points from masks and return their absolute coordinates.

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # Find the valid mask points
        indices = np.argwhere(masks[i] == 1)
        # Convert from (y, x) to (x, y)
        indices = indices[:, ::-1]

        if len(indices) == 0:
            # If there are no valid points, append an empty array
            points.append(np.array([]))
            continue

        # Resampling if there's not enough points
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)

        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # Convert to np.array
    points = np.array(points, dtype=np.float32)
    return points


def segment_moving_obj_data(
    rgb_dir: str,
    output_dir: str,
    txt_prompt: str | None = None,
    txt_prompt_index: int = 0,
    neg_txt_prompt: str | None = None,
    num_neg_frames: int = 10,
    debug_dir: str | None = None,
    gui_frames: list[str] | None = None,
):
    # Ensure mutual exclusivity between GUI and text prompts
    if gui_frames is not None:
        if txt_prompt is not None or neg_txt_prompt is not None:
            raise ValueError("Cannot use both GUI frames and text prompts.")
    else:
        if txt_prompt is None:
            raise ValueError(
                "Text prompt must be provided if GUI frames are not specified."
            )

    if txt_prompt == "gripper":
        sam2_checkpoint = (
            "./checkpoints/checkpoint_gripper_finetune_sam2_200epoch_4_1.pt"
        )
    else:
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # Download checkpoint if not exist.
    if not os.path.exists(sam2_checkpoint):
        logging.info("Downloading sam2_hiera_large.pt checkpoint...")
        BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        sam2_hiera_l_url = f"{BASE_URL}sam2_hiera_large.pt"
        status = os.system(f"wget {sam2_hiera_l_url} -P ./checkpoints/")
        if status != 0:
            raise RuntimeError("Failed to download the checkpoint.")

    """
    Step 1: Environment settings and model initialization for SAM 2
    """
    # Use bfloat16 for the entire script
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize SAM image predictor and video predictor models
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # Build Grounding DINO from Hugging Face (used only if not using GUI)
    if gui_frames is None:
        if txt_prompt == "gripper":
            config_file = "./configs/grounding_dino_swin-t_finetune_16xb2_1x_coco.py"
            checkpoint_file = "./checkpoints/best_coco_bbox_mAP_epoch_8.pth"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = init_detector(config_file, checkpoint_file, device=device)
        else:
            model_id = "IDEA-Research/grounding-dino-tiny"
            model_id = "./checkpoints/best_coco_bbox_mAP_epoch_8.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(model_id)
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(device)

    # Convert PNG to JPG as required for video predictor.
    jpg_dir = os.path.join(rgb_dir, "jpg")

    if gui_frames is not None:
        # Find minimum frame number from gui_frames
        min_frame_num = min(int(frame) for frame in gui_frames)

        # Only convert PNGs with numbers >= min_frame_num
        if os.path.exists(jpg_dir):
            shutil.rmtree(jpg_dir)
        os.makedirs(jpg_dir)
        for filename in os.listdir(rgb_dir):
            if filename.endswith(".png"):
                frame_num = int(os.path.splitext(filename)[0])
                if frame_num >= min_frame_num:
                    input_path = os.path.join(rgb_dir, filename)
                    with Image.open(input_path) as img:
                        img = img.convert("RGB")
                        output_filename = os.path.splitext(filename)[0] + ".jpg"
                        output_path = os.path.join(jpg_dir, output_filename)
                        img.save(output_path, "JPEG")
    else:
        # If not using GUI, convert all PNGs
        convert_png_to_jpg(rgb_dir, jpg_dir)

    # Scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(jpg_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_count = len(frame_names)

    # Create a mapping from image names to frame indices
    frame_name_to_idx = {
        os.path.splitext(p)[0]: idx for idx, p in enumerate(frame_names)
    }

    # Initialize video predictor state
    inference_state = video_predictor.init_state(
        video_path=jpg_dir,
        offload_video_to_cpu=OFFLOAD_VIDEO_TO_CPU,
        offload_state_to_cpu=OFFLOAD_STATE_TO_CPU,
    )

    # Clear the debug directory before writing
    if debug_dir is not None:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        else:
            # Clear the debug directory
            for item in os.listdir(debug_dir):
                item_path = os.path.join(debug_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)  # Remove file or symlink
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)  # Remove directory
                except Exception as e:
                    logging.error(f"Error deleting {item_path}: {e}")

    """
    Step 2: Prompt Grounding DINO for box coordinates or collect points via GUI
    """

    object_id_counter = 1  # Start object IDs from 1
    txt_prompt_counter = 0
    neg_txt_prompt_counter = 0

    if gui_frames is not None:
        for frame_name in gui_frames:
            if frame_name not in frame_name_to_idx:
                raise ValueError(f"Frame {frame_name} not found in the RGB directory.")
            frame_idx = frame_name_to_idx[frame_name]

            # Read the frame
            img_path = os.path.join(jpg_dir, frame_names[frame_idx])
            image = cv2.imread(img_path)
            image_display = image.copy()

            positive_points = []
            negative_points = []

            def mouse_callback(event, x, y, flags, param):
                image_display = param["image"].copy()  # Get fresh copy of image
                scale = param["scale"]
                orig_x = int(x / scale)
                orig_y = int(y / scale)

                # Draw all existing points
                for px, py in positive_points:
                    px, py = int(px * scale), int(py * scale)
                    cv2.circle(
                        image_display,
                        (px, py),
                        radius=5,
                        color=(0, 255, 0),
                        thickness=-1,
                    )
                for px, py in negative_points:
                    px, py = int(px * scale), int(py * scale)
                    cv2.circle(
                        image_display,
                        (px, py),
                        radius=5,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

                if event == cv2.EVENT_LBUTTONDOWN:
                    positive_points.append([orig_x, orig_y])
                    cv2.circle(
                        image_display, (x, y), radius=5, color=(0, 255, 0), thickness=-1
                    )
                elif event == cv2.EVENT_RBUTTONDOWN:
                    negative_points.append([orig_x, orig_y])
                    cv2.circle(
                        image_display, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                    )

                # Add instructions text
                instructions = [
                    "Left click: Add positive point (green)",
                    "Right click: Add negative point (red)",
                    "u: Undo last point",
                    "r: Reset all points",
                    "q: Finish frame",
                ]
                y_offset = 30
                for inst in instructions:
                    cv2.putText(
                        image_display,
                        inst,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    y_offset += 25

                cv2.imshow("Frame", image_display)

            # Use a default screen height (80% of 1080p)
            target_height = int(1080 * 0.8)

            # Calculate scaling factor to fit image to target height
            scale = min(1.0, target_height / image.shape[0])

            # Resize image for display
            image_display = cv2.resize(image.copy(), None, fx=scale, fy=scale)

            cv2.namedWindow("Frame")
            cv2.setMouseCallback(
                "Frame", mouse_callback, {"image": image, "scale": scale}
            )
            cv2.imshow("Frame", image_display)

            logging.info(
                f"Annotating frame {frame_name}. Left click to add positive points, "
                "right click to add negative points. Press 'q' to finish."
            )

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u"):  # Undo last point
                    if len(positive_points) + len(negative_points) > 0:
                        if len(negative_points) > 0:
                            negative_points.pop()
                        else:
                            positive_points.pop()
                        mouse_callback(
                            None, 0, 0, None, {"image": image, "scale": scale}
                        )
                elif key == ord("r"):  # Reset points
                    positive_points.clear()
                    negative_points.clear()
                    mouse_callback(None, 0, 0, None, {"image": image, "scale": scale})

            cv2.destroyAllWindows()

            # Convert points to numpy arrays
            positive_points = np.array(positive_points, dtype=np.float32)
            negative_points = np.array(negative_points, dtype=np.float32)

            # Combine points and labels
            if len(positive_points) > 0 and len(negative_points) > 0:
                point_coords = np.concatenate(
                    [positive_points, negative_points], axis=0
                )
                point_labels = np.concatenate(
                    [np.ones(len(positive_points)), np.zeros(len(negative_points))]
                )
            elif len(positive_points) > 0:
                point_coords = positive_points
                point_labels = np.ones(len(positive_points))
            elif len(negative_points) > 0:
                point_coords = negative_points
                point_labels = np.zeros(len(negative_points))
            else:
                raise ValueError(f"No points provided for frame '{frame_name}'.")

            # Set image for image predictor
            image_predictor.set_image(image)

            # Predict mask using the points
            masks, scores, logits = image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=False,
            )
            # Convert the mask shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            # Process the detection results
            OBJECTS = [object_id_counter]  # Assign unique object ID

            # Save debug images with points
            if debug_dir is not None:
                img_path = os.path.join(jpg_dir, frame_names[frame_idx])
                image_debug = cv2.imread(img_path)
                # Draw the query points
                for point in positive_points:
                    x, y = point.astype(int)
                    cv2.circle(
                        image_debug, (x, y), radius=5, color=(0, 255, 0), thickness=-1
                    )
                for point in negative_points:
                    x, y = point.astype(int)
                    cv2.circle(
                        image_debug, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                    )
                # Save the image
                save_name = f"gui_prompt_{frame_name}.jpg"
                save_path = os.path.join(debug_dir, save_name)
                cv2.imwrite(save_path, image_debug)

            """
            Step 3: Register each object's positive points to video predictor with separate add_new_points call
            """

            if PROMPT_TYPE_FOR_VIDEO == "point":
                for obj_id in OBJECTS:
                    labels = point_labels.astype(np.int32)
                    points = point_coords
                    (
                        _,
                        out_obj_ids,
                        out_mask_logits,
                    ) = video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )
            else:
                raise NotImplementedError(
                    "For GUI input, only point prompts are supported."
                )

            object_id_counter += 1  # Increment object ID for next object

    else:
        """
        Step 2: Prompt Grounding DINO for box coordinates
        """

        # Function to get DINO boxes
        def get_dino_boxes(text, frame_idx):
            img_path = os.path.join(jpg_dir, frame_names[frame_idx])
            if txt_prompt == "gripper":
                # Use mmdetection api for gripper
                with autocast(enabled=False):
                    results = inference_detector(model, img_path, text_prompt=text)

                input_boxes = results.pred_instances[0].bboxes.cpu().numpy()
                confidences = results.pred_instances[0].scores.cpu().numpy().tolist()
                class_names = results.pred_instances[0].label_names
            else:
                image = Image.open(img_path)

                inputs = processor(images=image, text=text, return_tensors="pt").to(
                    device
                )
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]],
                )
                input_boxes = results[0]["boxes"].cpu().numpy()
                confidences = results[0]["scores"].cpu().numpy().tolist()
                class_names = results[0]["labels"]

            return input_boxes, confidences, class_names

        while True:
            input_boxes, confidences, class_names = get_dino_boxes(
                txt_prompt, txt_prompt_index
            )
            if confidences[0] > DINO_CONFIDENCE_THRESHOLD:
                break
            else:
                txt_prompt_index += 1

        assert (
            len(input_boxes) > 0
        ), "No results found for the text prompt. Make sure that the prompt ends with a dot '.'!"

        # Prompt SAM image predictor to get the mask for the object
        img_path = os.path.join(jpg_dir, frame_names[txt_prompt_index])
        image = Image.open(img_path)
        image_predictor.set_image(np.array(image.convert("RGB")))

        # Process the detection results
        OBJECTS = class_names

        # Prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # Convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor with separate add_new_points call
        """

        if PROMPT_TYPE_FOR_VIDEO == "point":
            # sample the positive points from mask for each object
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            # Save debug images with bounding boxes and query points for txt_prompt
            if debug_dir is not None:
                img_path = os.path.join(jpg_dir, frame_names[txt_prompt_index])
                image = cv2.imread(img_path)
                # Draw the boxes
                for box in input_boxes:
                    x_min, y_min, x_max, y_max = box.astype(int)
                    cv2.rectangle(
                        image,
                        (x_min, y_min),
                        (x_max, y_max),
                        color=(0, 255, 0),
                        thickness=2,
                    )
                # Draw the query points
                for points in all_sample_points:
                    for point in points:
                        x, y = point.astype(int)
                        cv2.circle(
                            image, (x, y), radius=5, color=(0, 255, 0), thickness=-1
                        )
                # Save the image
                save_name = f"txt_prompt_{txt_prompt_counter}.jpg"
                save_path = os.path.join(debug_dir, save_name)
                cv2.imwrite(save_path, image)
                txt_prompt_counter += 1

            for object_id, (label, points) in enumerate(
                zip(OBJECTS, all_sample_points), start=object_id_counter
            ):
                # label one means positive (do mask), label zero means negative (don't mask)
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=txt_prompt_index,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
            object_id_counter += len(OBJECTS)

        # Handle negative prompts if provided
        if neg_txt_prompt is not None:
            image_predictor.reset_predictor()

            neg_id_start_orig = neg_id_start = object_id_counter
            for idx in tqdm(
                np.linspace(0, frame_count - 1, num_neg_frames, dtype=int),
                desc="Adding negative",
                leave=False,
            ):
                neg_input_boxes, _, neg_class_names = get_dino_boxes(
                    neg_txt_prompt, idx
                )
                if len(neg_input_boxes) == 0:
                    continue

                img_path = os.path.join(jpg_dir, frame_names[idx])
                image = Image.open(img_path)
                image_predictor.set_image(np.array(image.convert("RGB")))

                # prompt SAM image predictor to get the mask for the negative object
                neg_masks, _, _ = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=neg_input_boxes,
                    multimask_output=False,
                )
                # convert the mask shape to (n, H, W)
                if neg_masks.ndim == 4:
                    neg_masks = neg_masks.squeeze(1)

                if PROMPT_TYPE_FOR_VIDEO == "point":
                    # sample the negative points from mask for each object
                    num_points = 1
                    neg_all_sample_points = sample_points_from_masks(
                        masks=neg_masks, num_points=num_points
                    )

                    # Save debug images with bounding boxes and query points for neg_txt_prompt
                    if debug_dir is not None:
                        img_path_cv = os.path.join(jpg_dir, frame_names[idx])
                        image_cv = cv2.imread(img_path_cv)
                        # Draw the boxes
                        for box in neg_input_boxes:
                            x_min, y_min, x_max, y_max = box.astype(int)
                            cv2.rectangle(
                                image_cv,
                                (x_min, y_min),
                                (x_max, y_max),
                                color=(0, 0, 255),
                                thickness=2,
                            )
                        # Draw the query points
                        for points in neg_all_sample_points:
                            for point in points:
                                x, y = point.astype(int)
                                cv2.circle(
                                    image_cv,
                                    (x, y),
                                    radius=5,
                                    color=(0, 0, 255),
                                    thickness=-1,
                                )
                        # Save the image
                        save_name = f"neg_txt_prompt_{neg_txt_prompt_counter}.jpg"
                        save_path = os.path.join(debug_dir, save_name)
                        cv2.imwrite(save_path, image_cv)
                        neg_txt_prompt_counter += 1

                    for object_id, (label, points) in enumerate(
                        zip(neg_class_names, neg_all_sample_points), start=neg_id_start
                    ):
                        # label zero means negative (don't mask)
                        labels = np.zeros((points.shape[0]), dtype=np.int32)
                        _, _, _ = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=txt_prompt_index,
                            obj_id=object_id,
                            points=points,
                            labels=labels,
                        )
                    neg_id_start += len(neg_class_names)
                # Handle 'box' and 'mask' prompts similarly
            object_id_counter = neg_id_start

    # Clear GPU memory.
    image_predictor.model.cpu()
    del image_predictor

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # Contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Save masks and overlay images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else:
    # png_files = glob.glob(os.path.join(output_dir, "*.png"))
    # jpg_files = glob.glob(os.path.join(output_dir, "*.jpg"))

    # Loop through the list of .png and .jpg files and delete them.
    # for file in chain(png_files, jpg_files):
    #     try:
    #         os.remove(file)
    #     except Exception as e:
    #         logging.error(f"Error deleting {file}: {e}")

    # Create 'mask_overlayed' subfolder inside debug_dir
    if debug_dir is not None:
        overlay_dir = os.path.join(debug_dir, "mask_overlayed")
        if not os.path.exists(overlay_dir):
            os.makedirs(overlay_dir)
        else:
            # Clear the overlay directory
            files = glob.glob(os.path.join(overlay_dir, "*"))
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    logging.error(f"Error deleting {f}: {e}")

    # save black masks for gripper not found frames
    for frame_idx in range(txt_prompt_index):
        image_name = frame_names[frame_idx]
        img_path = os.path.join(jpg_dir, image_name)
        image = cv2.imread(img_path)
        mask = np.zeros_like(image).astype(np.uint8)

        mask_name = os.path.splitext(image_name)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, mask_name), mask)

    for frame_idx, segments in video_segments.items():
        if gui_frames is None and neg_txt_prompt is not None:
            pos_segments = {k: v for k, v in segments.items() if k < neg_id_start_orig}
        else:
            pos_segments = segments

        if len(pos_segments) == 0:
            continue  # Skip if there are no positive segments

        masks = list(pos_segments.values())
        masks = np.concatenate(masks, axis=0)

        # Save masks
        union_mask = np.any(masks, axis=0)
        union_mask_8bit = (union_mask.astype(np.uint8)) * 255

        # Get the corresponding image name and change the extension to .png
        image_name = frame_names[frame_idx]
        mask_name = os.path.splitext(image_name)[0] + ".png"

        cv2.imwrite(os.path.join(output_dir, mask_name), union_mask_8bit)

        # If debug_dir is specified, save the overlaid image
        if debug_dir is not None:
            # Read the original image
            img_path = os.path.join(jpg_dir, image_name)
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(
                    f"Could not read image at {img_path}. Skipping overlay."
                )
                continue

            # Create a color mask (reddish color)
            color_mask = np.zeros_like(image)
            color_mask[:, :, 2] = 150  # Red channel intensity
            color_mask[:, :, 1] = 0  # Green channel
            color_mask[:, :, 0] = 0  # Blue channel

            # Apply the mask to the color mask
            mask = union_mask.astype(bool)
            alpha = 0.5  # Transparency factor

            if mask.any():
                try:
                    # Blend the original image and the color mask where mask is True
                    image[mask] = cv2.addWeighted(
                        image[mask], 1 - alpha, color_mask[mask], alpha, 0
                    )
                except Exception as e:
                    logging.error(f"Error blending mask on image '{image_name}': {e}")
                    continue
            else:
                logging.warning(f"No mask to overlay on image '{image_name}'.")
                continue

            # Save the overlaid image
            overlay_name = os.path.splitext(image_name)[0] + ".jpg"
            cv2.imwrite(os.path.join(overlay_dir, overlay_name), image)

    # Delete the jpg folder.
    try:
        shutil.rmtree(jpg_dir)
    except Exception as e:
        logging.error(f"Error deleting {jpg_dir}: {e}")

    logging.info("Done segmenting moving object data.")
