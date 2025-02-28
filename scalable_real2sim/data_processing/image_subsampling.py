import concurrent.futures
import logging
import os
import shutil

import numpy as np
import torch

from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


def load_and_preprocess_images(image_paths, preprocess, device):
    features = []
    valid_image_paths = []

    def process_image(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            return input_tensor, img_path
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            return None, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_image, image_paths),
                total=len(image_paths),
                desc="Processing images",
            )
        )

    for feature, img_path in results:
        if feature is not None:
            features.append(feature)
            valid_image_paths.append(img_path)

    if not features:
        raise ValueError("No valid images were found.")
    return torch.cat(features, dim=0), valid_image_paths


def extract_features(model_name, model, images, batch_size: int):
    all_features = []

    # Split images into batches
    images_batched = torch.split(images, batch_size)

    with torch.no_grad():
        for image_batch in tqdm(images_batched, desc="Extracting features (batches)"):
            if model_name == "clip":
                batch_features = model.encode_image(image_batch)
            elif model_name == "dino":
                batch_features = model(image_batch)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # Normalize features
            batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)

            # Append batch features
            all_features.append(batch_features.cpu().numpy())

    # Concatenate all batch features
    return np.concatenate(all_features, axis=0)


def select_most_dissimilar_images(
    features: np.ndarray, K: int, N: int | None = None
) -> list[int]:
    """
    Selects the K most dissimilar images based on cosine distance between their features.

    Args:
        features (np.ndarray): An array of image features where each row corresponds to
            an image.
        K (int): The number of dissimilar images to select.
        N (int, optional): Maximum number of consecutive frames to skip between selected
            frames.

    Returns:
        list[int]: A sorted list of indices representing the selected dissimilar images.
    """
    distance_matrix = cdist(features, features, metric="cosine")
    N_total = len(features)

    # If N is provided, first select frames with uniform spacing.
    if N is not None:
        first_round_indices = list(range(0, N_total, N + 1))
        remaining_indices = set(range(N_total)) - set(first_round_indices)

        if len(first_round_indices) >= K:
            return first_round_indices[:K]

        selected_indices = first_round_indices
        num_remaining = K - len(first_round_indices)
        logging.info(
            f"Selected {len(selected_indices)} images with uniform spacing. Selecting "
            f"remaining {num_remaining} images based on dissimilarity..."
        )
    else:
        # Start with the most dissimilar image.
        selected_indices = [np.argmax(np.sum(distance_matrix, axis=1))]
        remaining_indices = set(range(N_total)) - set(selected_indices)
        num_remaining = K - 1

    # Select the remaining images based on dissimilarity
    for _ in tqdm(range(num_remaining), "Selecting most dissimilar images"):
        min_distances = np.min(
            distance_matrix[list(remaining_indices)][:, selected_indices], axis=1
        )
        next_index = list(remaining_indices)[np.argmax(min_distances)]

        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    return sorted(selected_indices)


def select_and_copy_dissimilar_images(
    image_dir: str,
    output_dir: str,
    K: int,
    N: int | None = None,
    model_name: str = "dino",
    device: str | None = None,
    batch_size: int = 256,
) -> None:
    """Selects the K most dissimilar images from a directory and copies them to an output directory.

    Args:
        image_dir (str): The directory containing the input images.
        output_dir (str): The directory where the selected images will be saved.
        K (int): The number of dissimilar images to select.
        N (int, optional): Maximum number of consecutive frames to skip between selected
            frames.
        model_name (str, optional): The name of the model to use for feature extraction
            ('clip' or 'dino').
        device (str, optional): The device to use for model inference ('cuda' or 'cpu').
            If not specified, it will automatically select based on availability.
        batch_size (int, optional): The batch size for processing images. Default is 256.
    """
    # Automatically select device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    else:
        device = torch.device(device)

    # Get all image file paths
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(image_extensions)
    ]

    if len(image_paths) < K:
        logging.warning(f"Number of images ({len(image_paths)}) is less than K ({K}).")
        return

    # Load the selected model
    if model_name == "clip":
        import clip

        model, preprocess = clip.load("ViT-B/32", device=device)
    elif model_name == "dino":
        # Install timm if not already installed
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is not installed. Please install it with `pip install timm`."
            )

        model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
        model.eval()
        model.to(device)

        # Define DINO preprocessing
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize(248),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),  # ImageNet means
                    std=(0.229, 0.224, 0.225),  # ImageNet stds
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        # Load and preprocess images
        images, valid_image_paths = load_and_preprocess_images(
            image_paths, preprocess, device
        )

        # Extract features
        features = extract_features(model_name, model, images, batch_size)

        # Select the K most dissimilar images
        selected_indices = select_most_dissimilar_images(features, K, N)
        selected_images = [valid_image_paths[idx] for idx in selected_indices]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Copy selected images to the output directory
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(
                        lambda img_path: shutil.copy(img_path, output_dir),
                        selected_images,
                    ),
                    total=len(selected_images),
                    desc="Copying selected images",
                )
            )

        logging.info(f"Selected {K} most dissimilar images using {model_name.upper()}.")
        logging.info(f"\nSelected images have been saved to '{output_dir}' directory.")
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            model.cpu()
            torch.cuda.empty_cache()
        del model, images, features
