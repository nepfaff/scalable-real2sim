import torch

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def detect_object(image_path: str) -> str:
    # Load LLaVA model and processor
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )

    # Load and process image
    raw_image = Image.open(image_path).convert("RGB")

    # Define conversation using the correct format
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is this object? Example: 'A red tomatoe soup can'.",
                },
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
        "cuda", torch.float16
    )

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=3,
    )
    response = processor.decode(outputs[0], skip_special_tokens=True)

    # Clean up the response to get only the object name
    response = response.split("ASSISTANT:")[-1].strip()
    if "is" in response.lower():  # Remove any "is" or "is a" or similar phrases
        response = response.split("is")[-1].strip()
    response = (
        response.strip('." ').lstrip("a ").lstrip("an ")
    )  # Remove articles and punctuation

    return response
