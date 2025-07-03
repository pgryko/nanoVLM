import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path


class CustomDataset(Dataset):
    """
    Custom dataset for training NanoVLM on your own image-text pairs.

    Expects a JSON file with the following structure:
    [
        {
            "image_path": "path/to/image1.jpg",
            "conversations": [
                {"role": "user", "content": "What do you see in this image?"},
                {"role": "assistant", "content": "I see a cat sitting on a mat."}
            ]
        },
        {
            "image_path": "path/to/image2.jpg",
            "conversations": [
                {"role": "user", "content": "Describe this scene."},
                {"role": "assistant", "content": "This is a beautiful sunset over the ocean."}
            ]
        }
    ]

    Or for multi-turn conversations:
    {
        "image_path": "path/to/image.jpg",
        "conversations": [
            {"role": "user", "content": "What's in this image?"},
            {"role": "assistant", "content": "I see a dog playing in a park."},
            {"role": "user", "content": "What color is the dog?"},
            {"role": "assistant", "content": "The dog is golden brown."}
        ]
    }
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        image_processor,
        mp_image_token_length: int,
        image_root_dir: Optional[str] = None,
        max_length: int = 1024,
    ):
        """
        Args:
            json_path: Path to JSON file containing the dataset
            tokenizer: Tokenizer for text processing
            image_processor: Processor for image preprocessing
            mp_image_token_length: Length of image tokens after modality projection
            image_root_dir: Root directory for images (if paths in JSON are relative)
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_length = max_length
        self.image_root_dir = Path(image_root_dir) if image_root_dir else None

        # Load dataset
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Validate data
        self._validate_data()

        # Get prefix length for chat template
        self.prefix_len = self._get_prefix_len()

    def _validate_data(self):
        """Validate that the data has the expected format."""
        for idx, item in enumerate(self.data):
            if "image_path" not in item:
                raise ValueError(f"Item {idx} missing 'image_path'")
            if "conversations" not in item:
                raise ValueError(f"Item {idx} missing 'conversations'")
            if not isinstance(item["conversations"], list):
                raise ValueError(f"Item {idx} 'conversations' must be a list")
            if len(item["conversations"]) == 0:
                raise ValueError(f"Item {idx} has empty conversations")

            # Check conversation format
            for conv_idx, conv in enumerate(item["conversations"]):
                if "role" not in conv or "content" not in conv:
                    raise ValueError(
                        f"Item {idx}, conversation {conv_idx} missing 'role' or 'content'"
                    )
                if conv["role"] not in ["user", "assistant"]:
                    raise ValueError(
                        f"Item {idx}, conversation {conv_idx} has invalid role: {conv['role']}"
                    )

    def _get_prefix_len(self):
        """Get the prefix length added by the chat template."""
        random_string = "xyzab"
        templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string}],
            tokenize=False,
            add_special_tokens=False,
        )
        location = templated.find(random_string)
        return len(self.tokenizer.encode(templated[:location]))

    def __len__(self):
        return len(self.data)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from path."""
        if self.image_root_dir:
            full_path = self.image_root_dir / image_path
        else:
            full_path = Path(image_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _prepare_messages(self, conversations: List[Dict], num_images: int = 1):
        """Prepare messages with image tokens."""
        messages = conversations.copy()

        # Add image tokens to the first user message
        if num_images > 0 and len(messages) > 0 and messages[0]["role"] == "user":
            image_tokens = (
                self.tokenizer.image_token * num_images * self.mp_image_token_length
            )
            messages[0] = {
                "role": messages[0]["role"],
                "content": image_tokens + messages[0]["content"],
            }

        return messages

    def _prepare_inputs_and_labels(self, messages):
        """Prepare input_ids, attention_mask, and labels for training."""
        # Tokenize conversation
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = torch.tensor(conv_ids["input_ids"])
        attention_mask = torch.tensor(conv_ids["attention_mask"])

        # Create labels mask (1 for assistant responses, 0 for user inputs)
        labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Mark assistant responses for loss computation
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                labels_mask[start:end] = True

            cursor += seg_len

        # Create labels (shift by 1 for causal LM)
        labels = input_ids.clone()
        labels[~labels_mask] = -100  # Ignore user inputs in loss
        labels = labels.roll(-1)
        labels[-1] = -100

        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and process image
        image = self._load_image(item["image_path"])
        processed_image = self.image_processor(image)

        # Prepare messages with image tokens
        messages = self._prepare_messages(item["conversations"], num_images=1)

        # Prepare inputs and labels
        input_ids, attention_mask, labels = self._prepare_inputs_and_labels(messages)

        return {
            "images": [processed_image],  # List for compatibility with multi-image
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CustomMultiImageDataset(CustomDataset):
    """
    Extended custom dataset that supports multiple images per example.

    JSON format:
    [
        {
            "image_paths": ["path/to/image1.jpg", "path/to/image2.jpg"],
            "conversations": [
                {"role": "user", "content": "Compare these two images."},
                {"role": "assistant", "content": "The first image shows... while the second..."}
            ]
        }
    ]
    """

    def _validate_data(self):
        """Validate multi-image data format."""
        for idx, item in enumerate(self.data):
            # Support both single and multiple images
            if "image_path" in item:
                # Convert single image to list format
                item["image_paths"] = [item["image_path"]]
            elif "image_paths" not in item:
                raise ValueError(f"Item {idx} missing 'image_path' or 'image_paths'")

            if "conversations" not in item:
                raise ValueError(f"Item {idx} missing 'conversations'")
            if not isinstance(item["conversations"], list):
                raise ValueError(f"Item {idx} 'conversations' must be a list")

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and process all images
        image_paths = item.get("image_paths", [item.get("image_path")])
        processed_images = []

        for img_path in image_paths:
            image = self._load_image(img_path)
            processed_image = self.image_processor(image)
            processed_images.append(processed_image)

        # Prepare messages with image tokens
        messages = self._prepare_messages(
            item["conversations"], num_images=len(processed_images)
        )

        # Prepare inputs and labels
        input_ids, attention_mask, labels = self._prepare_inputs_and_labels(messages)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
