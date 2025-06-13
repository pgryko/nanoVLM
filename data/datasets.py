import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end   = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len
        
        return torch.tensor(conv_ids["input_ids"]), torch.tensor(mask).to(torch.bool), torch.tensor(conv_ids["attention_mask"])


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        super().__init__(dataset, tokenizer, image_processor, mp_image_token_length)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item['images']
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images = []
        for image in images_data:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image = self.image_processor(image)
                processed_images.append(processed_image)
            else:
                raise ValueError(f"Error processing image at index {idx}")

        # Process text (should be a list)
        text_data = item['texts']
        if not isinstance(text_data, list):
            text_data = [text_data]

        messages = []

        for text in text_data:
            messages.append({"role": "user", "content": text['user']})
            messages.append({"role": "assistant", "content": text['assistant']})

        messages[0]["content"] = self.tokenizer.image_token * len(processed_images) * self.mp_image_token_length + messages[0]["content"]      

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1) # Shift labels for causal LM
        labels[-1] = -100 # Last token has no target
        
        return labels

class MMStarDataset(BaseDataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
            
        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            raise ValueError(f"Error processing image at index {idx}")
        
        messages = []

        messages.append({"role": "user", "content": item['question'] +  "\nAnswer only with the letter!"})
        messages.append({"role": "assistant", "content": item['answer']})

        messages[0]["content"] = self.tokenizer.image_token * self.mp_image_token_length + messages[0]["content"]      
        
        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)
        input_ids = input_ids.masked_fill(mask, self.tokenizer.pad_token_id)
        attention_mask = attention_mask.masked_fill(mask, 0)

        return {
            "images": [processed_image],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, self.tokenizer.pad_token_id)
        return labels
