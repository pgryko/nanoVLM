import torch
from PIL import Image
from torch.utils.data import Dataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

    def __len__(self):
        return len(self.dataset)

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

        return {
            "images": processed_images,
            "text_data": messages,
        }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

    def __len__(self):
        return len(self.dataset)
    
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
        
        return {
            "image": processed_image,
            "text_data": messages,
        }
