import torch
from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor
from PIL import Image

# Load model
model = VisionLanguageModel.from_pretrained("./models/pgryko-nanovlm-COCO-VQAv2")
tokenizer = model.tokenizer
image_processor = get_image_processor(model.cfg.vit_img_size)

# Prepare input
img = Image.open("assets/image.png").convert("RGB")
img_t = image_processor(img).unsqueeze(0)

messages = [
    {
        "role": "user",
        "content": tokenizer.image_token * model.cfg.mp_image_token_length
        + "What is in this image?",
    }
]
encoded_prompt = tokenizer.apply_chat_template(
    [messages], tokenize=True, add_generation_prompt=True
)
tokens = torch.tensor(encoded_prompt)

print("Testing generation with debugging...")
print("Max new tokens: 10")

# Call generate but with debugging
generated = model.generate(tokens, img_t, max_new_tokens=10, greedy=True)
print(f"Generated IDs shape: {generated.shape}")
print(f"Generated IDs: {generated}")
print(f"Decoded: {repr(tokenizer.batch_decode(generated, skip_special_tokens=True))}")

# Let's also decode without skipping special tokens
print(
    f"Decoded (with special tokens): {repr(tokenizer.batch_decode(generated, skip_special_tokens=False))}"
)
