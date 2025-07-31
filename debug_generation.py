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

# Manually step through generation
with torch.no_grad():
    # Get image embeddings
    image_embd = model.vision_encoder(img_t)
    image_embd = model.MP(image_embd)

    # Get initial embeddings
    prompt_token_embeds = model.decoder.token_embedding(tokens)
    initial_combined_embeds = model._replace_img_tokens_with_embd(
        tokens, prompt_token_embeds, image_embd
    )

    # Prefill
    prefill_output, kv_cache_list = model.decoder(
        initial_combined_embeds, attention_mask=None, kv_cache=None, start_pos=0
    )
    last_token_output = prefill_output[:, -1, :]

    # Get logits
    current_logits = model.decoder.head(last_token_output)

    # Get next token
    next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)

    print(f"Next token ID: {next_token_id.item()}")
    print(f"Decoded: {repr(tokenizer.decode([next_token_id.item()]))}")
    print(f"Top 10 logits: {torch.topk(current_logits[0], 10)}")

    # Check if this is being treated as EOS
    print(f"Tokenizer EOS ID: {tokenizer.eos_token_id}")
    print(f"Is next token EOS? {next_token_id.item() == tokenizer.eos_token_id}")
