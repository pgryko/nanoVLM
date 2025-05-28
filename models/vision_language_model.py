import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens)

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd) # [B, IMAGE_TOKEN_LENGTH, D_lm]

        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]

        updated_token_embd = token_embd.clone()
        
        image_token_id = self.tokenizer.image_token_id
        num_configured_image_tokens = self.cfg.IMAGE_TOKEN_LENGTH
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Create a mask for locations of the image placeholder token
        is_image_placeholder_mask = (input_ids == image_token_id)
        
        # Count the number of image placeholder tokens in each batch item
        num_image_tokens_per_row = is_image_placeholder_mask.sum(dim=1)
        
        # Identify rows that have at least the configured number of image tokens
        valid_rows_mask = (num_image_tokens_per_row >= num_configured_image_tokens)
        
        # Get the batch indices of these valid rows
        actual_batch_indices_to_update = valid_rows_mask.nonzero(as_tuple=True)[0]

        if actual_batch_indices_to_update.numel() > 0:
            # Select the relevant parts of input_ids and image_embd for these valid rows
            relevant_input_ids = input_ids[actual_batch_indices_to_update]
            relevant_image_embd = image_embd[actual_batch_indices_to_update]
            
            # For these valid rows, find the starting index of the first image placeholder token
            # argmax on the mask for rows known to contain the token.
            relevant_is_image_placeholder_mask = (relevant_input_ids == image_token_id)
            start_indices_for_update = relevant_is_image_placeholder_mask.int().argmax(dim=1)
            
            # Prepare row and column selectors for advanced indexing
            # Row selector for updated_token_embd: repeats the actual batch indices
            row_selector = actual_batch_indices_to_update.unsqueeze(1).expand(-1, num_configured_image_tokens)
            
            # Column selector for updated_token_embd: for each valid row, specifies the range of columns
            # [start_idx, start_idx + 1, ..., start_idx + num_configured_image_tokens - 1]
            col_offsets = torch.arange(num_configured_image_tokens, device=device).unsqueeze(0)
            col_selector = start_indices_for_update.unsqueeze(1) + col_offsets
            
            # Perform the vectorized replacement
            # updated_token_embd[rows, cols, :] = values
            # where rows and cols select the specific [Batch, Token_Position] locations
            updated_token_embd[row_selector, col_selector, :] = relevant_image_embd.to(updated_token_embd.dtype)


        # The combined_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        logits, _ = self.decoder(updated_token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):

        # 1. Process image
        image_embd = self.vision_encoder(image) # [B, T_img_feat, D_model]
        image_embd = self.MP(image_embd)      # [B, IMAGE_TOKEN_LENGTH, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]
        
        # Replace image token placeholders in prompt_token_embeds with image_embd
        image_token_id = self.tokenizer.image_token_id
        
        initial_combined_embeds = prompt_token_embeds.clone()

        for i in range(input_ids.size(0)):
            img_token_indices = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_token_indices) >= self.cfg.IMAGE_TOKEN_LENGTH:
                start_idx = img_token_indices[0]
                initial_combined_embeds[i, start_idx : start_idx + self.cfg.IMAGE_TOKEN_LENGTH, :] = image_embd[i]
            else:
                # Handle cases with insufficient placeholders, similar to forward method
                print(f"Warning: Not enough image token placeholders in generate prompt for sample {i}.")
                num_tokens_to_replace = min(len(img_token_indices), self.cfg.IMAGE_TOKEN_LENGTH, image_embd.size(1))
                if num_tokens_to_replace > 0 and len(img_token_indices) > 0:
                     start_idx = img_token_indices[0]
                     initial_combined_embeds[i, start_idx:start_idx + num_tokens_to_replace, :] = image_embd[i, :num_tokens_to_replace, :]


        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = image_embd.size(0) # Or initial_combined_embeds.size(0)
        
        # The attention_mask from input should correspond to input_ids and already be correct.
        # No explicit concatenation needed here if collator prepares it correctly.

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        return torch.cat(newly_generated_ids_list, dim=1)

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
