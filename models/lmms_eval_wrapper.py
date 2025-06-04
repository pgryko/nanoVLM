"""
LMMS-Eval wrapper for nanoVLM model.
This allows using lmms-eval for intermediate evaluation during training.
"""

import torch
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor


class NanoVLMWrapper(lmms):
    """Wrapper to make nanoVLM compatible with lmms-eval framework."""
    
    def __init__(
        self,
        model: VisionLanguageModel,
        tokenizer=None,
        image_processor=None,
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Set world_size and rank for non-distributed evaluation
        self._world_size = 1
        self._rank = 0
        
        # Add dummy accelerator for lmms-eval compatibility
        # This prevents errors when using "accelerate" backend
        class DummyAccelerator:
            def wait_for_everyone(self):
                pass
        
        self.accelerator = DummyAccelerator()
        
        # Get tokenizer and image processor from model config if not provided
        if tokenizer is None:
            self.tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens)
        else:
            self.tokenizer = tokenizer
            
        if image_processor is None:
            self.image_processor = get_image_processor(model.cfg.vit_img_size)
        else:
            self.image_processor = image_processor
            
    def _prepare_visual_input(self, visual_list: List[dict]) -> Optional[torch.Tensor]:
        """Convert visual inputs to model format."""
        if not visual_list or visual_list[0] is None:
            return None
            
        images = []
        for visual in visual_list:
            if isinstance(visual, dict) and "image" in visual:
                image = visual["image"]
                if isinstance(image, str):
                    # Load from path
                    image = Image.open(image).convert("RGB")
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                elif not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported image type: {type(image)}")
                
                # Process image
                processed = self.image_processor(image)
                images.append(processed)
        
        if images:
            return torch.stack(images).to(self.device)
        return None
        
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts."""
        results = []
        
        with torch.no_grad():
            for request in requests:
                context, continuation = request.args
                visual_list = request.visual if hasattr(request, 'visual') else None
                
                images_tensor = self._prepare_visual_input(visual_list)

                current_context_str_for_tokenizer = ""
                if images_tensor is not None:
                    # Prepend image tokens string placeholder
                    current_context_str_for_tokenizer += self.tokenizer.image_token * self.model.cfg.mp_image_token_length
                
                current_context_str_for_tokenizer += context # Append the textual context
                
                full_text_str_for_tokenizer = current_context_str_for_tokenizer + continuation

                # Tokenize the full text
                inputs = self.tokenizer(
                    full_text_str_for_tokenizer,
                    return_tensors="pt",
                    padding="longest", # Appropriate for single instance processing
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # Tokenize the context part (image_placeholders + text_context) to get its length
                context_inputs = self.tokenizer(current_context_str_for_tokenizer, return_tensors="pt") # Uses default add_special_tokens
                context_length = context_inputs["input_ids"].shape[1]
                
                # Prepare image tensor for the model
                images_for_model: Optional[torch.Tensor]
                if images_tensor is not None:
                    images_for_model = images_tensor.to(self.device)
                    if images_for_model.shape[0] != input_ids.shape[0]: # Ensure batch size consistency
                        images_for_model = images_for_model.expand(input_ids.shape[0], -1, -1, -1)
                else:
                    # If no images, pass None to the model
                    images_for_model = None
                
                # Forward pass
                outputs, _ = self.model(
                    input_ids,
                    images_for_model,
                    attention_mask=attention_mask,
                    targets=None  # No targets for loglikelihood computation
                )
                outputs = self.model.decoder.head(outputs) # Since no targets are provided, we need to apply the head to get the logits
                
                # Calculate log probabilities
                logits = outputs[:, :-1]  # Shift for next token prediction
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Get log likelihood of continuation tokens
                continuation_log_probs = []
                for i in range(context_length - 1, input_ids.shape[1] - 1):
                    token_id = input_ids[0, i + 1]
                    continuation_log_probs.append(log_probs[0, i, token_id].item())
                
                # Sum log probabilities for total log likelihood
                total_log_likelihood = sum(continuation_log_probs)
                
                # Check if continuation is greedy (would be generated by argmax)
                greedy_tokens = torch.argmax(logits[0, context_length - 1:], dim=-1)
                continuation_tokens = input_ids[0, context_length:]
                is_greedy = torch.all(greedy_tokens == continuation_tokens).item()
                
                results.append((total_log_likelihood, is_greedy))
                
        return results
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate text until a stopping condition is met."""
        results = []
        
        with torch.no_grad():
            for request in requests:
                context = request.args[0]
                gen_kwargs = request.args[1] if len(request.args) > 1 else {}
                visual_list = request.visual if hasattr(request, 'visual') else None

                images_tensor = self._prepare_visual_input(visual_list)

                current_prompt_str_for_tokenizer = ""
                if images_tensor is not None:
                    # Prepend image tokens string placeholder
                    current_prompt_str_for_tokenizer += self.tokenizer.image_token * self.model.cfg.mp_image_token_length
                
                current_prompt_str_for_tokenizer += context # Append the textual context

                # Prepare inputs
                inputs = self.tokenizer(
                    current_prompt_str_for_tokenizer,
                    return_tensors="pt",
                    padding="longest", # Appropriate for single instance processing
                    truncation=True,
                    max_length=self.max_length - gen_kwargs.get("max_new_tokens", 50) # Ensure prompt itself fits
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Prepare image tensor for the model
                images_for_model: Optional[torch.Tensor]
                if images_tensor is not None:
                    images_for_model = images_tensor.to(self.device)
                    if images_for_model.shape[0] != input_ids.shape[0]: # Ensure batch size consistency
                        images_for_model = images_for_model.expand(input_ids.shape[0], -1, -1, -1)
                else:
                    # If no images, pass None to the model
                    images_for_model = None
                
                # Extract generation parameters
                max_new_tokens = gen_kwargs.get("max_new_tokens", 50)
                temperature = gen_kwargs.get("temperature", 0.0)
                top_p = gen_kwargs.get("top_p", 1.0)
                greedy = temperature == 0.0

                # Generate
                generated_ids = self.model.generate(
                    input_ids,
                    images_for_model,
                    attention_mask,
                    max_new_tokens=max_new_tokens,
                    greedy=greedy,
                    temperature=temperature if not greedy else None,
                    top_p=top_p if not greedy else None,
                )
                
                # The model.generate returns only newly generated tokens, so we need to concatenate
                full_ids = torch.cat([input_ids, generated_ids], dim=1)
                
                # Decode only the generated part (exclude input)
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                results.append(generated_text)
                
        return results
    
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """Handle multi-round generation (for conversational tasks)."""
        # For now, we'll treat this the same as single-round generation
        # Can be extended later for true multi-turn conversation handling
        return self.generate_until(requests)
    
    @property
    def max_length(self):
        """Return the maximum sequence length."""
        return self.model.cfg.lm_max_position_embeddings
    
    @property
    def batch_size_per_gpu(self):
        """Return the batch size."""
        return self.batch_size