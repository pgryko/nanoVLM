import torch


# Our proposed greedy knapsack method
def balanced_greedy_knapsack(samples, L, delta=20):
    # Step 1: Sort the samples
    samples.sort(reverse=True)
    total_length = sum(samples)
    min_knapsacks = (total_length + L - 1) // L + delta
    # Step 2: Initialize knapsacks
    knapsacks=[[] for _ in range(min_knapsacks) ]
    knapsack_lengths = [0] * min_knapsacks
    # Step 3: Distribute samples across knapsacks
    ks_index = 0
    sample_index = 0
    while sample_index < len(samples): length = samples[sample_index]
    if knapsack_lengths[ks_index]+length<=L: 
        knapsacks[ks_index].append(length) 
        knapsack_lengths[ks_index] += length 
        sample_index += 1
    else:
        knapsacks.append([]) 
        knapsack_lengths.append(0)
        ks_index = min(range(len(knapsack_lengths)), key=knapsack_lengths.__getitem__)
    return knapsacks

class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)

        self.prefix_len = len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))
    
    def prepare_inputs_and_loss_mask(self, batched_messages, max_length=None):
        batch_token_ids: list[list[int]] = []
        batch_masks:     list[list[int]] = []
        batch_attentions: list[list[int]] = []

        for messages in batched_messages:
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

            batch_token_ids.append(conv_ids["input_ids"])
            batch_masks.append(mask)
            batch_attentions.append(conv_ids["attention_mask"])

        if max_length is not None:  # We need to keep the tokens to allow for the img embed replacing logic to work. Otherwise, we would need to track which images correspond to long samples.
            batch_token_ids = [ids[:max_length] for ids in batch_token_ids]
            batch_masks = [m if len(m) <= max_length else [0]*max_length for m in batch_masks] # Ignore samples that are longer than max_length
            batch_attentions = [a[:max_length] for a in batch_attentions]

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch_token_ids))
        batch_token_ids = [[self.tokenizer.pad_token_id]*(max_len-len(ids)) + ids for ids in batch_token_ids]
        batch_masks     = [[0]*(max_len-len(m)) + m         for m   in batch_masks]
        batch_attentions = [[0]*(max_len-len(a)) + a         for a   in batch_attentions]

        return torch.tensor(batch_token_ids), torch.tensor(batch_attentions), torch.tensor(batch_masks).to(torch.bool)

class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def __call__(self, batch):
        images = [item["images"] for item in batch]
        messages_batched = [item["text_data"] for item in batch]

        # Stack images
        imgs = [img for sublist in images for img in sublist]
        images = torch.stack(imgs)

        # Create inputs by concatenating special image tokens, question, and answer
        batch_input_ids, batch_attention_mask, loss_masks = self.prepare_inputs_and_loss_mask(messages_batched, max_length=self.max_length)

        # Create labels where only answer tokens are predicted
        labels = batch_input_ids.clone().masked_fill(~loss_masks, -100)
        labels[:, :-1] = labels[:, 1:] # Shift labels for causal LM
        labels[:, -1] = -100 # Last token has no target

        return {
            "image": images,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": labels,
        }

class MMStarCollator(BaseCollator):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
    
    def __call__(self, batch):
        images = [item["image"] for item in batch]
        messages_batched = [item["text_data"] for item in batch]

        # Stack images
        images = torch.stack(images)
        # Create inputs by concatenating special image tokens, question, and answer
        batch_input_ids, batch_attention_mask, loss_masks = self.prepare_inputs_and_loss_mask(messages_batched)

        input_ids = batch_input_ids.masked_fill(loss_masks, self.tokenizer.pad_token_id)
        attention_mask = batch_attention_mask.masked_fill(loss_masks, 0)
        labels = batch_input_ids.clone().masked_fill(~loss_masks, self.tokenizer.pad_token_id)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }