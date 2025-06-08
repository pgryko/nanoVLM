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
        tokenizer.chat_template = tokenizer.chat_template.replace("{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}", "")  # hack for smollm's chat template which is too verbose for training
        self.image_token_str = tokenizer.image_token
        self.len_assistant_tokens = len(self.tokenizer.encode("<|im_start|>assistant\n"))
        self.len_newline_tokens = len(self.tokenizer.encode("\n"))
    
    def format_messages_for_loss(self, batched_messages):
        input_sequences = []
        loss_masks = []

        for messages in batched_messages:
            full_text = ""
            mask = []

            for message in messages:
                segment = self.tokenizer.apply_chat_template([message], tokenize=False)
                tokens = self.tokenizer(segment, add_special_tokens=False)["input_ids"]
                full_text += segment

                # Mark loss only on assistant tokens
                if message["role"] == "assistant":
                    mask.extend([0] * self.len_assistant_tokens + [1] * (len(tokens) - self.len_assistant_tokens - self.len_newline_tokens) + [0] * self.len_newline_tokens)
                else:
                    mask.extend([0] * len(tokens))

            input_sequences.append(full_text)
            loss_masks.append(mask)

        return input_sequences, loss_masks        

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
        input_sequences, loss_masks = self.format_messages_for_loss(messages_batched)

        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded_full_sequences["input_ids"]
        attention_mask = encoded_full_sequences["attention_mask"]
        
        # Pad and align loss masks accordingly
        loss_masks_padded = []
        for i, input_row in enumerate(encoded_full_sequences["input_ids"]):
            length = len(input_row)
            unpadded_mask = loss_masks[i][-self.max_length:]  # truncate from the left
            pad_len = length - len(unpadded_mask)
            padded = [0] * pad_len + unpadded_mask
            loss_masks_padded.append(padded)

        loss_masks_tensor = torch.tensor(loss_masks_padded).to(torch.bool)
        
        # Create labels where only answer tokens are predicted
        labels = input_ids.clone().masked_fill(~loss_masks_tensor, -100)
        labels[:, :-1] = labels[:, 1:] # Shift labels for causal LM
        labels[:, -1] = -100 # Last token has no target

        # Determine original lengths before padding/truncation to handle truncation cases
        original_lengths = [len(self.tokenizer.encode(seq)) for seq in input_sequences]

        for i in range(len(batch)):
            # Case 1: If sequence was truncated (original is longer than max_length)
            if original_lengths[i] > self.max_length:
                labels[i, :] = -100 # Ignore this sample entirely
                # print(f"Sample {i} truncated: original length {original_lengths[i]} exceeds max_length {self.max_length}. Ignoring sample.")
                continue

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
        input_sequences, loss_masks = self.format_messages_for_loss(messages_batched)

        encoded_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        # Pad and align loss masks accordingly
        loss_masks_padded = []
        for i, input_row in enumerate(encoded_sequences["input_ids"]):
            length = len(input_row)
            unpadded_mask = loss_masks[i]  # truncate from the left
            pad_len = length - len(unpadded_mask)
            padded = [0] * pad_len + unpadded_mask
            loss_masks_padded.append(padded)

        loss_masks_tensor = torch.tensor(loss_masks_padded).to(torch.bool)

        input_ids = encoded_sequences['input_ids'].masked_fill(loss_masks_tensor, self.tokenizer.pad_token_id)
        attention_mask = encoded_sequences['attention_mask'].masked_fill(loss_masks_tensor, 0)
        labels = encoded_sequences['input_ids'].clone().masked_fill(~loss_masks_tensor, self.tokenizer.pad_token_id)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }