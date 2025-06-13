import torch


class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template([{"role": "assistant", "content": random_string_5_letters}], tokenize=False, add_special_tokens=False)
        random_string_location = random_string_chat_templated.find(random_string_5_letters)

        self.prefix_len = len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _balanced_greedy_knapsack(self, lengths, L, delta=0):
        # keep the position while sorting
        items = sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)

        min_knapsacks   = (sum(lengths) + L - 1) // L + delta
        knapsack_load   = [0]   * min_knapsacks
        knapsack_groups = [[] for _ in range(min_knapsacks)]

        for idx, item_len in items:
            # choose the lightest knapsack so far
            ks_id = min(range(len(knapsack_load)),
                        key=knapsack_load.__getitem__)

            # open a new one if the chosen bag would overflow
            if knapsack_load[ks_id] + item_len > L:
                ks_id = len(knapsack_load)
                knapsack_load   .append(0)
                knapsack_groups.append([])

            knapsack_groups[ks_id].append(idx)
            knapsack_load  [ks_id] += item_len

        # remove the completely empty bags that the +delta heuristic created
        return [g for g in knapsack_groups if g]

    def _pack_one_group(self, group_indices, batch,
                        max_len):
        ids, lbl, am, ims = [], [], [], []

        for i in group_indices:
            ids.extend(batch["input_ids"][i])
            lbl.extend(batch["labels"][i])
            am .extend(batch["attention_mask"][i])
            ims.extend(batch["images"][i])

        # safety: assert we never overflow
        if len(ids) > max_len:
            raise ValueError(f"Packed length {len(ids)} > max_len {max_len}")

        return torch.stack(ids), torch.stack(lbl), torch.stack(am), ims

    def prepare_batch(self, batch, max_length=None, packing=False):
        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if packing:
            if max_length is None:
                raise ValueError("max_length must be provided when packing is True")
            
            # 1) run knapsack on the lengths
            lengths = list(map(len, batch["input_ids"]))
            groups  = self._balanced_greedy_knapsack(lengths, max_length)

            # 2) build new (packed) lists
            packed_ids, packed_lbl, packed_am, packed_imgs = [], [], [], []

            for g in groups:
                ids, lbl, am, ims = self._pack_one_group(
                    g, batch, max_length
                )
                packed_ids .append(ids)
                packed_lbl .append(lbl)
                packed_am  .append(am)
                packed_imgs.append(ims)
            #Flatten images again
            # packed_imgs = [img for sublist in packed_imgs for img in sublist]

            batch["input_ids"] = packed_ids
            batch["labels"]    = packed_lbl
            batch["attention_mask"]= packed_am
            batch["images"]    = packed_imgs
        
        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_len - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(l, (max_len - len(l), 0), value=0) for l in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(a, (max_len - len(a), 0), value=0) for a in batch["attention_mask"]]

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return [], [], [], []
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length, packing=True)
        return batch

class MMStarCollator(BaseCollator):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __call__(self, batch):
        batch = self.prepare_batch(batch)
        return batch

class Buffer:
    def __init__(self):
        self.examples: list[dict] = []

    def __len__(self):
        return len(self.examples)

    def take(self, n: int) -> list[dict]:
        out, self.examples = self.examples[:n], self.examples[n:]
        return out

    def extend(self, xs: list[dict]) -> None:
        self.examples.extend(xs)

class BufferedCollator:
    def __init__(self,
                 inner_collator,
                 target_batch_size: int,
                 max_buffer: int | None = None):
        self.inner = inner_collator
        self.N = target_batch_size
        self.max_buffer = (
            max_buffer if max_buffer is not None
            else 4 * target_batch_size
        )
        self.buffer = Buffer()

    def __call__(self, dataset_batch):
        batch = self.inner(dataset_batch)
        packed_size = len(batch["input_ids"])

        if packed_size > self.N:
            keep, overflow = (
                {k: v[:self.N] for k, v in batch.items()},
                {k: v[self.N:] for k, v in batch.items()},
            )
            self.buffer.extend([
                {"images": img, "input_ids": ids, "attention_mask": attn, "labels": lbl}
                for img, ids, attn, lbl in zip(overflow["images"], overflow["input_ids"], overflow["attention_mask"], overflow["labels"])
            ])
            batch = keep
            packed_size = self.N

        if packed_size < self.N and len(self.buffer) > 0:
            extra_samples = self.buffer.take(self.N - packed_size)
            extra_batch = {}
            keys = extra_samples[0].keys()
            for k in keys:
                values = [d[k] for d in extra_samples]
                if k == 'images':
                    extra_batch[k] = values
                else:
                    extra_batch[k] = torch.stack(values, dim=0)

            # Merge the extra batch with the current batch
            for k in batch:
                if k == 'images':
                    batch[k].extend(extra_batch[k])
                else:
                    batch[k] = torch.cat([batch[k], extra_batch[k]], dim=0)
                
            packed_size = len(batch["input_ids"])

        return batch
