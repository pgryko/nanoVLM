"""
Enhanced training logging utilities for wandb
"""

import torch
import wandb
import numpy as np
from typing import Dict
from collections import defaultdict
import torch.nn.functional as F


def log_generation_samples(
    model,
    val_dataset,
    tokenizer,
    image_processor,
    device,
    global_step: int,
    num_samples: int = 3,
    max_new_tokens: int = 50,
) -> Dict:
    """
    Generate samples from validation data and log to wandb
    """
    model.eval()
    samples_data = []
    generation_lengths = []

    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            try:
                sample = val_dataset[i]

                # Prepare input
                if isinstance(sample["images"], list):
                    images = torch.stack(sample["images"]).unsqueeze(0).to(device)
                else:
                    images = sample["images"].unsqueeze(0).to(device)

                input_ids = sample["input_ids"].unsqueeze(0).to(device)

                # Generate
                generated_ids = model.generate(
                    input_ids, images, max_new_tokens=max_new_tokens, greedy=True
                )

                # Decode outputs
                generated_text = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                # Get target from labels (reconstruct answer from non -100 tokens)
                labels = sample["labels"]
                target_tokens = labels[labels != -100]
                if len(target_tokens) > 0:
                    target_text = tokenizer.decode(
                        target_tokens, skip_special_tokens=True
                    )
                else:
                    target_text = "No target found"

                # Extract question from input (remove image tokens and template)
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                # Simple extraction - look for user content
                if "user" in input_text and "assistant" in input_text:
                    user_start = input_text.find("user") + 4
                    assistant_start = input_text.find("assistant")
                    question = input_text[user_start:assistant_start].strip()
                else:
                    question = input_text

                gen_length = (
                    len(generated_ids[0])
                    if len(generated_ids.shape) > 1
                    else len(generated_ids)
                )
                generation_lengths.append(gen_length)

                samples_data.append([question, generated_text, target_text, gen_length])

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue

    # Log to wandb
    if samples_data:
        table = wandb.Table(
            columns=["Question", "Generated", "Target", "Length"], data=samples_data
        )

        metrics = {
            "generation/samples": table,
            "generation/avg_length": (
                np.mean(generation_lengths) if generation_lengths else 0
            ),
            "generation/min_length": (
                min(generation_lengths) if generation_lengths else 0
            ),
            "generation/max_length": (
                max(generation_lengths) if generation_lengths else 0
            ),
        }

        wandb.log(metrics, step=global_step)

        return {
            "avg_length": np.mean(generation_lengths) if generation_lengths else 0,
            "lengths": generation_lengths,
        }

    return {"avg_length": 0, "lengths": []}


def log_token_statistics(
    input_ids: torch.Tensor, labels: torch.Tensor, tokenizer, global_step: int
) -> None:
    """
    Log token-level statistics
    """
    with torch.no_grad():
        # Basic token stats
        total_tokens = input_ids.numel()
        unique_tokens = len(torch.unique(input_ids))

        # EOS token analysis
        eos_count = (input_ids == tokenizer.eos_token_id).sum().item()
        pad_count = (input_ids == tokenizer.pad_token_id).sum().item()

        # Label statistics (actual learning targets)
        valid_labels = labels[labels != -100]
        if len(valid_labels) > 0:
            answer_length = len(valid_labels)
            unique_answer_tokens = len(torch.unique(valid_labels))
            answer_eos_count = (valid_labels == tokenizer.eos_token_id).sum().item()

            # Check if answers are dominated by single tokens
            most_common_answer_token = torch.mode(valid_labels).values.item()
            most_common_count = (valid_labels == most_common_answer_token).sum().item()
            answer_diversity = (
                most_common_count / len(valid_labels) if len(valid_labels) > 0 else 0
            )
        else:
            answer_length = 0
            unique_answer_tokens = 0
            answer_eos_count = 0
            answer_diversity = 0
            most_common_answer_token = -1

        metrics = {
            "tokens/total_tokens": total_tokens,
            "tokens/unique_tokens": unique_tokens,
            "tokens/unique_token_ratio": unique_tokens / tokenizer.vocab_size,
            "tokens/eos_count": eos_count,
            "tokens/pad_ratio": pad_count / total_tokens,
            "tokens/answer_length": answer_length,
            "tokens/unique_answer_tokens": unique_answer_tokens,
            "tokens/answer_eos_count": answer_eos_count,
            "tokens/answer_diversity_ratio": answer_diversity,  # Higher = less diverse
        }

        wandb.log(metrics, step=global_step)


def log_loss_breakdown(
    logits: torch.Tensor, targets: torch.Tensor, total_loss: float, global_step: int
) -> None:
    """
    Log detailed loss breakdown to identify training issues
    """
    with torch.no_grad():
        # Calculate per-position losses
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )

        # Separate losses for different positions in valid sequences
        valid_mask = targets.reshape(-1) != -100

        if valid_mask.any():
            valid_losses = per_token_loss[valid_mask]

            # First token vs later tokens
            first_token_loss = valid_losses[0].item() if len(valid_losses) > 0 else 0
            later_tokens_loss = (
                valid_losses[1:].mean().item() if len(valid_losses) > 1 else 0
            )

            # Loss distribution stats
            loss_std = valid_losses.std().item()
            loss_max = valid_losses.max().item()
            loss_min = valid_losses.min().item()

            # Check if loss is concentrated on few tokens (indicating problems)
            high_loss_tokens = (
                (valid_losses > valid_losses.mean() + 2 * valid_losses.std())
                .sum()
                .item()
            )
            high_loss_ratio = high_loss_tokens / len(valid_losses)

        else:
            first_token_loss = 0
            later_tokens_loss = 0
            loss_std = 0
            loss_max = 0
            loss_min = 0
            high_loss_ratio = 0

        metrics = {
            "loss/total": total_loss,
            "loss/first_token": first_token_loss,
            "loss/later_tokens": later_tokens_loss,
            "loss/std": loss_std,
            "loss/max": loss_max,
            "loss/min": loss_min,
            "loss/high_loss_token_ratio": high_loss_ratio,
            "loss/first_vs_later_ratio": first_token_loss / (later_tokens_loss + 1e-8),
        }

        wandb.log(metrics, step=global_step)


def detect_model_collapse(
    model,
    val_dataset,
    tokenizer,
    image_processor,
    device,
    global_step: int,
    num_samples: int = 10,
    max_new_tokens: int = 20,
) -> Dict:
    """
    Detect if model has collapsed to repetitive patterns
    """
    model.eval()
    token_counts = defaultdict(int)
    sequence_patterns = defaultdict(int)
    total_tokens = 0
    generated_sequences = []

    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            try:
                sample = val_dataset[i]

                # Prepare input
                if isinstance(sample["images"], list):
                    images = torch.stack(sample["images"]).unsqueeze(0).to(device)
                else:
                    images = sample["images"].unsqueeze(0).to(device)

                input_ids = sample["input_ids"].unsqueeze(0).to(device)

                # Generate
                generated_ids = model.generate(
                    input_ids, images, max_new_tokens=max_new_tokens, greedy=True
                )

                # Analyze generated tokens
                if len(generated_ids.shape) > 1:
                    tokens = generated_ids[0].tolist()
                else:
                    tokens = generated_ids.tolist()

                generated_sequences.append(tokens)

                for token in tokens:
                    token_counts[token] += 1
                    total_tokens += 1

                # Check for repetitive patterns
                if len(tokens) > 2:
                    # Look for immediate repetitions
                    pattern = tuple(tokens)
                    sequence_patterns[pattern] += 1

            except Exception as e:
                print(f"Error in collapse detection sample {i}: {e}")
                continue

    if total_tokens == 0:
        return {"collapsed": True, "reason": "no_tokens_generated"}

    # Analysis
    most_common_token = max(token_counts.keys(), key=lambda k: token_counts[k])
    most_common_ratio = token_counts[most_common_token] / total_tokens
    unique_tokens = len(token_counts)

    # Check for collapse indicators
    collapsed = False
    collapse_reason = "none"

    # Collapse indicator 1: Single token dominance
    if most_common_ratio > 0.7:
        collapsed = True
        collapse_reason = "single_token_dominance"

    # Collapse indicator 2: Very low diversity
    elif unique_tokens < 10 and total_tokens > 50:
        collapsed = True
        collapse_reason = "low_diversity"

    # Collapse indicator 3: Identical sequences
    elif len(sequence_patterns) == 1 and num_samples > 1:
        collapsed = True
        collapse_reason = "identical_sequences"

    # Collapse indicator 4: Only EOS tokens after first
    eos_count = token_counts.get(tokenizer.eos_token_id, 0)
    if eos_count / total_tokens > 0.8 and total_tokens > 20:
        collapsed = True
        collapse_reason = "excessive_eos"

    most_common_token_text = (
        tokenizer.decode([most_common_token])
        if most_common_token in token_counts
        else "unknown"
    )

    metrics = {
        "collapse/is_collapsed": collapsed,
        "collapse/collapse_reason": collapse_reason,
        "collapse/most_common_token_id": most_common_token,
        "collapse/most_common_token_text": most_common_token_text,
        "collapse/most_common_ratio": most_common_ratio,
        "collapse/unique_tokens": unique_tokens,
        "collapse/total_tokens": total_tokens,
        "collapse/token_diversity": unique_tokens / tokenizer.vocab_size,
        "collapse/unique_sequences": len(sequence_patterns),
        "collapse/eos_ratio": eos_count / total_tokens if total_tokens > 0 else 0,
    }

    wandb.log(metrics, step=global_step)

    return {
        "collapsed": collapsed,
        "reason": collapse_reason,
        "most_common_token": most_common_token_text,
        "diversity": unique_tokens / tokenizer.vocab_size,
    }


def log_gradient_statistics(model: torch.nn.Module, global_step: int) -> None:
    """
    Log gradient statistics to monitor training health
    """
    grad_stats = {}
    total_norm = 0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()

            # Aggregate stats by module type
            module_type = name.split(".")[0] if "." in name else "other"

            grad_stats[f"gradients/{module_type}/{name.replace('.', '_')}/norm"] = (
                grad_norm
            )
            grad_stats[f"gradients/{module_type}/{name.replace('.', '_')}/mean"] = (
                grad_mean
            )
            grad_stats[f"gradients/{module_type}/{name.replace('.', '_')}/std"] = (
                grad_std
            )
            grad_stats[f"gradients/{module_type}/{name.replace('.', '_')}/max"] = (
                grad_max
            )

            # For global stats
            total_norm += grad_norm**2
            param_count += 1

    if param_count > 0:
        grad_stats["gradients/global/total_norm"] = total_norm**0.5
        grad_stats["gradients/global/avg_norm"] = (total_norm / param_count) ** 0.5

    wandb.log(grad_stats, step=global_step)


def log_answer_length_distribution(
    labels_batch: torch.Tensor, global_step: int, epoch: int = 0
) -> None:
    """
    Log answer length distribution to detect data quality issues
    """
    answer_lengths = []

    for labels in labels_batch:
        # Count non -100 tokens (actual answer tokens)
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens > 0:
            answer_lengths.append(valid_tokens)

    if answer_lengths:
        metrics = {
            f"data/answer_length_mean_epoch_{epoch}": np.mean(answer_lengths),
            f"data/answer_length_std_epoch_{epoch}": np.std(answer_lengths),
            f"data/answer_length_min_epoch_{epoch}": min(answer_lengths),
            f"data/answer_length_max_epoch_{epoch}": max(answer_lengths),
            f"data/single_token_answers_ratio_epoch_{epoch}": sum(
                1 for x in answer_lengths if x <= 1
            )
            / len(answer_lengths),
        }

        # Create histogram every few steps to avoid too much data
        if global_step % 1000 == 0:
            metrics[f"data/answer_length_histogram_epoch_{epoch}"] = wandb.Histogram(
                answer_lengths
            )

        wandb.log(metrics, step=global_step)
