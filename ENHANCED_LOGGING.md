# Enhanced Wandb Logging for nanoVLM Training

This document describes the comprehensive logging enhancements added to detect training issues and model collapse early.

## Overview

The enhanced logging system adds critical metrics to help diagnose:
- Model collapse (repetitive/single token generation)
- Training instability 
- Data quality issues
- Gradient problems
- Generation quality degradation

## New Metrics Logged to Wandb

### 1. Generation Quality Metrics
- **Frequency**: Every 50 steps (configurable via `generation_eval_interval`)
- **Metrics**:
  - `generation/samples`: Table showing input questions, generated outputs, targets, and lengths
  - `generation/avg_length`: Average generation length
  - `generation/min_length`: Minimum generation length  
  - `generation/max_length`: Maximum generation length

### 2. Model Collapse Detection
- **Frequency**: Every 100 steps (configurable via `collapse_detection_interval`)
- **Metrics**:
  - `collapse/is_collapsed`: Boolean indicating if collapse detected
  - `collapse/collapse_reason`: Reason for collapse detection
  - `collapse/most_common_token_text`: Most frequently generated token
  - `collapse/most_common_ratio`: Ratio of most common token
  - `collapse/unique_tokens`: Number of unique tokens generated
  - `collapse/token_diversity`: Token diversity ratio
  - `collapse/eos_ratio`: Ratio of EOS tokens

**Collapse Detection Criteria**:
- Single token dominance (>70% of outputs)
- Very low diversity (<10 unique tokens in 50+ total)
- Identical sequences across samples
- Excessive EOS tokens (>80% of outputs)

### 3. Token-Level Statistics
- **Frequency**: Every 12-25 steps
- **Metrics**:
  - `tokens/total_tokens`: Total tokens in batch
  - `tokens/unique_tokens`: Unique tokens in batch
  - `tokens/unique_token_ratio`: Unique tokens / vocab size
  - `tokens/eos_count`: Number of EOS tokens
  - `tokens/pad_ratio`: Padding token ratio
  - `tokens/answer_length`: Average answer length
  - `tokens/answer_diversity_ratio`: Answer token repetition ratio

### 4. Loss Breakdown Analysis  
- **Frequency**: Every 25-100 steps
- **Metrics**:
  - `loss/total`: Total loss
  - `loss/first_token`: Loss on first generated token
  - `loss/later_tokens`: Average loss on subsequent tokens
  - `loss/std`: Loss standard deviation
  - `loss/max`: Maximum per-token loss
  - `loss/min`: Minimum per-token loss
  - `loss/high_loss_token_ratio`: Ratio of tokens with high loss
  - `loss/first_vs_later_ratio`: First token loss / later tokens loss

### 5. Gradient Statistics
- **Frequency**: Every 100 steps (configurable via `gradient_log_interval`)
- **Metrics**: Per-module gradient statistics
  - `gradients/{module}/{param}/norm`: Gradient norm
  - `gradients/{module}/{param}/mean`: Gradient mean
  - `gradients/{module}/{param}/std`: Gradient standard deviation
  - `gradients/{module}/{param}/max`: Maximum gradient value
  - `gradients/global/total_norm`: Total gradient norm
  - `gradients/global/avg_norm`: Average gradient norm

### 6. Answer Length Distribution
- **Frequency**: Every 50-200 steps  
- **Metrics**:
  - `data/answer_length_mean_epoch_{epoch}`: Mean answer length
  - `data/answer_length_std_epoch_{epoch}`: Answer length std dev
  - `data/answer_length_min_epoch_{epoch}`: Minimum answer length
  - `data/answer_length_max_epoch_{epoch}`: Maximum answer length
  - `data/single_token_answers_ratio_epoch_{epoch}`: Ratio of single-token answers
  - `data/answer_length_histogram_epoch_{epoch}`: Length distribution histogram

### 7. Learning Rate Tracking
- **Frequency**: Every 25-100 steps
- **Metrics**:
  - `lr/modality_projector`: Modality projector learning rate
  - `lr/vision_encoder`: Vision encoder learning rate  
  - `lr/language_model`: Language model learning rate

## Configuration Parameters

New configuration parameters added to `TrainConfig`:

```python
generation_eval_interval: int = gradient_accumulation_steps * 50  # Generation sampling interval
gradient_log_interval: int = gradient_accumulation_steps * 100    # Gradient stats interval  
collapse_detection_interval: int = gradient_accumulation_steps * 100  # Collapse detection interval
```

## Early Warning System

The system provides console warnings when issues are detected:

- **Model Collapse**: `⚠️ MODEL COLLAPSE DETECTED: {reason}`
- **Generation Quality**: Reports average generation length
- **Training Progress**: Shows key metrics at each evaluation

## Usage

The enhanced logging is automatically enabled when `log_wandb: True` in training config. No additional setup required.

### Example Wandb Dashboard Sections

1. **Generation Quality**: Monitor if model generates reasonable text lengths
2. **Collapse Detection**: Track model health and diversity metrics  
3. **Loss Analysis**: Understand which tokens are causing high loss
4. **Token Statistics**: Monitor vocabulary usage and answer patterns
5. **Gradient Health**: Detect vanishing/exploding gradients
6. **Data Quality**: Analyze answer length distributions

## Benefits

This enhanced logging system would have detected the model collapse issue in your training:

- **Generation length**: Would show average length dropping to ~1 token
- **Collapse detection**: Would trigger "single_token_dominance" warning  
- **Token statistics**: Would show 90%+ dominance of token "A"
- **Loss breakdown**: Would show unusual loss patterns

The system enables early intervention to save compute time and diagnose training issues quickly.