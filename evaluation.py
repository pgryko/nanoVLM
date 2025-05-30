"""
Integration of lmms-eval for intermediate evaluation during training.
"""

import os
import torch
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from lmms_eval import evaluator
    from lmms_eval.api.instance import Instance
    from lmms_eval.loggers import EvaluationTracker
    from lmms_eval.tasks import TaskManager
    LMMS_EVAL_AVAILABLE = True
except ImportError:
    LMMS_EVAL_AVAILABLE = False
    logger.warning("lmms-eval not installed. Install with: pip install lmms-eval")

from models.lmms_eval_wrapper import NanoVLMWrapper
from models.vision_language_model import VisionLanguageModel


def run_lmms_evaluation(
    model: VisionLanguageModel,
    tokenizer,
    image_processor,
    tasks: List[str],
    device: str = "cuda",
    batch_size: int = 8,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    log_samples: bool = False,
    verbosity: str = "INFO",
) -> Dict[str, Any]:
    """
    Run lmms-eval evaluation on specified tasks.
    
    Args:
        model: The nanoVLM model to evaluate
        tokenizer: The tokenizer for the model
        image_processor: The image processor for the model
        tasks: List of task names to evaluate (e.g., ["mmstar", "mme", "mathvista"])
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples (if applicable)
        limit: Limit number of examples per task (for debugging)
        output_path: Path to save evaluation results
        log_samples: Whether to log individual samples
        verbosity: Logging verbosity level
        
    Returns:
        Dictionary containing evaluation results
    """
    if not LMMS_EVAL_AVAILABLE:
        logger.error("lmms-eval is not installed. Cannot run evaluation.")
        return {}
    
    # Wrap the model for lmms-eval compatibility
    wrapped_model = NanoVLMWrapper(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        batch_size=batch_size
    )
    
    # Initialize task manager
    task_manager = TaskManager(verbosity=verbosity)
    
    # Create model args string
    model_args = f"device={device}"

    # Create a simple args object with the necessary attributes
    class SimpleArgs:
        def __init__(self):
            self.process_with_media = True  # Set to True for VLM tasks
    
    cli_args = SimpleArgs()
    
    # Run evaluation
    try:
        # Debug: check distributed_executor_backend
        logger.info(f"Using distributed_executor_backend: accelerate")
        
        results = evaluator.simple_evaluate(
            model=wrapped_model,
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            log_samples=log_samples,
            task_manager=task_manager,
            verbosity=verbosity,
            cli_args=cli_args,
            distributed_executor_backend="accelerate",
        )
        
        # Save results if output path is provided
        if output_path and results:
            import json
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during lmms evaluation: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return {}


def get_available_tasks() -> List[str]:
    """Get list of available evaluation tasks from lmms-eval."""
    if not LMMS_EVAL_AVAILABLE:
        return []
    
    try:
        from lmms_eval.tasks import TaskManager
        task_manager = TaskManager()
        return sorted(task_manager.all_tasks)
    except Exception as e:
        logger.error(f"Error getting available tasks: {e}")
        return []


def print_evaluation_results(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    if not results:
        return
        
    print("\n" + "="*50)
    print("LMMS-Eval Results")
    print("="*50)
    
    # Print overall results
    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    # Print group results if available
    if "groups" in results:
        print("\nGroup Results:")
        for group_name, group_results in results["groups"].items():
            print(f"\n{group_name}:")
            for metric, value in group_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
                    
    print("="*50 + "\n")


# Convenience function for common VLM benchmarks
def evaluate_vlm_benchmarks(
    model: VisionLanguageModel,
    tokenizer,
    image_processor,
    benchmarks: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on common VLM benchmarks.
    
    Args:
        model: The nanoVLM model to evaluate
        tokenizer: The tokenizer
        image_processor: The image processor
        benchmarks: List of benchmarks to run. If None, runs a default set.
        device: Device to run on
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        limit: Limit examples per task (for debugging)
        
    Returns:
        Dictionary with all benchmark results
    """
    # Default set of lightweight benchmarks suitable for intermediate evaluation
    if benchmarks is None:
        benchmarks = [
            "mmstar",      # Multi-modal reasoning
            "mme",         # Multi-modal understanding
            "ai2d",        # Multi-modal K12 science questions
            "chartqa",     # Chart understanding
            "docvqa",      # Document VQA
        ]
    
    all_results = {}
    
    for benchmark in benchmarks:
        logger.info(f"Running evaluation on {benchmark}...")
        
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{benchmark}_results.json")
        
        results = run_lmms_evaluation(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            tasks=[benchmark],
            device=device,
            batch_size=batch_size,
            limit=limit,
            output_path=output_path,
            log_samples=False,
            verbosity="DEBUG",
        )
        
        if results:
            all_results[benchmark] = results
            print_evaluation_results(results)
    
    return all_results