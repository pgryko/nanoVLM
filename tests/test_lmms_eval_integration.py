"""Test lmms-eval integration with nanoVLM."""

import unittest
import torch
import numpy as np

try:
    from lmms_eval.api.model import lmms
    from lmms_eval.api.instance import Instance
    LMMS_EVAL_AVAILABLE = True
except ImportError:
    LMMS_EVAL_AVAILABLE = False
    print("lmms-eval not installed. Skipping integration tests.")

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from eval.lmms_eval_wrapper import NanoVLMWrapper
from data.processors import get_tokenizer, get_image_processor


class TestLMMSEvalIntegration(unittest.TestCase):
    
    @unittest.skipIf(not LMMS_EVAL_AVAILABLE, "lmms-eval not installed")
    def setUp(self):
        """Set up a minimal model for testing."""
        # Minimal config for testing using VLMConfig
        self.cfg = VLMConfig()
        
        # Create model without loading pretrained weights
        self.model = VisionLanguageModel(self.cfg)
        self.model.eval()
        
        # Get tokenizer and image processor
        self.tokenizer = get_tokenizer(self.cfg.lm_tokenizer)
        self.image_processor = get_image_processor(self.cfg.vit_img_size)
        
        # Create wrapper
        self.wrapper = NanoVLMWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            device="cpu",
            batch_size=1
        )
    
    @unittest.skipIf(not LMMS_EVAL_AVAILABLE, "lmms-eval not installed")
    def test_wrapper_inherits_lmms(self):
        """Test that wrapper properly inherits from lmms base class."""
        self.assertIsInstance(self.wrapper, lmms)
    
    @unittest.skipIf(not LMMS_EVAL_AVAILABLE, "lmms-eval not installed")
    def test_generate_until(self):
        """Test generate_until method."""
        # Create a mock request
        # The arguments tuple must now match the expected structure for the wrapper's Collator
        # (context_str, gen_kwargs, doc_to_visual_fn, doc_id, task_name, split_name)
        request = Instance(
            request_type="generate_until",
            doc={}, # doc can be empty for this test
            arguments=(
                "Hello",  # context_str
                {"max_new_tokens": 10, "temperature": 0.0},  # gen_kwargs
                lambda x: [], # doc_to_visual_fn (dummy function returning empty list of visuals)
                0,  # doc_id
                "test_task",  # task_name
                "test_split"  # split_name
            ),
            idx=0,
            metadata={'task': 'test_task', 'doc_id': 0, 'repeats': 1}
            # For the wrapper, task_name and doc_id are now part of 'arguments'.
        )
        # request.visual is not used by generate_until directly; doc_to_visual handles visuals.
        
        # Mock task_dict for the wrapper.
        # This is necessary because the wrapper's generate_until accesses self.task_dict[task][split][ids]
        # even if the doc_to_visual function (like our dummy lambda) doesn't use the result.
        self.wrapper.task_dict = {
            "test_task": {
                "test_split": {
                    0: {}  # Mocking an entry for doc_id 0. The content {} is arbitrary for the dummy lambda.
                }
            }
        }
        
        # Generate
        results = self.wrapper.generate_until([request])
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)
    
    @unittest.skipIf(not LMMS_EVAL_AVAILABLE, "lmms-eval not installed")
    def test_loglikelihood(self):
        """Test loglikelihood computation."""
        # Create a mock request
        request = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("The cat is", " sitting"),
            idx=0,
            metadata={'task': 'test_task', 'doc_id': 0, 'repeats': 1}
        )
        request.visual = None
        
        # Assert that NotImplementedError is raised
        with self.assertRaises(NotImplementedError):
            self.wrapper.loglikelihood([request])
    
    @unittest.skipIf(not LMMS_EVAL_AVAILABLE, "lmms-eval not installed")
    def test_visual_input_preparation(self):
        """Test visual input preparation."""
        from PIL import Image
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Test with PIL image
        visual_list = [pil_image] # Pass the image directly
        processed = self.wrapper._prepare_visual_input(visual_list)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape[0], 1)  # Batch size 1
        
        # Test with numpy array
        visual_list = [dummy_image] # Pass the numpy array directly
        processed = self.wrapper._prepare_visual_input(visual_list)
        self.assertIsNotNone(processed)
        
        # Test with None (visual_list itself is None, not a list containing None)
        processed = self.wrapper._prepare_visual_input(None)
        self.assertIsNone(processed)
        
        # Test with empty list
        processed = self.wrapper._prepare_visual_input([])
        self.assertIsNone(processed)


if __name__ == '__main__':
    unittest.main()