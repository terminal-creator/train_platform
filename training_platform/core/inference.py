"""
Local Model Inference Engine

Provides inference capabilities using vLLM for:
- Local pre-trained models
- Training checkpoints
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
_model_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    model_path: str
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    trust_remote_code: bool = True


class VLLMInferenceEngine:
    """
    vLLM-based inference engine for local models and checkpoints.

    Uses vLLM's offline inference API for efficient batch processing.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _get_cache_key(self) -> str:
        """Generate cache key for the model."""
        return f"{self.config.model_path}"

    def initialize(self) -> bool:
        """
        Initialize the vLLM model.

        Returns True if successful, False otherwise.
        """
        if self._initialized:
            return True

        cache_key = self._get_cache_key()

        with _cache_lock:
            if cache_key in _model_cache:
                self.model = _model_cache[cache_key]
                self._initialized = True
                logger.info(f"Using cached model: {self.config.model_path}")
                return True

        try:
            from vllm import LLM, SamplingParams

            # Validate model path
            if not os.path.exists(self.config.model_path):
                logger.error(f"Model path does not exist: {self.config.model_path}")
                return False

            logger.info(f"Loading model from: {self.config.model_path}")

            # Initialize vLLM model
            self.model = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                # Disable tokenizer parallelism warning
                tokenizer_mode="auto",
            )

            # Cache the model
            with _cache_lock:
                _model_cache[cache_key] = self.model

            self._initialized = True
            logger.info(f"Model loaded successfully: {self.config.model_path}")
            return True

        except ImportError as e:
            logger.error(f"vLLM not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response for the given messages.

        Args:
            messages: List of chat messages [{"role": "user/assistant", "content": "..."}]
            max_tokens: Override default max_tokens
            temperature: Override default temperature

        Returns:
            Generated text response
        """
        if not self._initialized:
            if not self.initialize():
                return "[Model initialization failed]"

        try:
            from vllm import SamplingParams

            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            # Set sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
            )

            # Generate
            outputs = self.model.generate([prompt], sampling_params)

            if outputs and len(outputs) > 0:
                return outputs[0].outputs[0].text.strip()
            return ""

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation error: {str(e)}]"

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Generate responses for a batch of message lists.

        More efficient than calling generate() multiple times.
        """
        if not self._initialized:
            if not self.initialize():
                return ["[Model initialization failed]"] * len(batch_messages)

        try:
            from vllm import SamplingParams

            # Convert all messages to prompts
            prompts = [self._messages_to_prompt(msgs) for msgs in batch_messages]

            # Set sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
            )

            # Batch generate
            outputs = self.model.generate(prompts, sampling_params)

            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("")

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [f"[Generation error: {str(e)}]"] * len(batch_messages)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a prompt string.

        Uses ChatML format by default, which is compatible with most models.
        """
        # Try to use the model's chat template if available
        try:
            if hasattr(self.model, 'get_tokenizer'):
                tokenizer = self.model.get_tokenizer()
                if hasattr(tokenizer, 'apply_chat_template'):
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
        except Exception as e:
            logger.debug(f"Could not apply chat template: {e}")

        # Fallback: Use ChatML format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add generation prompt
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    logger.info("Model cache cleared")


def get_cached_models() -> List[str]:
    """Get list of currently cached model paths."""
    with _cache_lock:
        return list(_model_cache.keys())


# Convenience function for single inference
def infer_with_vllm(
    model_path: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    Convenience function for single inference.

    Args:
        model_path: Path to the model
        messages: Chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response
    """
    config = InferenceConfig(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    engine = VLLMInferenceEngine(config)
    return engine.generate(messages)


# Batch inference function
def batch_infer_with_vllm(
    model_path: str,
    batch_messages: List[List[Dict[str, str]]],
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> List[str]:
    """
    Batch inference function for efficiency.

    Args:
        model_path: Path to the model
        batch_messages: List of chat message lists
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of generated responses
    """
    config = InferenceConfig(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    engine = VLLMInferenceEngine(config)
    return engine.generate_batch(batch_messages)
