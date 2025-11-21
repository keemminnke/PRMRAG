"""Policy model wrapper for trajectory generation using vLLM (for GH200)."""

import random
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install it or use Docker image for GH200.")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class PolicyModelVLLM:
    """Wrapper for policy model using vLLM for faster inference on GH200.

    This class provides the same interface as PolicyModel (HF Transformers version)
    for drop-in replacement.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 42,  # Random seed for reproducibility
        device: str = "cuda",  # Ignored, for compatibility with PolicyModel
        torch_dtype=None,  # Ignored, vLLM handles this automatically
    ):
        """Initialize policy model with vLLM.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility (default: 42)
            device: Ignored (for compatibility with PolicyModel)
            torch_dtype: Ignored (vLLM handles dtype automatically)
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. "
                "Use the GH200 Docker image: ghcr.io/abacusai/gh200-llm/llm-train-serve:latest"
            )

        print(f"Loading policy model with vLLM: {model_name}")

        # Set random seed for reproducibility
        self.seed = seed
        set_seed(seed)
        print(f"  Random seed set to: {seed}")

        # Load tokenizer separately for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Initialize vLLM engine with enforce_eager=True to avoid compilation bugs
        # in development versions of vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,  # Required for vLLM dev versions with GH200
            disable_log_stats=True,  # Disable verbose logging
            seed=seed,  # Set seed for vLLM sampling
        )

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"✓ vLLM model loaded successfully (seed={seed})")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            stop_sequences: List of strings to stop generation

        Returns:
            Generated text (string, compatible with PolicyModel interface)
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        # Generate with vLLM (use_tqdm=False to suppress progress bar)
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        # Extract result - return string only (compatible with PolicyModel)
        output = outputs[0].outputs[0]
        generated_text = output.text.strip()

        return generated_text

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate text for multiple prompts efficiently with vLLM batching.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: List of strings to stop generation

        Returns:
            List of generated texts (compatible with PolicyModel interface)
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        # Batch generate with vLLM (automatically optimized, use_tqdm=False to suppress progress bar)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        # Extract results - return strings only (compatible with PolicyModel)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)

        return results

    def format_prompt_for_qwen(self, user_message: str) -> str:
        """Format prompt for Qwen2.5 chat template.

        Args:
            user_message: User's message/prompt

        Returns:
            Formatted prompt
        """
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                {"role": "user", "content": user_message}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual format
            return f"""<|im_start|>system
You are a helpful assistant that solves problems step by step.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    def generate_with_chat_template(
        self,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate using Qwen chat template.

        Args:
            user_message: User's prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: Stop sequences

        Returns:
            Generated text (string, compatible with PolicyModel interface)
        """
        prompt = self.format_prompt_for_qwen(user_message)
        return self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)

    def batch_generate_with_chat_template(
        self,
        user_messages: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """Batch generate using Qwen chat template (optimized for MC rollouts).

        Args:
            user_messages: List of user prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: Stop sequences

        Returns:
            List of generated texts
        """
        # Format all prompts with chat template
        prompts = [self.format_prompt_for_qwen(msg) for msg in user_messages]
        return self.batch_generate(prompts, max_tokens, temperature, top_p, stop_sequences)


# Alias for compatibility - use the same name as the HF version
PolicyModel = PolicyModelVLLM


def load_policy_model(config: dict) -> PolicyModelVLLM:
    """Load vLLM policy model from config.

    This function provides the same interface as the HF Transformers version
    for drop-in replacement.

    Args:
        config: Model configuration dict with keys:
            - model_name: HuggingFace model name
            - device: Device to use (ignored, for compatibility)
            - temperature: Sampling temperature (default: 0.8)
            - top_p: Nucleus sampling (default: 0.95)
            - max_tokens: Max tokens per generation (default: 200)
            - tensor_parallel_size: Number of GPUs (default: 1)
            - gpu_memory_utilization: GPU memory utilization (default: 0.7)
            - seed: Random seed for reproducibility (default: 42)

    Returns:
        PolicyModelVLLM instance
    """
    return PolicyModelVLLM(
        model_name=config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
        tensor_parallel_size=config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.7),
        max_new_tokens=config.get('max_tokens', 200),
        temperature=config.get('temperature', 0.8),
        top_p=config.get('top_p', 0.95),
        seed=config.get('seed', 42),
        device=config.get('device', 'cuda'),
    )
