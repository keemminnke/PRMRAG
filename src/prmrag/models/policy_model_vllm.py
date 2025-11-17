"""Policy model wrapper for trajectory generation using vLLM (for GH200)."""

from typing import List, Optional, Tuple
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install it or use Docker image for GH200.")


class PolicyModelVLLM:
    """Wrapper for policy model using vLLM for faster inference on GH200."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """Initialize policy model with vLLM.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. "
                "Use the GH200 Docker image: ghcr.io/abacusai/gh200-llm/llm-train-serve:latest"
            )

        print(f"Loading policy model with vLLM: {model_name}")

        # Load tokenizer separately for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"✓ vLLM model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            top_p: Nucleus sampling (overrides default)
            stop_sequences: List of strings to stop generation

        Returns:
            Tuple of (generated_text, finish_reason)
            finish_reason: "stop" (EOS), "length" (max_tokens), or "abort"
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        # Generate with vLLM
        outputs = self.llm.generate([prompt], sampling_params)

        # Extract result
        output = outputs[0].outputs[0]
        generated_text = output.text.strip()
        finish_reason = output.finish_reason  # "stop", "length", or "abort"

        return generated_text, finish_reason

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Generate text for multiple prompts efficiently with vLLM batching.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: List of strings to stop generation

        Returns:
            List of tuples (generated_text, finish_reason)
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        # Batch generate with vLLM (automatically optimized)
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            finish_reason = output.outputs[0].finish_reason
            results.append((generated_text, finish_reason))

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
    ) -> Tuple[str, str]:
        """Generate using Qwen chat template.

        Args:
            user_message: User's prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            stop_sequences: Stop sequences

        Returns:
            Tuple of (generated_text, finish_reason)
        """
        prompt = self.format_prompt_for_qwen(user_message)
        return self.generate(prompt, max_tokens, temperature, top_p, stop_sequences)


def load_policy_model_vllm(config: dict):
    """Load vLLM policy model from config.

    Args:
        config: Model configuration dict

    Returns:
        PolicyModelVLLM instance
    """
    return PolicyModelVLLM(
        model_name=config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct'),
        tensor_parallel_size=config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.8),
        max_new_tokens=config.get('max_tokens', 200),
        temperature=config.get('temperature', 0.8),
        top_p=config.get('top_p', 0.95),
    )
