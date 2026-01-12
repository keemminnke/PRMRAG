"""Async OpenAI-compatible client for vLLM server."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp


@dataclass
class AsyncOpenAIClient:
    """Async client for vLLM OpenAI-compatible server.

    Designed for high-throughput batch processing with concurrent requests.
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 120.0
    max_concurrent: int = 8  # Max concurrent requests

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion for a single prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (default: self.temperature)
            max_tokens: Max tokens to generate (default: self.max_tokens)
            stop: Stop sequences

        Returns:
            Generated text
        """
        async with self._semaphore:
            session = await self._get_session()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if stop:
                payload["stop"] = stop

            async with session.post(
                f"{self.base_url}/completions",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM server error: {response.status} - {error_text}")

                result = await response.json()
                return result["choices"][0]["text"]

    async def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate completions for multiple prompts concurrently.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences

        Returns:
            List of generated texts (same order as input)
        """
        tasks = [
            self.generate(prompt, temperature, max_tokens, stop)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Chat completion (for chat-formatted models).

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences

        Returns:
            Assistant's response text
        """
        async with self._semaphore:
            session = await self._get_session()

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if stop:
                payload["stop"] = stop

            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM server error: {response.status} - {error_text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def chat_completion_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Chat completions for multiple conversations concurrently.

        Args:
            messages_list: List of message lists
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences

        Returns:
            List of assistant responses
        """
        tasks = [
            self.chat_completion(messages, temperature, max_tokens, stop)
            for messages in messages_list
        ]
        return await asyncio.gather(*tasks)


class SyncOpenAIClient:
    """Synchronous wrapper for AsyncOpenAIClient.

    Drop-in replacement for existing PolicyModel interface.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.async_client = AsyncOpenAIClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion (sync wrapper)."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.async_client.generate(prompt, temperature, max_tokens, stop)
        )

    def close(self):
        """Close the client."""
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self.async_client.close())
            self._loop.close()


def create_async_client(config: Dict[str, Any]) -> AsyncOpenAIClient:
    """Create AsyncOpenAIClient from config dict.

    Args:
        config: Dict with keys:
            - base_url: vLLM server URL (default: http://localhost:8000/v1)
            - model_name: Model name (default: Qwen/Qwen2.5-7B-Instruct)
            - temperature: Sampling temperature (default: 0.7)
            - max_tokens: Max tokens (default: 2048)
            - max_concurrent: Max concurrent requests (default: 8)

    Returns:
        AsyncOpenAIClient instance
    """
    return AsyncOpenAIClient(
        base_url=config.get("base_url", "http://localhost:8000/v1"),
        model=config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 2048),
        max_concurrent=config.get("max_concurrent", 8),
    )
