import os
from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content

# Providers that use the OpenAI-compatible chat completions API
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek",
    "qwen", "qwen-cn",
    "glm", "glm-cn",
    "minimax", "minimax-cn",
    "ollama", "openrouter",
)

_DEFAULT_BEDROCK_REGION = "ap-southeast-1"
_DEFAULT_BEDROCK_CONNECT_TIMEOUT = 10
_DEFAULT_BEDROCK_READ_TIMEOUT = 300

_BEDROCK_PASSTHROUGH_KWARGS = (
    "callbacks",
    "config",
    "credentials_profile_name",
    "disable_streaming",
    "guardrails",
    "metadata",
    "provider",
    "rate_limiter",
    "tags",
)


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock Converse API chat models."""

    def get_llm(self) -> Any:
        """Return configured ChatBedrockConverse instance."""
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse

        class NormalizedChatBedrockConverse(ChatBedrockConverse):
            """ChatBedrockConverse with normalized content output."""

            def invoke(self, input, config=None, **kwargs):
                return normalize_content(super().invoke(input, config, **kwargs))

        llm_kwargs = {
            "config": Config(
                connect_timeout=_get_int_env(
                    "TRADINGAGENTS_BEDROCK_CONNECT_TIMEOUT",
                    _DEFAULT_BEDROCK_CONNECT_TIMEOUT,
                ),
                read_timeout=_get_int_env(
                    "TRADINGAGENTS_BEDROCK_READ_TIMEOUT",
                    _DEFAULT_BEDROCK_READ_TIMEOUT,
                ),
            ),
            "model_id": self.model,
            "region_name": os.getenv("AWS_DEFAULT_REGION", _DEFAULT_BEDROCK_REGION),
        }

        if self.base_url:
            llm_kwargs["endpoint_url"] = self.base_url

        for key in _BEDROCK_PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        """Bedrock accepts model IDs and inference profile ARNs."""
        return True


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_client import OpenAIClient
        return OpenAIClient(model, base_url, provider=provider_lower, **kwargs)

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model, base_url, **kwargs)

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return AzureOpenAIClient(model, base_url, **kwargs)

    if provider_lower == "bedrock":
        return BedrockClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
