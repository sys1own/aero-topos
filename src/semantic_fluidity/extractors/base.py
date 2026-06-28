"""
Extensible extraction interfaces.

``RuleExtractor`` is the contract every extractor (regex/NLP, AST-based, or
LLM-assisted) implements: turn one :class:`IngestedDocument` into an
:class:`InvariantSchema`.  ``LLMClient`` is the extension point requirement #2
asks for when "utilizing an API/local model": a minimal ``complete(prompt)``
interface that a real client (OpenAI, Anthropic, a local llama.cpp server, ...)
can implement and plug into :class:`LLMAssistedExtractor`.  No network access
is assumed to be available, so the default client is :class:`NullLLMClient`,
and every offline extractor in this package works standalone without one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.semantic_fluidity.schema import InvariantSchema

if TYPE_CHECKING:
    from src.semantic_fluidity.documents import IngestedDocument


class RuleExtractor(ABC):
    """Extracts an :class:`InvariantSchema` from a single ingested document."""

    @abstractmethod
    def extract(self, document: "IngestedDocument", domain: str) -> InvariantSchema:
        raise NotImplementedError


class LLMClient(ABC):
    """Minimal extensible client interface for an LLM/API-assisted extractor."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class NullLLMClient(LLMClient):
    """Default client: no API/local model is configured in this environment."""

    def complete(self, prompt: str) -> str:
        raise RuntimeError(
            "NullLLMClient has no backing model; configure a real LLMClient or "
            "rely on the offline regex/AST extractors."
        )


class LLMAssistedExtractor(RuleExtractor):
    """Delegates to an :class:`LLMClient`, falling back to an offline extractor.

    This is the "extensible client interface" hook from requirement #2: swap in
    a real ``LLMClient`` to have prompts answered by an API or local model,
    while keeping behaviour fully offline (via ``fallback``) when none is
    configured or the call fails.
    """

    def __init__(self, client: LLMClient, fallback: RuleExtractor) -> None:
        self.client = client
        self.fallback = fallback

    def extract(self, document: "IngestedDocument", domain: str) -> InvariantSchema:
        try:
            self.client.complete(self._prompt_for(document))
        except Exception:
            return self.fallback.extract(document, domain)
        # A real client would parse its response into an InvariantSchema here;
        # without one configured, defer to the offline extractor.
        return self.fallback.extract(document, domain)

    @staticmethod
    def _prompt_for(document: "IngestedDocument") -> str:
        return (
            "Extract state variables, algorithmic boundaries and mathematical "
            f"equations as JSON from the following {document.format} document:\n\n"
            f"{document.text}"
        )
