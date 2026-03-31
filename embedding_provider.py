"""
Embedding Provider - Protocol-based interface for text embeddings.

Supports OpenAI API and local models (BGE-M3 etc.) via sentence-transformers.
Follows the same pattern as llm_client.py.
"""

import os
import logging
from typing import List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector."""
        ...

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider using langchain_openai."""

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        self._embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
        logger.info(f"Initialized OpenAI embedding: {model} (dim={self._dimension})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbeddingProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-m3", device: str = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._device = device or "cpu"
        self._model = SentenceTransformer(model, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Initialized local embedding: {model} (dim={self._dimension}, device={self._device})"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(config: dict) -> EmbeddingProvider:
    """
    Factory function to create embedding provider from config.

    Config expects a top-level 'embedding' section:
        embedding:
          provider: "openai" | "local"
          model: "text-embedding-3-small"   # or "BAAI/bge-m3" etc.

    Defaults to OpenAI text-embedding-3-small if no section provided.
    """
    emb_config = config.get("embedding", {})
    provider = emb_config.get("provider", "openai")
    model = emb_config.get("model")

    if provider == "local":
        model = model or "BAAI/bge-m3"
        device = emb_config.get("device", "cpu")
        # Hide CUDA before torch is imported to prevent GPU memory allocation
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return LocalEmbeddingProvider(model=model, device=device)

    # Default: OpenAI
    model = model or "text-embedding-3-small"
    api_key = os.environ.get("OPENAI_API_KEY")
    return OpenAIEmbeddingProvider(model=model, api_key=api_key)
