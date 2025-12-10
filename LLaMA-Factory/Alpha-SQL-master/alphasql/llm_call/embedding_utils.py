"""
Embedding utility module for handling embedding model calls.
Supports OpenAI API compatible embedding services.
"""

import os
from typing import List

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv(override=True)


class EmbeddingModel:
    """
    A wrapper class for embedding model calls using OpenAI API.
    Supports configurable embedding models through environment variables.
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize the embedding model.

        Args:
            model (str): The embedding model name. If None, reads from EMBEDDING_MODEL env var.
            api_key (str): The API key. If None, reads from EMBEDDING_API_KEY or OPENAI_API_KEY env var.
            base_url (str): The base URL. If None, reads from EMBEDDING_BASE_URL or OPENAI_BASE_URL env var.
        """

        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        # Use dedicated embedding API key if provided, otherwise fall back to OPENAI_API_KEY
        self.api_key = (
            api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        # Use dedicated embedding base URL if provided, otherwise fall back to OPENAI_BASE_URL
        self.base_url = (
            base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        )

        if not self.api_key:
            raise ValueError(
                "API key is required. Please set EMBEDDING_API_KEY or OPENAI_API_KEY in .env file."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(
            f"Initialized EmbeddingModel with model: {self.model}, base_url: {self.base_url}"
        )
        # {{END MODIFICATIONS}}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): The list of texts to embed.

        Returns:
            List[List[float]]: The list of embedding vectors.
        """
        # {{START MODIFICATIONS}}
        if not texts:
            return []

        try:
            embeddings = []
            batch_size = 20  # Use a small batch size to avoid API limits
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # Replace empty strings with a placeholder or space, as some APIs fail on empty strings
                sanitized_batch_texts = [t if t.strip() else " " for t in batch_texts]
                
                # OpenAI API allows batch embedding
                response = self.client.embeddings.create(model=self.model, input=sanitized_batch_texts)
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            logger.debug(f"Successfully embedded {len(texts)} documents")
            return embeddings

        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            raise
        # {{END MODIFICATIONS}}

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """

        try:
            response = self.client.embeddings.create(model=self.model, input=[text])

            embedding = response.data[0].embedding

            logger.debug("Successfully embedded query")
            return embedding

        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            raise


EMBEDDING_MODEL_CALLABLE = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the global embedding model instance.

    Returns:
        EmbeddingModel: The global embedding model instance.
    """

    global EMBEDDING_MODEL_CALLABLE

    if EMBEDDING_MODEL_CALLABLE is None:
        EMBEDDING_MODEL_CALLABLE = EmbeddingModel()

    return EMBEDDING_MODEL_CALLABLE
