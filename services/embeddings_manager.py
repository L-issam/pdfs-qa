import logging
from typing import List
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.hardware_manager import HardwareManager

class EmbeddingsManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware_manager = HardwareManager()
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings with the appropriate device."""
        try:
            device = self.hardware_manager.device
            self.logger.info(f"Initializing HuggingFace embeddings on {device}")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            self.logger.info("Falling back to CPU embeddings")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            if self.hardware_manager.device == 'cuda':
                self.logger.info("Falling back to CPU for embeddings")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                return self.embeddings.embed_documents(texts)