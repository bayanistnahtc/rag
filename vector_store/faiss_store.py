import logging

from langchain_community.vectorstores import FAISS

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    Implementation of vector storage based on FAISS.
    """

    def build_from_documents(self, documents):
        logger.info("Building the FAISS index...")
        # NOTE: We are using the default implementation. For larger data sets,
        # a more complex index type (e.g. with IVF) may be required.
        self._store = FAISS.from_documents(documents, self.embedding_function)

    def save(self, path: str):
        self._store.save_local(folder_path=path)
        logger.info(f"FAISS index is saved in {path}")

    @staticmethod
    def load(path: str, embedding_function):
        instance = FAISSVectorStore(embedding_function)
        # WARNING: allow_dangerous_deserialization=True may be unsafe.
        # TODO: Learn safer ways to serialize/deserialize.
        instance._store = FAISS.load_local(
            folder_path=path,
            embeddings=embedding_function,
            allow_dangerous_deserialization=True,
        )
        return instance
