from abc import ABC, abstractmethod

class BaseVectorStore(ABC):

    @abstractmethod
    def insert_many(self, embedded_chunks):
        ...

    @abstractmethod
    def similarity_search(self, embedding, top_k):
        ...

    @abstractmethod
    def keyword_search(self, query, limit):
        ...

    @abstractmethod
    def delete_document(self, document_id):
        ...