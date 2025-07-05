from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models


class EmbeddingRetriever:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        dense_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        sparse_embedding_model: str = 'Qdrant/bm25',
        late_interaction_embedding_model: str = 'colbert-ir/colbertv2.0',
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

        # Define the models that will be used
        self.dense_model_name = dense_embedding_model
        self.sparse_model_name = sparse_embedding_model
        self.late_interaction_model_name = late_interaction_embedding_model

        # Load the embedding models
        self.dense_embedding_model = TextEmbedding(self.dense_model_name)
        self.sparse_embedding_model = SparseTextEmbedding(self.sparse_model_name)
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(
            self.late_interaction_model_name
        )

    def sparse_query(self, query: str, labels: list[str], top_k: int = 10):
        # Create embeddings for the query
        sparse_vec = next(self.sparse_embedding_model.query_embed(query))

        # Filter the documents based on labels
        label_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key='metadata.label', match=models.MatchValue(value=lbl)
                )
                for lbl in labels
            ],
            min_should_match=1,
        )

        # Create a NamedSparseVector for the query
        named_sparse = models.NamedSparseVector(
            name=self.sparse_model_name,
            vector=models.SparseVector(**sparse_vec.as_object()),
        )

        # Sparse search
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=named_sparse,
            query_filter=label_filter,
            limit=top_k,
            with_payload=True,
        )

        return results

    def dense_query(
        self,
        query: str,
        labels: list[str],
        top_k: int = 10,
    ):
        # Create embeddings for the query
        dense_vectors = next(self.dense_embedding_model.query_embed(query))

        # Filter the documents based on labels
        label_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key='metadata.label', match=models.MatchAny(any=labels)
                )
            ]
        )

        # Dense search
        results = self.qdrant_client.query_points(
            self.collection_name,
            query=dense_vectors,
            using=self.dense_model_name,
            with_payload=True,
            limit=top_k,
            query_filter=label_filter,
        )

        return results

    def query(
        self,
        query: str,
        labels: list[str],
        top_k: int = 10,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
    ):
        # Create embeddings for the query
        dense_vectors = next(self.dense_embedding_model.query_embed(query))
        sparse_vectors = next(self.sparse_embedding_model.query_embed(query))
        late_vectors = next(self.late_interaction_embedding_model.query_embed(query))

        # Filter the documents based on labels
        label_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key='metadata.label', match=models.MatchAny(any=labels)
                )
            ]
        )

        # Hybrid search subquery
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using=self.dense_model_name,
                limit=top_k_dense,
                filter=label_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using=self.sparse_model_name,
                limit=top_k_sparse,
                filter=label_filter,
            ),
        ]

        # Rerank with the late interaction model
        results = self.qdrant_client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=late_vectors,
            using=self.late_interaction_model_name,
            with_payload=True,
            limit=top_k,
        )

        return results


if __name__ == '__main__':
    retriever = EmbeddingRetriever(
        qdrant_url='http://localhost:6333', collection_name='news_hybrid_search'
    )

    while True:
        query = input('Enter your query: ')

        if query.lower() == 'exit':
            print('Exiting...')
            break

        query_result = retriever.query(query, top_k=5, labels=['Business'])

        for point in query_result.points:
            print(f'- Score: {point.score}: Text: {point.payload["text"]}')
