from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from tqdm import tqdm


class Document(BaseModel):
    text: str = Field(..., description='Content of the document to be embedded')
    metadata: dict = Field(
        default_factory=dict,
        description='Metadata for the document, e.g., name, classification.',
    )

    class Config:
        json_schema_extra = {
            'example': {
                'text': 'This is an example document.',
                'metadata': {
                    'document_name': 'cat_news.md',
                    'classification': 'lifestyle',
                },
            }
        }


class EmbeddingPipeline:
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

        # Defines the models to be used
        self.dense_model_name = dense_embedding_model
        self.sparse_model_name = sparse_embedding_model
        self.late_interaction_model_name = late_interaction_embedding_model

    def _create_embeddings(self, documents: list[Document]):
        self.documents = documents
        total_documents = len(documents)

        # Process texts to create dense embeddings
        self.dense_embedding_model = TextEmbedding(self.dense_model_name)
        self.dense_embeddings = list(
            self.dense_embedding_model.embed(
                tqdm(
                    (doc.text for doc in self.documents),
                    desc='Processing embeddings with dense model',
                    total=total_documents,
                ),
            )
        )

        # Process texts to create sparse embeddings
        self.sparse_embedding_model = SparseTextEmbedding(self.sparse_model_name)
        self.sparse_embeddings = list(
            self.sparse_embedding_model.embed(
                tqdm(
                    (doc.text for doc in self.documents),
                    desc='Processing embeddings with sparse model',
                    total=total_documents,
                ),
            )
        )

        # Process texts to create late interaction (re-ranking) embeddings
        self.late_interaction_embedding_model = LateInteractionTextEmbedding(
            self.late_interaction_model_name
        )
        self.late_interaction_embeddings = list(
            self.late_interaction_embedding_model.embed(
                tqdm(
                    (doc.text for doc in self.documents),
                    desc='Processing embeddings with late interaction model',
                    total=total_documents,
                ),
            )
        )

    def _create_collection_if_not_exists(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                self.collection_name,
                vectors_config={
                    self.dense_model_name: models.VectorParams(
                        size=len(self.dense_embeddings[0]),
                        distance=models.Distance.COSINE,
                    ),
                    self.late_interaction_model_name: models.VectorParams(
                        size=len(self.late_interaction_embeddings[0][0]),
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),  # Not necessary for R-R
                    ),
                },
                sparse_vectors_config={
                    self.sparse_model_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                        index=models.SparseIndexParams(on_disk=False),
                    )
                },
            )

    def _upsert_values(self, batch_size: int = 4):
        # Creates a point that joins all vectors
        points = []
        for idx, (
            dense_embedding,
            sparse_embedding,
            late_interaction_embedding,
            document,
        ) in enumerate(
            zip(
                self.dense_embeddings,
                self.sparse_embeddings,
                self.late_interaction_embeddings,
                self.documents,
            )
        ):
            point = models.PointStruct(
                id=idx,
                # Vectors that will compose the point
                vector={
                    self.dense_model_name: dense_embedding,
                    self.sparse_model_name: sparse_embedding.as_object(),
                    self.late_interaction_model_name: late_interaction_embedding,
                },
                # Metadata
                payload=document.model_dump(),
            )
            points.append(point)

        # Performs the actual upsert
        for i in tqdm(range(0, len(points), batch_size), desc='Upserting in batches'):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=batch, wait=True
            )

    def run(self, documents: list[Document], batch_size: int = 4):
        self._create_embeddings(documents)
        self._create_collection_if_not_exists()
        self._upsert_values(batch_size=batch_size)


if __name__ == '__main__':
    import os

    import pandas as pd

    # Loads the dataset from the parquet file in resources
    # Note: You can download the original dataset from https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
    # and save it as 'resources/amazon.csv', them run the get_dataset.py script
    df = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'resources',
            'dataset.parquet',
        )
    )

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                text=row['text'],
                metadata={
                    'label': row['category'],
                },
            ),
        )

    pipeline = EmbeddingPipeline(
        qdrant_url='http://localhost:6333',
        collection_name='products_hybrid_search',
        dense_embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        sparse_embedding_model='Qdrant/bm25',
        late_interaction_embedding_model='colbert-ir/colbertv2.0',
    )

    pipeline.run(documents=documents, batch_size=8)
