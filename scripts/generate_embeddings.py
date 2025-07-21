import os

import pandas as pd

from modules.embedding_pipeline import Document, EmbeddingPipeline

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


# Read connection details from environment variables
qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
qdrant_port = os.getenv('QDRANT_PORT', '6333')

pipeline = EmbeddingPipeline(
    qdrant_url=f'http://{qdrant_host}:{qdrant_port}',
    collection_name='products_hybrid_search',
    dense_embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    sparse_embedding_model='Qdrant/bm25',
    late_interaction_embedding_model='colbert-ir/colbertv2.0',
)

pipeline.run(documents=documents, batch_size=8)
