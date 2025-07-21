
# Hybrid Search with Qdrant

This repository provides a practical implementation of an advanced hybrid search system using the [Qdrant vector database](https://qdrant.tech/). It demonstrates how to combine traditional keyword-based search (sparse vectors) with modern semantic search (dense vectors) for highly relevant results. A reranking step further refines the output, creating a powerful and intelligent search experience.

## Main Features

- **Vector Database**: Utilizes Qdrant for high-performance vector storage and retrieval.
- **Hybrid Search**: Merges dense vectors (semantic meaning) and sparse vectors (keyword matching) into a single query.
- **Reranking**: Implements a second-stage reranker to boost accuracy and relevance.
- **Embedding Generation**: Uses the [`fastembed`](https://github.com/qdrant/fastembed) library for efficient text embedding creation.
- **Streamlit UI**: Interactive web interface for exploring search results.

## Local Deployment

### With Docker
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gbrlmoraes/hybrid_search_with_qdrant.git
   cd hybrid_search_with_qdrant
   ```

2. **Run Docker Compose**:
   - Note: You'll need Docker installed.
   ```bash
   docker compose up --build
   ```

### Optional (Updating the dataset)

1. **Prepare the dataset**:
   - Download the [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) and place it in `resources/amazon.csv`.

2. **Set up the Python environment:**
    - With [`uv`](https://docs.astral.sh/uv/) (recommended)
        ```bash
        uv sync --no-dev
        ```
    - With Python
        ```bash
        python -m venv .venv
        ```
    - Activate the environment
        - **Windows**
            ```bash
            .\.venv\Scripts\activate
            ```
        - **Linux/macOS**
            ```bash
            source .venv/bin/activate
            ```
    - Install dependencies
        ```bash
        pip install -r requirements.txt
        ```

3. Run the task to transform the original csv in a parquet ready for the pipeline:
    - With [`uv`](https://docs.astral.sh/uv/) (recomended)
        ```bash
        uv run task embed
        ```
    - With python
        ```bash
        python -m scripts.get_dataset
        ```

## Repository Structure

```
├── app.py                      # Streamlit web application for hybrid search
├── Dockerfile                  # Docker image for the app
├── docker-compose.yaml         # Multi-container orchestration (Qdrant + app)
├── entrypoint.sh               # Entrypoint script for Docker
├── pyproject.toml              # Python dependencies and project config
├── requirements.txt            # Python dependencies
├── uv.lock                     # Poetry lock file
├── LICENSE                     # License information
├── README.md                   # Project documentation
├── assets/
│   └── search_overview.png     # Visual overview of the search process
├── modules/
│   ├── __init__.py
│   ├── embedding_pipeline.py   # Embedding pipeline for document processing
│   └── embedding_retriever.py # Hybrid search and reranking logic
├── resources/
│   └── dataset.parquet         # Prepared dataset for search
├── scripts/
│   ├── get_dataset.py          # Script to prepare the dataset
│   └── generate_embeddings.py  # Script to generate and upload embeddings
└── .streamlit/
    └── config.toml             # Streamlit configuration
```

## Sources and Further Reading

Understanding hybrid search and Qdrant implementation
- [Hybrid Search Revamped - Building with Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/)
- [Beginner Tutorial: Setup Hybrid Search with FastEmbed](https://qdrant.tech/documentation/beginner-tutorials/hybrid-search-fastembed/)
- [Advanced Tutorial: Reranking in Hybrid Search](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)

Best practices
- [Bulk Upload Vectors to a Qdrant Collection](https://qdrant.tech/documentation/database-tutorials/bulk-upload/)

Good to know:
- [BM42: New Baseline for Hybrid Search](https://qdrant.tech/articles/bm42/)
- [What is ColBERT and Late Interaction and Why They Matter in Search?](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/)

Dataset source
- [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
