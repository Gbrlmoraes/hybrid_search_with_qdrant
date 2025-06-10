# Hybrid Search with Qdrant

This repository provides a practical implementation of an advanced hybrid search system using the Qdrant vector database.

The project demonstrates how to effectively combine traditional keyword-based search (sparse vectors) with modern semantic search (dense vectors) to achieve highly relevant results. It also incorporates a reranking step to further refine the search output for maximum precision, creating a powerful and intelligent search experience.

## Core Concepts

* **Vector Database**: Utilizes **Qdrant** for high-performance vector storage and retrieval.
* **Hybrid Search**: Merges dense vectors (for semantic meaning) and sparse vectors (for keyword matching) into a single, powerful query.
* **Reranking**: Implements a second-stage reranker that re-orders the retrieved results to significantly boost accuracy and relevance.
* **Embedding Generation**: Leverages the `fastembed` library for efficient and high-quality text embedding creation.

## Sources and Further Reading

The implementation in this repository is based on concepts and tutorials from the official Qdrant documentation. For a deeper understanding, please refer to the original articles:

* **[Hybrid Search Revamped - Building with Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/)**: An overview of the core concepts behind Qdrant's modern hybrid search capabilities.
* **[Beginner Tutorial: Setup Hybrid Search with FastEmbed](https://qdrant.tech/documentation/beginner-tutorials/hybrid-search-fastembed/)**: A practical guide to setting up a basic hybrid search system.
* **[Advanced Tutorial: Reranking in Hybrid Search](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)**: A tutorial on implementing a reranking step to improve search precision.

---
