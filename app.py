import os
import time

import pandas as pd
import streamlit as st

from modules.embedding_retriever import EmbeddingRetriever

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

st.set_page_config(
    page_title="Hybrid Search Explained",
    layout="wide",
)


@st.cache_resource
def get_dataset():
    """Reads the dataset from the parquet file in resources."""
    return pd.read_parquet(os.path.join(RESOURCES_DIR, "dataset.parquet"))


@st.cache_resource
def get_retriever():
    """Initializes the retriever and caches it for the session."""
    return EmbeddingRetriever(
        qdrant_url="http://localhost:6333",
        collection_name="products_hybrid_search",
    )


def display_search_results(title: str, time_ms: float, results, explanation: str):
    """A reusable function to display search results in a consistent format."""
    st.subheader(title)

    # Explanation for each search type
    with st.expander("What am I looking at?"):
        st.markdown(explanation)

    st.info(f"Query Time: {time_ms:.2f} ms")

    if not results or not results.points:
        st.warning("No results found.")
        return

    for point in results.points:
        with st.container(border=True):
            st.markdown(f"**Score: {point.score:.4f}**")
            st.write(point.payload["text"])


with st.sidebar:
    st.header("Search Parameters")
    top_k_sparse = st.slider("Number of sparse results", 1, 20, 5)
    top_k_dense = st.slider("Number of dense results", 1, 20, 5)
    top_k = st.slider("Number of final results", 1, 10, 5)

    st.header("Filters")
    label_options = get_dataset()["category"].unique().tolist()
    labels = st.multiselect(
        "Filter by category", options=label_options, default=label_options
    )

    st.header("About")
    st.markdown(
        """
        To know more about the techniques and the project implementation, 
        check out the [GitHub documentation](https://github.com/your-username/hybrid_search_with_qdrant).
        """
    )

st.title("🔬 Hybrid Search Explained")

PROJECT_DESCRIPTION = """
This interactive app demonstrates how a modern hybrid search system works by breaking it down into its core components. It allows you to visually compare the results from three distinct search strategies, all powered by models defined in your project:

-   **Sparse Search**: A fast keyword search using **`Qdrant/bm25`**.
-   **Dense Search**: A semantic search that understands meaning, powered by **`sentence-transformers/all-MiniLM-L6-v2`**.
-   **Final Reranked Search**: A sophisticated two-stage process where candidates from the first two searches are intelligently re-ordered by a **`colbert-ir/colbertv2.0`** late-interaction model for maximum accuracy.
"""

st.markdown(PROJECT_DESCRIPTION)

col1, col2, col3 = st.columns(spec=[1, 3, 1])

with col2:
    with st.container():
        st.image(
            os.path.join(ASSETS_DIR, "search_overview.png"),
            caption="Source: [Reranking Hybrid Search Results with Qdrant Vector Database](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)",
            use_container_width=True,
        )

st.markdown(
    "Use the sidebar controls to adjust the number of results from each initial search and see how the different components contribute to the final, high-quality outcome."
)

retriever = get_retriever()
query = st.text_input("Enter your query:", "High capacity SSD")

if query:
    # Performing each search type
    tic = time.time()
    sparse_result = retriever.sparse_query(query, top_k=top_k_sparse, labels=labels)
    sparse_result_time_ms = (time.time() - tic) * 1000

    tic = time.time()
    dense_result = retriever.dense_query(query, top_k=top_k_dense, labels=labels)
    dense_result_time_ms = (time.time() - tic) * 1000

    tic = time.time()
    final_result = retriever.query(
        query,
        top_k=top_k,
        labels=labels,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
    )
    final_result_time_ms = (time.time() - tic) * 1000

    # Explanations for Each Search
    SPARSE_EXPLANATION = """
    ### Sparse Search

    **Keyword-Based Search using `Qdrant/bm25`**

    - **How it works:** This method uses the classic BM25 algorithm to find documents that contain the exact keywords from your query. It scores documents based on the frequency and relevance of the query terms, giving less importance to common words (like 'the' or 'a').
    - **Strengths:** It is extremely fast and precise for matching specific product codes, acronyms, or technical terms. For example, searching for `S21` will reliably find products with that exact string.
    - **Weaknesses:** It has no understanding of semantics or concepts. A search for "quick charger" will *not* match a document that only says "fast power adapter."
    - **Good for**: Finding exact matches or specific terms that may not be semantically related. Example: Product model names, "like BN59-01259E", and specific keywords for technical specifications, such as "128Gb Storage"
    """

    DENSE_EXPLANATION = """
    ### Dense Search

    **Semantic Search using `sentence-transformers/all-MiniLM-L6-v2`**

    - **How it works:** This method uses an AI model to convert your query and the documents into numerical vectors that capture their *semantic meaning* or *intent*. The search finds the vectors that are closest in meaning using Cosine Similarity.
    - **Strengths:** It excels at finding conceptually similar items, even if they don't share keywords. It understands that "quick charger" and "fast power adapter" are related concepts.
    - **Weaknesses:** It can sometimes miss a result if a critical keyword is present but doesn't dominate the overall meaning of the text.
    - **Good for**: Finding products that match the intent of your query, even if they don't use the exact same words. Example: Searching for "fast charging cable" will return results that include "quick charger" or "rapid charge cable," even if those exact phrases aren't present.
    """

    FINAL_EXPLANATION = """
    ### Final (Reranked) Search

    **Hybrid Search with ColBERT Reranking (`colbert-ir/colbertv2.0`)**

    This is a sophisticated, multi-stage process that delivers the highest accuracy:

    1.  **Candidate Gathering (`prefetch`):** First, it performs a broad, efficient search using both the **Sparse (BM25)** and **Dense (MiniLM)** models to gather a combined list of promising candidates.
    2.  **Detailed Reranking:** Then, the powerful **ColBERT** model inspects this smaller list. Instead of comparing single vectors, ColBERT uses a **'late interaction'** technique. It compares the individual tokens of your query against the tokens of each candidate document, calculating a **Maximum Similarity (MaxSim)** score for the best final ranking.

    - **Result:** This combines the keyword precision of sparse search with the conceptual understanding of dense search, all refined by a powerful reranker for the most relevant results possible.
    """

    # Display the results
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        display_search_results(
            "1. Sparse Search", sparse_result_time_ms, sparse_result, SPARSE_EXPLANATION
        )
    with col2:
        display_search_results(
            "2. Dense Search", dense_result_time_ms, dense_result, DENSE_EXPLANATION
        )
    with col3:
        display_search_results(
            "3. Final (Reranked) Search", final_result_time_ms,
            final_result, FINAL_EXPLANATION
        )
