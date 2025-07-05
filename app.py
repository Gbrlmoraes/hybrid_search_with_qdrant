import time

import streamlit as st

from modules.embedding_retriever import EmbeddingRetriever

st.set_page_config(
    page_title='Hybrid Search',
)

with st.sidebar:
    # Number of results selector
    top_k = st.slider(
        'Number of final results', min_value=1, max_value=10, value=3, step=1
    )

    top_k_sparse = st.slider(
        'Number of sparse results', min_value=1, max_value=20, value=10, step=1
    )

    top_k_dense = st.slider(
        'Number of dense results', min_value=1, max_value=20, value=10, step=1
    )

    # Filter labels selector
    labels = st.multiselect(
        'Filter by label',
        options=[
            'World',
            'Sports',
            'Business',
            'Sci/Tech',
        ],
        default=['Sci/Tech'],
    )


st.title('Hybrid Search')

# Initialize the retriever
if 'retriever' not in st.session_state:
    st.session_state.retriever = EmbeddingRetriever(
        qdrant_url='http://localhost:6333',
        collection_name='news_hybrid_search',
    )

# Make the query
query = st.text_input('Enter your query:')

if query:
    # Sparse search
    tic = time.time()
    sparse_result = st.session_state.retriever.sparse_query(
        query, top_k=top_k_sparse, labels=labels
    )
    tac = time.time()
    sparse_result_time_ms = (tac - tic) * 1000

    # Dense search
    tic = time.time()
    dense_result = st.session_state.retriever.dense_query(
        query, top_k=top_k_dense, labels=labels
    )
    tac = time.time()
    dense_result_time_ms = (tac - tic) * 1000

    # Final search (late interaction model)
    tic = time.time()
    result = st.session_state.retriever.query(
        query,
        top_k=top_k,
        labels=labels,
        top_k_dense=top_k_dense,
        top_k_sparse=top_k_sparse,
    )
    tac = time.time()
    result_time_ms = (tac - tic) * 1000

    st.markdown('### Result:')
    st.markdown('---')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('#### Sparse Search Results:')
        st.markdown(
            f'{sparse_result_time_ms:.2f} ms',
        )
        for point in sparse_result.points:
            st.markdown(f'Score: {point.score:.4f}')
            st.markdown(f'{point.payload["text"]}')
            st.markdown('---')

    with col2:
        st.markdown('#### Dense Search Results:')
        st.markdown(
            f'{dense_result_time_ms:.2f} ms',
        )
        for point in dense_result.points:
            st.markdown(f'Score: {point.score:.4f}')
            st.markdown(f'{point.payload["text"]}')
            st.markdown('---')

    with col3:
        st.markdown('#### Final Search Results:')
        st.markdown(
            f'{result_time_ms:.2f} ms',
        )
        for point in result.points:
            st.markdown(f'Score: {point.score:.4f}')
            st.markdown(f'{point.payload["text"]}')
            st.markdown('---')
