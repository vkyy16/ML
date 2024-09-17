import streamlit as st
import numpy as np
import pandas as pd
import pyLDAvis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Function import generate_matrix, plot_umap, print_cluster_words

def lda_modeling(reviews):
    # Sidebar configuration for LDA
    st.sidebar.header("LDA Model Parameters")
    best_n_components = st.sidebar.slider("Number of Topics", 2, 10, value=2)
    best_alpha = st.sidebar.slider("Alpha (Doc-Topic Prior)", 0.01, 1.0, value=0.775, step=0.01)
    best_beta = st.sidebar.slider("Beta (Topic-Word Prior)", 0.01, 1.0, value=0.5, step=0.01)

    # Visualization selection (restricted to UMAP and PyLDAvis)
    st.sidebar.header("Visualization Options")
    plot_choice = st.sidebar.selectbox("Choose visualization", ["UMAP", "PyLDAvis"])

    st.markdown("## LDA Topic Modeling")

    # Step 1: Generate the BoW matrix
    document_term_matrix, vectorizer = generate_matrix(reviews)
    st.markdown("### Using Bag of Words (BoW) matrix")

    st.write("")

    # Step 2: LDA Model Setup
    lda_model = LatentDirichletAllocation(
        n_components=best_n_components,
        doc_topic_prior=best_alpha,
        topic_word_prior=best_beta,
        random_state=42
    )

    # Step 3: Fit the LDA model
    lda_model.fit(document_term_matrix)

    # Step 4: Transform BoW matrix to document-topic distributions
    doc_topic_dists = lda_model.transform(document_term_matrix)

    # Step 5: Get dominant topics for each document
    dominant_topics = np.argmax(doc_topic_dists, axis=1)

    # Step 6: Create two columns for evaluation and cluster words
    col1, col2 = st.columns([2, 2])

    # Step 7: Evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    with col1:
        st.markdown("#### Evaluation")
        if len(set(dominant_topics)) > 1:  # Ensure there is more than one cluster
            silhouette_avg = silhouette_score(doc_topic_dists, dominant_topics, metric='euclidean')
            st.write(f'Silhouette Score: {silhouette_avg:.4f}')
            calinski_harabasz = calinski_harabasz_score(doc_topic_dists, dominant_topics)
            st.write(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')
            davies_bouldin = davies_bouldin_score(doc_topic_dists, dominant_topics)
            st.write(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
        else:
            st.write("Not enough clusters to compute metrics.")

    # Step 8: Print top words per topic
    with col2:
        st.markdown("#### Cluster Words")
        cluster_words = print_cluster_words(dominant_topics, document_term_matrix, vectorizer.get_feature_names_out())
        st.text_area("Cluster Words", value=cluster_words, height=400)

    # Step 9: Visualization based on user choice (UMAP or PyLDAvis)
    st.write("")
    st.markdown("#### Visualization")
    if plot_choice == "UMAP":
        plot_umap(doc_topic_dists, dominant_topics)

    # Step 10: PyLDAvis interactive topic visualization
    if plot_choice == "PyLDAvis":
        st.subheader("Interactive LDA Visualization")
        lda_vis_data = pyLDAvis.prepare(
            topic_term_dists=lda_model.components_,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=np.sum(document_term_matrix, axis=1),
            vocab=vectorizer.get_feature_names_out(),
            term_frequency=np.sum(document_term_matrix, axis=0)
        )
        pyLDAvis.display(lda_vis_data)
