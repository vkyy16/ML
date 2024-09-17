import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from Function import generate_bow_matrix, generate_tfidf_matrix, plot_svd, plot_umap, print_cluster_words, run_tsne, create_tsne_figure

def reduce_dimensions(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced

def perform_nmf_topic_modeling(X, n_topics, random_state=42):
    nmf_model = NMF(n_components=n_topics, random_state=random_state)
    W = nmf_model.fit_transform(X)  # Document-topic matrix
    H = nmf_model.components_  # Topic-term matrix

    # Assign each document to the topic with the highest weight
    labels = np.argmax(W, axis=1)
    
    return labels, nmf_model, W, H

def compute_coherence_score(nmf_model, reviews, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    W = nmf_model.transform(reviews)
    H = nmf_model.components_

    # Create a dictionary for coherence calculation
    texts = [review.split() for review in reviews]
    dictionary = Dictionary(texts)

    top_terms = [[feature_names[i] for i in topic.argsort()[-10:]] for topic in H]
    cm = CoherenceModel(topics=top_terms, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = cm.get_coherence()
    
    return coherence_score

def nmf_modeling(reviews):
    # NMF parameters
    st.sidebar.header("NMF Parameters")
    n_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)  # Number of topics for NMF

    matrix_choice = st.sidebar.radio("Choose Document-Term Matrix", ["BoW", "TF-IDF"])

    # Display the visualization in the right column
    st.sidebar.header("Visualization Options")
    plot_choice = st.sidebar.selectbox("Choose Visualization", ["SVD", "UMAP", "2D t-SNE", "3D t-SNE"])

    # t-SNE parameters
    st.sidebar.header("t-SNE Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 10, 500, 200)
    max_iter = st.sidebar.slider("Number of Iterations", 250, 2000, 1000)
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)

    st.markdown("## NMF Topic Modeling")
    st.write("")

    # Generate the document-term matrix
    if matrix_choice == "BoW":
        document_term_matrix, vectorizer = generate_bow_matrix(reviews)
        st.markdown("### Using Bag of Words (BoW) Matrix")
    elif matrix_choice == "TF-IDF":
        document_term_matrix, vectorizer = generate_tfidf_matrix(reviews)
        st.markdown("### Using TF-IDF Matrix")

    st.write("")

    # Perform NMF Topic Modeling using the original document-term matrix
    labels, nmf_model, W, H = perform_nmf_topic_modeling(document_term_matrix, n_topics)

    # Calculate evaluation metrics
    reconstruction_error = nmf_model.reconstruction_err_
    cosine_similarities = cosine_similarity(H)

    # Apply dimensionality reduction for visualization
    document_term_matrix_reduced = reduce_dimensions(document_term_matrix)

    # Create columns for displaying results
    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("#### Evaluation")
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(document_term_matrix, labels, metric='euclidean')
            st.write(f'Silhouette Score: {silhouette_avg:.4f}')
            st.write(f'Reconstruction Error: {reconstruction_error:.4f}')
            
            # Format Cosine Similarities
            similarities_str = "Cosine Similarities Between Topics:\n"
            for i, row in enumerate(cosine_similarities):
                similarities_str += f"Topic {i}: {', '.join([f'Topic {j} ({value:.2f})' for j, value in enumerate(row) if i != j])}\n"
            st.text_area("Cosine Similarities", value=similarities_str, height=300)

        else:
            st.write("Not enough clusters to compute metrics.")

    with col2:
        st.markdown("#### Cluster Words")
        cluster_words = print_cluster_words(labels, document_term_matrix, vectorizer.get_feature_names_out())
        st.text_area("Cluster Words", value=cluster_words, height=400)

    st.write("")

    st.markdown("#### Visualization")
    if plot_choice == "SVD":
        plot_svd(document_term_matrix_reduced, labels)
    elif plot_choice == "UMAP":
        plot_umap(document_term_matrix_reduced, labels)
    elif plot_choice == "2D t-SNE":
        X_tsne = run_tsne(document_term_matrix_reduced, labels, dimension=2, learning_rate=learning_rate, max_iter=max_iter, perplexity=perplexity)
        fig = create_tsne_figure(dimension=2, X_tsne=X_tsne)
        st.plotly_chart(fig)
    elif plot_choice == "3D t-SNE":
        X_tsne = run_tsne(document_term_matrix_reduced, labels, dimension=3, learning_rate=learning_rate, max_iter=max_iter, perplexity=perplexity)
        fig = create_tsne_figure(dimension=3, X_tsne=X_tsne)
        st.plotly_chart(fig)
