# GMM Model
import pandas as pd
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from Function import generate_bow_matrix, generate_tfidf_matrix, plot_svd, plot_umap, print_cluster_words

# Function to reduce dimensions using SVD
def reduce_dimensions(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced

# Function to perform GMM clustering
def perform_gmm_clustering(X, n_clusters, covariance_type):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type)
    gmm.fit(X)
    labels = gmm.predict(X)
    return labels, gmm

# Streamlit dashboard for GMM Clustering
def gmm_clustering_modeling(reviews):
    # Sidebar for GMM clustering parameters
    st.sidebar.header("GMM Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
    covariance_type = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
    matrix_choice = st.sidebar.radio("Choose Document-Term Matrix", ["BoW", "TF-IDF"])

    # Sidebar for visualization options
    st.sidebar.header("Visualization Options")
    plot_choice = st.sidebar.selectbox("Choose visualization", ["SVD", "UMAP"])

    # Main page header
    st.markdown("## GMM Clustering")
    st.write("")

    # Generate the document-term matrix based on user's choice
    if matrix_choice == "BoW":
        document_term_matrix, vectorizer = generate_bow_matrix(reviews)
        st.markdown("### Using Bag of Words (BoW) matrix")
    elif matrix_choice == "TF-IDF":
        document_term_matrix, vectorizer = generate_tfidf_matrix(reviews)
        st.markdown("### Using TF-IDF matrix")

    st.write("")

    # Apply dimensionality reduction
    document_term_matrix_reduced = reduce_dimensions(document_term_matrix)

    # Perform GMM Clustering
    labels, gmm = perform_gmm_clustering(document_term_matrix_reduced, n_clusters, covariance_type)

    # Create columns for displaying results
    col1, col2 = st.columns([2, 2])

    with col1:
        st.markdown("#### Evaluation")
        # Check if there are enough clusters for metrics
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(document_term_matrix_reduced, labels, metric='euclidean')
            st.write(f'Silhouette Score: {silhouette_avg:.4f}')
            calinski_harabasz = calinski_harabasz_score(document_term_matrix_reduced, labels)
            st.write(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')
            davies_bouldin = davies_bouldin_score(document_term_matrix_reduced, labels)
            st.write(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
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


