# Agglomerative Model
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from Function import generate_bow_matrix, generate_tfidf_matrix, plot_svd, plot_umap, print_cluster_words

# Function to reduce dimensions using SVD
def reduce_dimensions(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced

# Function to plot dendrogram using linkage matrix
def plot_dendrogram(linkage_matrix, **kwargs):
    # Plot the hierarchical clustering as a dendrogram.
    dendrogram(linkage_matrix, **kwargs)

# Streamlit dashboard for Agglomerative Clustering
def agglomerative_clustering_modeling(reviews):
    # Sidebar for Agglomerative Clustering parameters
    st.sidebar.header("Agglomerative Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of clusters", 5, 10, 5)
    metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan", "cosine"])
    linkage_method = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
    matrix_choice = st.sidebar.radio("Choose Document-Term Matrix", ["BoW", "TF-IDF"])

    # Sidebar for visualization options
    st.sidebar.header("Visualization Options")
    plot_choice = st.sidebar.selectbox("Choose visualization", ["SVD", "UMAP", "Dendrogram"])

    # Main page header
    st.markdown("## Agglomerative Clustering")
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

    # Compute linkage matrix for dendrogram
    linkage_matrix = linkage(document_term_matrix_reduced, method=linkage_method, metric=metric)

    # Perform Agglomerative Clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage_method)
    labels = model.fit_predict(document_term_matrix_reduced)

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
    elif plot_choice == "Dendrogram":
        st.markdown("### Dendrogram")
        fig, ax = plt.subplots()
        plot_dendrogram(linkage_matrix, ax=ax)
        st.pyplot(fig)

