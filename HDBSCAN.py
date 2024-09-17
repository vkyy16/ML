# HBDSCAN model
# HBDSCAN model
import hdbscan
import pandas as pd
import streamlit as st
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from Function import generate_bow_matrix, generate_tfidf_matrix, plot_svd, plot_umap, print_cluster_words

def reduce_dimensions(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    return X_reduced

def perform_hdbscan(X, min_cluster_size, min_samples, cluster_selection_epsilon, distance_metric):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=distance_metric)
    labels = clusterer.fit_predict(X)
    return labels

def hdbscan_modeling(reviews):
    # HDBSCAN parameters
    st.sidebar.header("HDBSCAN Parameters")
    min_cluster_size = st.sidebar.slider("Min cluster size", 5, 10)
    min_samples = st.sidebar.slider("Min samples", 5, 15)
    distance_metrics = st.sidebar.selectbox("Distance metric", ["manhattan", "euclidean", "cosine"])
    matrix_choice = st.sidebar.radio("Choose Document-Term Matrix", ["BoW", "TF-IDF"])

    # Display the visualization in the right column
    st.sidebar.header("Visualization Options")
    plot_choice = st.sidebar.selectbox("Choose visualization", ["SVD", "UMAP"])

    st.markdown("## HDBSCAN Clustering")
    st.write("")

    # Generate the document-term matrix
    if matrix_choice == "BoW":
        document_term_matrix, vectorizer = generate_bow_matrix(reviews)
        st.markdown("### Using Bag of Words (BoW) matrix")
    elif matrix_choice == "TF-IDF":
        document_term_matrix, vectorizer = generate_tfidf_matrix(reviews)
        st.markdown("### Using TF-IDF matrix")

    st.write("")
    
    # Apply dimensionality reduction
    document_term_matrix_reduced = reduce_dimensions(document_term_matrix)

    # Perform clustering
    labels = perform_hdbscan(document_term_matrix_reduced, min_cluster_size, min_samples, distance_metrics)

    # Create three columns: one for scores, one for visualization, one for cluster words
    col1, col2 = st.columns([2, 2])  # Adjust the ratio to allocate more space to the cluster words

    with col1:
        st.markdown("#### Evaluation")
        # Display clustering metrics in the left column
        if len(set(labels)) > 1:  # Check if there is more than one cluster
            silhouette_avg = silhouette_score(document_term_matrix_reduced, labels, metric='euclidean')
            st.write(f'Silhouette Score: {silhouette_avg:.4f}')
            calinski_harabasz = calinski_harabasz_score(document_term_matrix_reduced, labels)
            st.write(f'Calinski-Harabasz Score: {calinski_harabasz:.4f}')
            davies_bouldin = davies_bouldin_score(document_term_matrix_reduced, labels)
            st.write(f'Davies-Bouldin Score: {davies_bouldin:.4f}')
        else:
            st.write("Not enough clusters to compute metrics.")

    st.write("")
    with col2:
        st.markdown("#### Cluster Words")
        # Print representative words for each cluster
        cluster_words = print_cluster_words(labels, document_term_matrix, vectorizer.get_feature_names_out())
        st.text_area("Cluster Words", value=cluster_words, height=400)

    st.write("")    
    
    st.markdown("#### Visualization")
    if plot_choice == "SVD":
        plot_svd(document_term_matrix_reduced, labels)
    elif plot_choice == "UMAP":
        plot_umap(document_term_matrix_reduced, labels)