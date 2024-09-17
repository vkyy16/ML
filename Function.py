# Function.py
import streamlit as st
import pandas as pd
import numpy as np
import umap
import ast
import gensim
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go
from sklearn.manifold import TSNE



def preprocess_reviews(reviews):
    reviews['review'] = reviews['review'].apply(lambda x: ast.literal_eval(x))
    return reviews

def generate_matrix(reviews):
    processed_docs = reviews['review'].tolist()
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    reviews['review_text'] = reviews['review'].apply(lambda x: ' '.join(x))
    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(reviews['review_text'])
    return document_term_matrix, vectorizer


def generate_bow_matrix(reviews):
    processed_docs = reviews['review'].tolist()
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    vectorizer_bow = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    document_term_matrix_bow = vectorizer_bow.fit_transform([' '.join(doc) for doc in processed_docs])
    return document_term_matrix_bow, vectorizer_bow

def generate_tfidf_matrix(reviews):
    processed_docs = reviews['review'].tolist()
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
    document_term_matrix_tfidf = vectorizer_tfidf.fit_transform([' '.join(doc) for doc in processed_docs])
    return document_term_matrix_tfidf, vectorizer_tfidf

def plot_svd(X, labels):
    svd_model = TruncatedSVD(n_components=2)
    X_svd = svd_model.fit_transform(X)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('SVD Visualization of Clusters')
    st.pyplot(plt)

def plot_umap(X, labels):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP Visualization of Clusters')
    st.pyplot(plt)

# Function to run t-SNE
def run_tsne(X, y, dimension=3, learning_rate=200, max_iter=1000, perplexity=30):
    tsne_model = TSNE(n_components=dimension, random_state=0, learning_rate=learning_rate, max_iter=max_iter, perplexity=perplexity)
    X_tsne = tsne_model.fit_transform(X)
    X_tsne = pd.DataFrame(X_tsne, columns=[f'dim {i + 1}' for i in range(dimension)])
    X_tsne["label"] = y.astype(str)
    return X_tsne

# Function to create 3D Plotly t-SNE figure
def create_tsne_figure(dimension, X_tsne):
    traces = []
    if dimension == 3:
        for index, group in X_tsne.groupby('label'):
            scatter = go.Scatter3d(
                name=f'Class {index}',
                x=group['dim 1'],
                y=group['dim 2'],
                z=group['dim 3'],
                mode='markers',
                marker=dict(size=4, symbol='circle')
            )
            traces.append(scatter)
        fig = go.Figure(traces, layout=go.Layout(height=500, margin=dict(l=0, r=0, b=0, t=30), uirevision='foo'))
    else:
        for index, group in X_tsne.groupby('label'):
            scatter = go.Scatter(
                name=f'Class {index}',
                x=group['dim 1'],
                y=group['dim 2'],
                mode='markers',
                marker=dict(size=4.5, symbol='circle')
            )
            traces.append(scatter)
        fig = go.Figure(traces, layout=go.Layout(height=500, margin=dict(l=0, r=0, b=0, t=30), uirevision='foo'))
    return fig

# Define the function to print representative words for each cluster
def print_cluster_words(cluster_labels, document_term_matrix, feature_names, top_n=10):
    unique_labels = np.unique(cluster_labels)
    cluster_words = ""
    
    for label in unique_labels:
        if label != -1:
            # Handle non-noise clusters
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_matrix = document_term_matrix[cluster_indices]
            cluster_term_sums = np.asarray(cluster_matrix.sum(axis=0)).flatten()
            
            # Get the top terms
            top_indices = cluster_term_sums.argsort()[-top_n:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            cluster_words += f"**Cluster {label}:**\n" + ", ".join(top_terms) + "\n\n"
        else:
            # Handle noise points
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) > 0:
                cluster_matrix = document_term_matrix[cluster_indices]
                cluster_term_sums = np.asarray(cluster_matrix.sum(axis=0)).flatten()
                
                # Get the top terms for noise points
                top_indices = cluster_term_sums.argsort()[-top_n:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                
                cluster_words += f"**Noise Points (label {label}):**\n" + ", ".join(top_terms) + "\n\n"
            else:
                cluster_words += f"**Noise Points (label {label}):**\nNo data available\n\n"
    
    return cluster_words
