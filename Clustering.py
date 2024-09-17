# Clustering Dashboard.py
import streamlit as st
import pandas as pd
from Function import preprocess_reviews, generate_bow_matrix, generate_tfidf_matrix
from HDBSCAN import hdbscan_modeling
from Agglomerative import agglomerative_clustering_modeling
from GMM import gmm_clustering_modeling
from NMF import nmf_modeling
from LDA import lda_modeling
from Function import preprocess_reviews


# Set the title of the page
st.set_page_config(page_title="Clustering Dashboard")

# Streamlit app layout
st.title('Clustering Dashboard')

# Importing data (preprocessed, so no additional processing needed)
@st.cache_data
def load_data():
    reviews_df = pd.read_csv("cleaned_reviews.csv")
    return reviews_df

# Load data
reviews = load_data()

# Load and preprocess data
reviews = preprocess_reviews(reviews)

# Model selection
st.sidebar.header("Choose Clustering Model")
model_choice = st.sidebar.selectbox("Clustering Model", ["LDA", "HDBSCAN", "GMM", "NMF", "Agglomerative"])

# Run selected model
if model_choice == "HDBSCAN":
    hdbscan_modeling(reviews)
elif model_choice == "LDA":
    st.write("LDA model implementation")
    lda_modeling(reviews)
elif model_choice == "GMM":
    st.write("GMM model implementation")
    gmm_clustering_modeling(reviews)
elif model_choice == "NMF":
    st.write("NMF model implementation")
    nmf_modeling(reviews)
elif model_choice == "Agglomerative":
    st.write("Agglomerative model implementation")
    agglomerative_clustering_modeling(reviews)
