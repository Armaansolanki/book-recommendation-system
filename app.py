import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Set page config
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender System")
st.header("give me a movie, and I'll find similar ones!")

# File uploads
movie_file = st.sidebar.file_uploader("Upload a movie dataset (CSV only)", type=["csv"])
rating_file = st.sidebar.file_uploader("Upload a rating file (optional, CSV only)", type=["csv"])

# Load dataset dynamically
@st.cache_resource
def process_movie_data(movie_file):
    df = pd.read_csv(movie_file)
    
    # You must preprocess the data to create `movie_features_df` that works with KNN
    # For now, let's assume movie titles are in a column named 'title'
    df = df.dropna(subset=["title"])  # Drop rows with missing titles
    df.set_index("title", inplace=True)

    # Replace this with your actual feature extraction logic
    movie_features_df = df.select_dtypes(include='number')  # use numeric columns as features

    movie_matrix = csr_matrix(movie_features_df.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movie_matrix)

    return movie_features_df, movie_matrix, model_knn

if movie_file is not None:
    try:
        movie_features_df, movie_matrix, model_knn = process_movie_data(movie_file)

        # User input
        movie_name = st.text_input("Enter a movie name (e.g., Toy Story (1995))")

        if st.button("Get Recommendations"):
            matches = movie_features_df.index[
                movie_features_df.index.str.lower().str.contains(movie_name.lower())
            ]

            if len(matches) > 0:
                selected_movie = matches[0]
                index = movie_features_df.index.get_loc(selected_movie)
                distances, indices = model_knn.kneighbors(movie_matrix[index], n_neighbors=6)

                st.subheader(f"📽️ Recommendations for **{selected_movie}**:")
                for i in range(1, len(indices.flatten())):
                    similar_movie = movie_features_df.index[indices.flatten()[i]]
                    similarity_score = distances.flatten()[i]
                    st.write(f"{i}. {similar_movie} — Distance: `{similarity_score:.4f}`")
            else:
                st.error("❌ Movie not found in the dataset. Please try another title.")
    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.warning("👈 Please upload a movie dataset to begin.")
