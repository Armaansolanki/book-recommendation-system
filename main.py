import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import re

st.title("🎬 Movie Recommender System with CSV Upload & Merge")

# File upload section
file1 = st.sidebar.file_uploader("Upload First CSV File (e.g., ratings.csv)", type=["csv"], key="file1")
file2 = st.sidebar.file_uploader("Upload Second CSV File (e.g., movies.csv)", type=["csv"], key="file2")

if file1 is not None and file2 is not None:
    try:
        # Load CSVs
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)


        st.subheader("📄 First File Preview")
        st.write(df1.head())

        st.subheader("📄 Second File Preview")
        st.write(df2.head())

        # Merge on movieId
        if "movieId" in df1.columns and "movieId" in df2.columns:
            merged_df = pd.merge(df1, df2, on="movieId")
        else:
            st.error("❌ 'movieId' column not found in both files. Cannot merge.")
            st.stop()

        # Clean data
        if "timestamp" in merged_df.columns:
            merged_df.drop(columns=["timestamp"], inplace=True)
        merged_df.dropna(subset=["title"], inplace=True)

        st.subheader("🔗 Merged CSV (after cleaning)")
        st.write(merged_df.head())

        # Ratings count
        st.subheader("📊 Total Rating Count per Movie")
        movie_rating_counts = (
            merged_df.groupby("title")["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "total_rating"})
            .sort_values(by="total_rating", ascending=False)
        )
        st.write(movie_rating_counts.head())

        combined_df = pd.merge(merged_df, movie_rating_counts, on="title")

        # Pivot matrix
        matrix_df = combined_df.pivot_table(index="title", columns="userId", values="rating").fillna(0)
        sparse_matrix = scipy.sparse.csr_matrix(matrix_df.values)

        # Fit model
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(sparse_matrix)

        # Collaborative filtering input
        st.subheader("🎯 Movie Recommendation (Collaborative Filtering)")
        movie_input = st.text_input("Enter movie title (e.g., Toy Story)")

        if movie_input:
            matches = matrix_df.index[matrix_df.index.str.lower().str.contains(movie_input.lower())]

            if len(matches) == 0:
                st.error("❌ No matching movie found.")
            else:
                selected_movie = matches[0]
                movie_index = matrix_df.index.get_loc(selected_movie)

                distances, indices = model.kneighbors(
                    matrix_df.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=6)

                st.success(f"✅ Recommendations for: **{selected_movie}**")
                for i in range(1, len(indices.flatten())):
                    st.write(f"{i}. {matrix_df.index[indices.flatten()[i]]}")

        # Genre-based recommendations
        st.subheader("🎬 Genre-Based Recommendation")

        # Normalize all column names: lowercase and remove surrounding whitespace
        df1.columns = df1.columns.str.strip().str.lower()
        df2.columns = df2.columns.str.strip().str.lower()


        movie_list = merged_df['title'].unique().tolist()
        genre_selected_movie = st.selectbox("Select a movie to find its genres", movie_list)    

        # Find genres of selected movie
        genre_row = df2[df2['title'].str.lower() == genre_selected_movie.lower()]

        if not genre_row.empty and "genres" in genre_row.columns:
            genres = genre_row['genres'].values[0]

            if pd.isna(genres):
                st.warning("⚠️ Genre data missing for this movie.")
            else:
                genres_list = genres.split('|')
                st.write(f"🎭 Genres for **{genre_selected_movie}**: {', '.join(genres_list)}")

                # Escape special characters
                escaped_genres = [re.escape(genre) for genre in genres_list]

                # Filter by genre
                genre_filtered_df = df2[
                    (df2['genres'].str.contains('|'.join(escaped_genres), na=False, regex=True)) &
                    (df2['title'] != genre_selected_movie)
                ]

                st.subheader("📌 Top 5 Genre-Based Recommendations")
                st.write(genre_filtered_df[['title', 'genres']].drop_duplicates().head(5))
                st.success("✅ Genre-based recommendations generated successfully!")
        else:
            st.warning("⚠️ Genre information not found for the selected movie.")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("👈 Upload two CSV files using the sidebar to begin.")





