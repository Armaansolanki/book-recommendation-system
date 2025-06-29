from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = Flask(__name__)

# ----------------- Load & Preprocess Data -----------------
df = pd.read_csv("amazon-products.csv")

# Drop irrelevant columns
df_cleaned = df.drop(columns=[
    'product_details', 'prices_breakdown', 'country_of_origin','timestamp','seller_name','description','initial_price',
    'buybox_seller','number_of_sellers','images', 'badge','root_bs_rank','answered_questions', 'upc', 'origin_url',
    'ingredients','parent_asin', 'input_asin','domain','images_count','url','video_count','image_url','item_weight',
    'product_dimensions','seller_id','date_first_available','format','buybox_prices','bought_past_month','is_available',
    'root_bs_category','plus_content','video','bs_category','bs_rank','model_number','manufacturer','subcategory_rank','variations'
], errors='ignore')

# Fill missing values
df_cleaned['features'] = df_cleaned['features'].fillna("No features listed")
df_cleaned['discount'] = df_cleaned['discount'].str.replace('%', '', regex=False)
df_cleaned['discount'] = pd.to_numeric(df_cleaned['discount'], errors='coerce')
df_cleaned['discount'] = df_cleaned['discount'].fillna(df_cleaned['discount'].median())
df_cleaned['amazon_choice'] = df_cleaned['amazon_choice'].fillna("0").replace({False: 0, True: 1})
df_cleaned['title'] = df_cleaned['title'].fillna("0")
df_cleaned['categories'] = df_cleaned['categories'].fillna("0")

# Combine text features
df_cleaned['combined'] = df_cleaned['title'] + ' ' + df_cleaned['categories'] + ' ' + df_cleaned['features']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cleaned['combined'])

# KNN Model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(tfidf_matrix)

# ----------------- API Route -----------------
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_title = data.get('title', '')

    # Find matching product
    matched_rows = df_cleaned[df_cleaned['title'].str.contains(user_title, case=False, na=False)]

    if not matched_rows.empty:
        product_index = matched_rows.index[0]
        matched_title = matched_rows.iloc[0]['title']

        # Get neighbors
        query_vector = tfidf_matrix[product_index]
        distances, indices = model_knn.kneighbors(query_vector, n_neighbors=6)

        recommended_titles = [df_cleaned.iloc[i]['title'] for i in indices.flatten() if i != product_index]

        return jsonify({
            "selected_product": matched_title,
            "recommendations": recommended_titles
        })
    else:
        return jsonify({"error": "No matching product found."}), 404

# ----------------- Home Route -----------------
@app.route('/')
def home():
    return "🎯 Product Recommendation API is running!"

# ----------------- Run App -----------------
if __name__ == '__main__':
    app.run(debug=True)
