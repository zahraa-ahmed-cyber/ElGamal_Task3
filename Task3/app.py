# To run the app : streamlit run app.py --server.enableXsrfProtection false

import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Function to extract features from an image using the pre-trained VGG model
def extract_features_from_image(img):
    img = img.resize((224, 224)) 
    img = np.array(img)
    img = img[..., ::-1] 
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = vgg_model.predict(img)
    features = features.flatten()
    return features

# Function to calculate cosine similarity
def calculate_cosine_similarity(query_vector, database_vectors):
    similarities = cosine_similarity(query_vector.reshape(1, -1), database_vectors)
    return similarities.flatten()

# Function to retrieve similar images
def retrieve_similar_images(query_vector, database_vectors, top_k=5):
    similarities = calculate_cosine_similarity(query_vector, database_vectors)
    similar_indices = np.argsort(similarities)[::-1][:top_k]
    return similar_indices

# Function to display similar images
# def display_similar_images(uploaded_image , similar_indices, images_path):
#     st.subheader("Query Image:")
    
#     st.image(uploaded_image, caption='Query Image', use_column_width=True)

#     st.subheader("Similar Images:")
#     cols = st.columns(len(similar_indices))
#     for i, idx in enumerate(similar_indices):
#         similar_img_path = images_path[idx]
#         similar_img = Image.open(similar_img_path)
#         cols[i].image(similar_img, caption=f'Similar Image {i+1}', use_column_width=True)

# Function to display similar images
def display_similar_images(uploaded_image , similar_indices, images_path):
    st.subheader("Query Image:")
    # Resize the query image to a smaller size
    uploaded_image_small = uploaded_image.resize((200, 200))
    st.image(uploaded_image_small, caption='Query Image', use_column_width=False)

    st.subheader("Similar Images:")
    cols = st.columns(len(similar_indices))
    for i, idx in enumerate(similar_indices):
        similar_img_path = images_path[idx]
        similar_img = Image.open(similar_img_path)
        cols[i].image(similar_img, caption=f'Similar Image {i+1}', use_column_width=True)



# Load precomputed features
vector_database = np.load("vector_database.npy")

# Load image paths
root_folder = "C:/Users/zahra/Downloads/Capstone_Architectural_Styles_dataset"
image_paths = []

for folder, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')): 
            image_paths.append(os.path.join(folder, file))

# Load pre-trained VGG model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Streamlit app
st.title('Similar Image Retrieval')

uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    st.subheader('Uploaded Image:')
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    try:
        uploaded_image = Image.open(uploaded_image)
        uploaded_image_features = extract_features_from_image(uploaded_image)
    except Exception as e:
        st.error(f"Error: Unable to process uploaded image: {e}")
        uploaded_image_features = None

    if uploaded_image_features is not None:
        similar_indices = retrieve_similar_images(uploaded_image_features, vector_database, top_k=5)
        display_similar_images(uploaded_image, similar_indices, image_paths)


