from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np
import base64
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2

app = Flask(__name__)

# Initialize ChromaDB client
client = chromadb.Client(Settings(allow_reset=True))
image_collection = client.create_collection("image_collection")

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Function to get image embedding
def get_image_embedding(image):
    # Preprocess image
    image = image.resize((224, 224)).convert('RGB')
    img_emb = model.encode(image)
    return img_emb / np.linalg.norm(img_emb)  # Normalize embedding

# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to get top matches
def get_top_matches(query_embedding, collection_embeddings, k=5):
    similarities = [cosine_similarity(query_embedding, emb) for emb in collection_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices, [similarities[i] for i in top_indices]

# TF-IDF weighted embedding function
def get_tfidf_weighted_embedding(text, model, tfidf_vectorizer):
    words = text.lower().split()
    word_embeddings = np.array([model.encode(word) for word in words])
    tfidf_vector = tfidf_vectorizer.transform([text]).toarray()[0]
    weighted_embeddings = word_embeddings * tfidf_vector[:, np.newaxis]
    return np.mean(weighted_embeddings, axis=0)

# Real-time image processing function
def process_image_stream(image_stream):
    image = Image.open(image_stream)
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    processed_image = Image.fromarray(edges)
    return processed_image

# Add sample images
image_dir = "sample_images"
all_texts = []
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    embedding = get_image_embedding(Image.open(image_path))
    image_collection.add(
        embeddings=[embedding.tolist()],
        metadatas=[{"filename": image_file}],
        ids=[image_file]
    )
    all_texts.append(image_file)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_texts)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form['text_query']
        image_file = request.files['image_query']
        
        # Process the image in real-time
        processed_image = process_image_stream(image_file)
        
        # Get embeddings
        text_embedding = get_tfidf_weighted_embedding(text_query, model, tfidf_vectorizer)
        image_embedding = get_image_embedding(processed_image)
        
        # Combine embeddings with equal weight
        combined_embedding = (text_embedding + image_embedding) / 2
        
        # Get top matches
        collection_embeddings = [np.array(emb[0]) for emb in image_collection.get(include=['embeddings'])['embeddings']]
        top_indices, similarities = get_top_matches(combined_embedding, collection_embeddings)
        
        # Prepare results for display
        result_images = []
        for i, idx in enumerate(top_indices):
            img_file = image_collection.get(ids=[str(idx)])['ids'][0]
            img_path = os.path.join(image_dir, img_file)
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            result_images.append({'image': encoded_string, 'similarity': similarities[i]})
        
        # Encode processed image for display
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'images': result_images, 'processed_image': processed_image_base64})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
