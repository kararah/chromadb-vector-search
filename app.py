import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Initialize ChromaDB client
client = chromadb.Client(Settings(allow_reset=True))

# Create a collection for images
image_collection = client.create_collection("image_collection")

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Function to get image embedding
def get_image_embedding(image_path):
    img = Image.open(image_path)
    img_emb = model.encode(img)
    return img_emb.tolist()

# Function to display images
def display_images(image_files):
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 5))
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        if len(image_files) > 1:
            axes[i].imshow(img)
            axes[i].set_title(img_file)
            axes[i].axis('off')
        else:
            axes.imshow(img)
            axes.set_title(img_file)
            axes.axis('off')
    plt.tight_layout()
    plt.show()

# Add sample images
image_dir = "sample_images"
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    embedding = get_image_embedding(image_path)
    image_collection.add(
        embeddings=[embedding],
        metadatas=[{"filename": image_file}],
        ids=[image_file]
    )

# Multi-modal query function
def multi_modal_query(text_query, image_query_path, n_results=2):
    text_embedding = model.encode(text_query).tolist()
    image_embedding = get_image_embedding(image_query_path)
    
    # Combine text and image embeddings
    combined_embedding = np.mean([text_embedding, image_embedding], axis=0).tolist()
    
    results = image_collection.query(
        query_embeddings=[combined_embedding],
        n_results=n_results
    )
    return results

# Example usage
text_query = "a cat sitting on a couch"
image_query_path = os.path.join(image_dir, "query_image.jpeg")


results = multi_modal_query(text_query, image_query_path)

# Display the results
result_images = [id for sublist in results['ids'] for id in sublist]
display_images(result_images)

print(results)
