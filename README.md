# chromadb-vector-search
 A vector search application using ChromaDB".
ChromaDB Vector Search Application
This project implements an advanced vector search application using ChromaDB, combining text and image search capabilities for efficient and accurate retrieval of similar images.

Features
Multimodal Search: Combines text and image inputs for comprehensive search queries.
ChromaDB Integration: Utilizes ChromaDB for efficient vector storage and retrieval.
TF-IDF Weighted Embeddings: Enhances text search accuracy using TF-IDF weighting.
Real-time Image Processing: Processes uploaded images on-the-fly using OpenCV.
CLIP Model Integration: Leverages the CLIP model for generating image embeddings.
Flask Web Interface: Provides a user-friendly web interface for query input and result display.
Technologies Used
ChromaDB: Vector database for storing and querying embeddings
Flask: Web framework for the application interface
CLIP (ViT-B/32): For generating image embeddings
Sentence Transformers: For text embedding generation
OpenCV: For real-time image processing
NumPy: For numerical operations on embeddings
Pillow: For image handling
scikit-learn: For TF-IDF vectorization
Installation
Clone the repository
Install required packages: pip install -r requirements.txt
Ensure you have a sample_images directory with some images for the initial database
Usage
Run the application: python web_app.py
Open a web browser and navigate to http://localhost:5000
Enter a text query and upload an image
View the processed image and similar results
How It Works
The application combines text and image inputs for search queries.
Text input is processed using TF-IDF weighted embeddings.
Uploaded images are processed in real-time using OpenCV.
CLIP model generates embeddings for both text and images.
ChromaDB performs similarity search on the combined embeddings.
Results are displayed as similar images with similarity scores.
