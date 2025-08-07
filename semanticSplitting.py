import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import time
import uuid
import warnings
from PyPDF2 import PdfReader
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Initialize Weaviate client
print("Initializing Weaviate client...")
weaviate_client = weaviate.Client("http://localhost:8080")

# Custom embedding function using Ollama
def get_ollama_embedding(text):
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return embedding['embedding']

# Function to visualize chunks
def visualize_chunks(embeddings, clusters):
    print("Visualizing chunks...")
    # Convert list of embeddings to a NumPy array
    embeddings = np.array(embeddings)
    
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot the reduced embeddings
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Semantic Chunks Visualization (Clustered)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.savefig('chunks_visualization_clustered.png')
    plt.close()
    print("Visualization saved as 'chunks_visualization_clustered.png'")

# Function to process a single PDF file
def process_pdf(file_path):
    print(f"\nProcessing file: {file_path}")
    
    pdf_reader = PdfReader(file_path)
    num_pages = len(pdf_reader.pages)
    print(f"Loaded PDF with {num_pages} page(s)")

    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"

    # Chunk the document using semantic splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Document split into {len(chunks)} chunks.")

    return chunks, file_path

# Read folder names from txt file
def read_folder_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Process all PDFs in all folders
def process_all_pdfs(folder_names):
    all_chunks = []
    all_sources = []
    
    for folder_name in folder_names:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            print(f"\nProcessing folder: {folder_name}")
            for file_name in os.listdir(folder_name):
                if file_name.lower().endswith('.pdf'):
                    file_path = os.path.join(folder_name, file_name)
                    chunks, source = process_pdf(file_path)
                    all_chunks.extend(chunks)
                    all_sources.extend([source] * len(chunks))
        else:
            print(f"Folder not found or is not a directory: {folder_name}")
    
    return all_chunks, all_sources

# Main execution
if __name__ == "__main__":
    # Define Weaviate schema
    class_name = "SemanticTest1"
    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["string"]},
            {"name": "cluster_id", "dataType": ["int"]}
        ]
    }

    # Create the schema in Weaviate
    print(f"Creating schema in Weaviate for class: {class_name}")
    weaviate_client.schema.create_class(class_obj)

    # Read folder names
    folder_names = read_folder_names("Folder Names.txt")
    print(f"Found {len(folder_names)} folders to process")

    # Process all PDFs
    all_chunks, all_sources = process_all_pdfs(folder_names)
    print(f"\nTotal chunks across all PDFs: {len(all_chunks)}")

    # Generate embeddings for all chunks
    print("Generating embeddings for all chunks...")
    start_time = time.time()
    all_embeddings = [get_ollama_embedding(chunk) for chunk in all_chunks]
    end_time = time.time()
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

    # Cluster the chunks
    print("Clustering chunks...")
    n_clusters = min(len(all_chunks) // 10, 100)  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)

    # Visualize clustered chunks
    visualize_chunks(all_embeddings, cluster_labels)

    # Upload vectors to Weaviate
    print(f"Uploading vectors to Weaviate...")
    start_time = time.time()
    batch_size = 100
    with weaviate_client.batch as batch:
        batch.batch_size = batch_size
        for i, (chunk, embedding, source, cluster_id) in enumerate(zip(all_chunks, all_embeddings, all_sources, cluster_labels)):
            print(f"Processing chunk {i+1}/{len(all_chunks)}", end="\r")
            data_object = {
                "content": chunk,
                "source": source,
                "cluster_id": int(cluster_id)
            }
            batch.add_data_object(
                data_object=data_object,
                class_name=class_name,
                vector=embedding,
                uuid=uuid.uuid4()
            )
    end_time = time.time()
    print(f"\nVectors uploaded to Weaviate in {end_time - start_time:.2f} seconds.")

    print("\nAll PDFs processed, chunks clustered, and data uploaded to Weaviate.")
