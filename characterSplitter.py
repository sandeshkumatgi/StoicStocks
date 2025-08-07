import weaviate
from langchain.text_splitter import CharacterTextSplitter
import ollama
import time
import uuid
import warnings
from PyPDF2 import PdfReader
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Initialize Weaviate client
print("Initializing Weaviate client...")
weaviate_client = weaviate.Client("http://localhost:8080")

# Custom embedding function using Ollama
def get_ollama_embedding(text):
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return embedding['embedding']

# Function to process a single PDF file
def process_pdf(file_path, source_folder):
    print(f"\nProcessing file: {file_path}")
    
    # Load and print the document
    print("Loading PDF file")
    pdf_reader = PdfReader(file_path)
    num_pages = len(pdf_reader.pages)
    print(f"Loaded PDF with {num_pages} page(s)")

    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"

    print("\nFull text content (first 500 characters):")
    print(full_text[:500] + "...")
    print("\n" + "="*50 + "\n")

    # Chunk the document
    print("Chunking document...")
    start_time1 = time.time()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100, separator='\n', strip_whitespace=False)
    chunks = text_splitter.split_text(full_text)
    end_time1 = time.time()
    print(f"\nText chunked in {end_time1 - start_time1:.2f} seconds.")
    print(f"Document split into {len(chunks)} chunks.")

    # Upload vectors to Weaviate
    print(f"Uploading vectors to Weaviate...")
    start_time = time.time()
    batch_size = 100
    with weaviate_client.batch as batch:
        batch.batch_size = batch_size
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}", end="\r")
            embedding = get_ollama_embedding(chunk)
            data_object = {
                "content": chunk,
                "source": source_folder
            }
            print("\nPrinting chunks:")
            print(chunk)
            print("-"*50)
            batch.add_data_object(
                data_object=data_object,
                class_name=class_name,
                vector=embedding,
                uuid=uuid.uuid4()
            )
    end_time = time.time()
    print(f"\nVectors uploaded to Weaviate in {end_time - start_time:.2f} seconds.")

# Read folder names from txt file
def read_folder_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Process all PDFs in a folder
def process_folder(folder_path):
    print(f"\nProcessing folder: {folder_path}")
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            process_pdf(file_path, folder_path)

# Main execution
if __name__ == "__main__":
    # Define Weaviate schema
    class_name = "StockPDF"
    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["string"]}
        ]
    }

    # Create the schema in Weaviate
    print(f"Creating schema in Weaviate for class: {class_name}")
    weaviate_client.schema.create_class(class_obj)

    # Read folder names
    folder_names = read_folder_names("Folder Names.txt")
    print(f"Found {len(folder_names)} folders to process")

    # Process each folder
    for folder_name in folder_names:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            process_folder(folder_name)
        else:
            print(f"Folder not found or is not a directory: {folder_name}")

    print("\nAll folders processed and PDFs uploaded to Weaviate.")