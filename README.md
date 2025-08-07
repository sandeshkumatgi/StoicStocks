# StoicStocks

StoicStocks is a modern chatbot application that combines rule-based generation with a Retrieval-Augmented Generation (RAG) pipeline. It uses natural language processing to understand user intent and provides responses either from a predefined set or by querying a knowledge base.

## Features

- Rule-based intent recognition for common queries
- RAG pipeline for complex queries using Weaviate as a vector database
- Llama3:8b language model running on Ollama for generating responses
- Nomic-embed-text model for text embeddings
- Sleek, responsive UI with real-time streaming of bot responses
- Toggle for context-aware responses

## Prerequisites

- Python 3.7+
- Flask
- Docker (for running Weaviate)
- Ollama (for running Llama3:8b and nomic-embed-text)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Hamhunter23/CDSAML-LLM-RAG.git
   cd stoicstocks
   ```

2. Install the required Python packages:
   ```
   pip install flask weaviate-client ollama langchain
   ```

3. Install and set up Docker:
   Follow the instructions at [Docker Installation Guide](https://docs.docker.com/get-docker/)

4. Install and set up Weaviate using Docker:
   ```
   docker-compose up -d
   ```

5. Install Ollama:
   Follow the instructions at [Ollama Installation Guide](https://ollama.ai/download)

6. Download the required models for Ollama:
   ```
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```

## Usage

1. Ensure the Weaviate Docker container is running:
   ```
   docker start weaviate
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to `http://localhost:5000`.

4. Start chatting with StoicStocks!

## Data Ingestion

To populate the Weaviate database with your custom data:

1. Ensure the Weaviate Docker container is running.

2. Run the data ingestion script:
   ```
   python charcterSplitter.py
   ```

This script will chunk the text, generate embeddings using the nomic-embed-text model, and upload the data to Weaviate.

## Customization

- Modify the `intents` and `responses` dictionaries in `app.py` to add or change rule-based responses.
- Adjust the chunking parameters in `chunk_and_upload.py` to optimize for your specific use case.
- Customize the UI by editing `templates/index.html`.

## Troubleshooting

- If you encounter issues with Weaviate, ensure the Docker container is running and accessible on port 8080.
- For Ollama-related problems, check that the required models (llama3:8b and nomic-embed-text) are correctly installed.
- If you face memory issues when running the Llama3:8b model, consider using a smaller model or adjusting Ollama's resource allocation.
