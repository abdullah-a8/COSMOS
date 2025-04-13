from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
import os

def add_chunks_to_vector_store(vector_store, chunks, chunk_ids):
    if not chunks or not chunk_ids:
        print("No chunks or IDs provided to add_chunks_to_vector_store.")
        return False
    if not vector_store:
        print("Vector store object is None, cannot add chunks.")
        return False

    try:
        source_info = chunks[0].metadata.get('source_id', 'N/A') if chunks else 'N/A'
        print(f"Adding/updating {len(chunks)} chunks in Pinecone for source ID associated with first chunk: {source_info}")
        vector_store.add_documents(documents=chunks, ids=chunk_ids)
        print("Successfully added/updated chunks in Pinecone.")
        return True
    except Exception as e:
        print(f"Error adding documents to Pinecone: {e}")
        return False

def get_pinecone_vector_store():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY") 

    if not index_name:
        print("Error: PINECONE_INDEX_NAME environment variable not set.")
        return None
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY environment variable not set.")
        return None
    if not openai_api_key:
         print("Error: OPENAI_API_KEY environment variable not set (needed for embeddings).")
         return None

    try:
        # Initialize OpenAI Embeddings - 3072 for text-embedding-3-large
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        print(f"Initializing Pinecone connection for index: {index_name}")
        # Connect to existing index for retrieval/adding
        vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
        print("Pinecone connection initialized.")
        return vector_store
    except Exception as e:
        print(f"Error initializing Pinecone connection: {e}")
        return None 