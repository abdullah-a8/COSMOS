from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
import os
import time
from functools import lru_cache

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

# Create a cached version of the embeddings creation to avoid redundant API calls
class CachedEmbeddings:
    """Wrapper around OpenAIEmbeddings that caches embedding results to avoid redundant API calls"""
    
    def __init__(self, model="text-embedding-3-large", cache_size=100):
        self.embeddings = OpenAIEmbeddings(model=model)
        self.cache_size = cache_size
        self._setup_cache()
    
    def _setup_cache(self):
        # Setup LRU cache for the embed_query method
        self.embed_query = lru_cache(maxsize=self.cache_size)(self.embed_query)
    
    def embed_query(self, text):
        """Generate embeddings for a single query text (cached)"""
        print(f"Generating embedding for query: '{text[:30]}...'")
        # Hash the text to create a unique cache key
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, documents):
        """Generate embeddings for a list of documents (not cached)"""
        return self.embeddings.embed_documents(documents)

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
        # Initialize cached OpenAI Embeddings - 3072 for text-embedding-3-large
        embeddings = CachedEmbeddings(model="text-embedding-3-large")
        
        print(f"Initializing Pinecone connection for index: {index_name}")
        # Connect to existing index for retrieval/adding
        vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
        print("Pinecone connection initialized.")
        return vector_store
    except Exception as e:
        print(f"Error initializing Pinecone connection: {e}")
        return None 