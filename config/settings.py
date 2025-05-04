import os
from dotenv import load_dotenv
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Model Configuration ---
DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.7

# --- Processing Configuration ---
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 20

# --- OCR Configuration ---
MAX_IMAGES_PER_PDF = 10
MISTRAL_OCR_API_ENDPOINT = "https://api.mistral.ai/v1/ocr"