# Multi-Model RAG Chatbot

![RAG Chatbot Banner](https://)

## Project Intuition

Interacting with large amounts of text data from documents and web articles can be cumbersome. Finding specific information or getting summaries often requires manual effort. This project was started to create an intelligent chatbot that could understand and answer questions based on the content of provided documents (PDFs) and web articles.

The core idea is to use **Retrieval-Augmented Generation (RAG)**. Instead of relying solely on a Large Language Model's (LLM) internal knowledge (which can be outdated or lack specific details), the RAG approach first retrieves relevant passages from the user-provided documents and then uses those passages to help the LLM generate a more accurate and contextually relevant answer.

This chatbot aims to be a flexible tool for researchers, students, and anyone needing to quickly query and understand text-based content, leveraging multiple powerful LLMs for diverse needs.

## Overview

This project provides an interactive RAG (Retrieval-Augmented Generation) Chatbot built with Streamlit. It allows users to upload PDF documents or provide web URLs, processes the content, stores it in a searchable knowledge base (vector database), and enables users to ask questions about the content using various Large Language Models (LLMs).

The chatbot retrieves relevant information from the ingested documents before generating answers, ensuring responses are grounded in the provided source material.

## Features

### 🤖 RAG Chatbot (`1_RAG_Chatbot.py`)
- **Multi-Model Flexibility**: Supports various LLMs via Groq (Llama, Mistral, Mixtral, Gemma, etc.), allowing users to choose based on performance or preference.
- **Document Ingestion**: Processes text content from:
    - PDF files uploaded by the user.
    - Web articles specified by URL.
- **Vector Knowledge Base**: Chunks the ingested content, generates vector embeddings (using OpenAI), and stores them in a Pinecone vector database for efficient similarity search.
- **Contextual Chat**: Implements the RAG pattern:
    1. Retrieves relevant text chunks from Pinecone based on user query similarity.
    2. Augments the LLM prompt with this retrieved context.
    3. Generates an answer grounded in the provided documents.
- **Source Metadata**: Associates chunks with their original source (PDF filename hash or URL).
- **Adjustable Parameters**: Allows users to control LLM temperature (randomness), text chunk size, and chunk overlap for processing.

## Technical Architecture

The chatbot is built using Python, Streamlit, LangChain, and Pinecone.

### Core Components (`core/`)

- **`data_extraction.py`**: Extracts text from PDFs (`PyMuPDF`) and URLs (`newspaper4k`).
- **`processing.py`**: Splits extracted text into overlapping chunks (`RecursiveCharacterTextSplitter`) and adds source metadata.
- **`vector_store.py`**: Manages the connection to Pinecone and handles embedding generation (`OpenAIEmbeddings`) and upserting of text chunks.
- **`chain.py`**: Defines the LangChain Expression Language (LCEL) chain, including the prompt template (`ChatPromptTemplate`), the selected Groq LLM (`ChatGroq`), and the output parser (`StrOutputParser`).

### Configuration (`config/`)

- **`settings.py`**: Stores default values for temperature, chunk size, etc.
- **`prompts.py`**: Contains the main RAG prompt template guiding the LLM.

### User Interface (`Home.py`, `pages/1_RAG_Chatbot.py`)

- **`Home.py`**: The main Streamlit application landing page.
- **`pages/1_RAG_Chatbot.py`**: The specific page providing the UI for uploading content, setting parameters, and interacting with the chatbot.

### Data Flow

1. **Upload/Input**: User uploads a PDF or provides a URL via the Streamlit UI.
2. **Extraction**: `data_extraction.py` gets the raw text.
3. **Processing**: `processing.py` chunks the text and adds metadata.
4. **Embedding & Storage**: `vector_store.py` creates embeddings and stores chunks in Pinecone.
5. **Query**: User asks a question in the chat interface.
6. **Retrieval**: Relevant text chunks are retrieved from Pinecone based on semantic similarity to the query.
7. **Augmentation**: Retrieved chunks are formatted and inserted into the prompt template along with the user's question.
8. **Generation**: The complete prompt is sent to the selected LLM via `chain.py`.
9. **Display**: The LLM's generated response is shown in the UI.

## Getting Started

### Prerequisites

- Python 3.13.2+
- Pip (Python package installer)
- Git
- Pinecone Account & API Key: Create an index at [Pinecone](https://www.pinecone.io/).
- OpenAI API Key: Needed for text embeddings. Obtain from [OpenAI](https://platform.openai.com/signup/).
- Groq API Key (Optional but Recommended): For using various LLMs via Groq. Obtain from [Groq](https://groq.com/).

### Installation

1.  **Clone the Repository**:
    ```bash
    # Use the appropriate URL for your repository
    git clone https://github.com/abdullah-a8/COSMOS.git
    cd COSMOS
    ```

2.  **Set Up Virtual Environment**: (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # OR
    .venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` only contains dependencies needed for the RAG chatbot and its core components)*

4.  **Configure Environment Variables**:
    Create a file named `.env` in the project root:
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key"
    GROQ_API_KEY="your_groq_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_INDEX_NAME="your_pinecone_index_name"
    # GOOGLE_API_KEY might not be needed for this specific version
    ```
    **Important**: Your Pinecone index must use a dimension compatible with OpenAI's `text-embedding-3-large` (3072 dimensions).

### Running the Application

1.  **Start Streamlit**:
    ```bash
    streamlit run Home.py
    ```

2.  **Access the App**: Open your browser to `http://localhost:8501` (or the URL provided).

## Usage Guide

1.  Navigate to the **RAG Chatbot** page (it might be the only page besides Home).
2.  Use the sidebar to upload PDF files or enter web URLs. Wait for processing confirmation.
3.  Adjust the LLM model, temperature, chunk size, and overlap settings in the sidebar if desired.
4.  Type your questions about the content you added into the chat input field.
5.  The chatbot will retrieve relevant information and generate an answer.

## Customization

### Modifying LLM Settings

Defaults can be changed in `config/settings.py`.

### Modifying Prompts

The core RAG prompt is in `config/prompts.py`. Adjusting this can significantly alter the chatbot's response style and focus.

## Project Structure

```
RAG-Chatbot/
├── Home.py                   # Main application entry point
├── pages/
│   └── 1_RAG_Chatbot.py     # RAG interface for document Q&A
├── core/
│   ├── chain.py             # LLM chain management
│   ├── data_extraction.py   # Source content extraction (PDF, URL)
│   ├── processing.py        # Text processing pipeline
│   └── vector_store.py      # Vector database operations
├── config/
│   ├── settings.py          # Default parameters
│   └── prompts.py           # RAG prompt template
├── requirements.txt         # Project dependencies (subset for RAG)
├── .env                     # Environment variables (gitignored)
└── README.md                # This documentation file
```

## Acknowledgments

-   Built with [Streamlit](https://streamlit.io/)
-   Uses [LangChain](https://www.langchain.com/)
-   Vector search powered by [Pinecone](https://www.pinecone.io/)
-   PDF parsing via [PyMuPDF](https://pymupdf.readthedocs.io/)
-   Web scraping via [newspaper4k](https://github.com/funkeeler/newspaper4k)