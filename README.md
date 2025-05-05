# COSMOS - Collaborative Organized System for Multiple Operating Specialists

![COSMOS Banner](./Images/COSMOS_1.png)

## Project Intuition

In today's information-rich world, dealing with diverse content streams – documents, online articles, videos, emails – can be overwhelming. Traditional tools often operate in silos, forcing users to switch contexts constantly. COSMOS was born from the idea of creating a unified, intelligent workspace where specialized AI agents can collaborate to manage this information seamlessly.

The core intuition is to leverage the strengths of different AI models and techniques:

- **Retrieval-Augmented Generation (RAG)**: Provides a powerful way to ground AI responses in factual information from provided sources, reducing hallucinations and improving accuracy when chatting about documents or other knowledge base content.
- **Specialized Agents**: Recognizes that different tasks require different approaches. A dedicated Gmail agent understands email context better, while a YouTube processor knows how to handle transcripts effectively.
- **Collaborative Ecosystem**: Instead of isolated tools, COSMOS integrates these agents. Knowledge extracted by one agent (like the YouTube processor) becomes available to others (like the RAG chatbot), creating a synergistic effect.

COSMOS aims to be more than just a collection of tools; it's envisioned as an extensible platform where new intelligent agents can be added over time, creating a truly comprehensive digital assistant.

## Overview

COSMOS is an integrated AI assistant platform built using Streamlit. It brings together multiple specialized agents, each designed for a specific task, allowing them to work together within a single interface. Whether you need to query documents, process YouTube videos, or manage your Gmail inbox, COSMOS provides dedicated AI specialists to help.

The system combines Retrieval-Augmented Generation (RAG) for knowledge-based chat with specialized agents for tasks like email handling and video transcript processing, creating a versatile and collaborative AI workspace.

## Features

### 🤖 RAG Chatbot (`1_RAG_Chatbot.py`)
- **Multi-Model Flexibility**: Choose from a wide range of Large Language Models (LLMs) via Groq (Llama, Mistral, Mixtral, Gemma, etc.) to tailor performance and cost.
- **Unified Knowledge Base**: Ingests and processes content from PDFs, web URLs, and YouTube transcripts (processed by the YouTube agent), making all information searchable in one place.
- **OCR Processing**: Extracts text from images and image-containing PDFs using Mistral AI's OCR capabilities, adding this content to the knowledge base.
- **Contextual Chat**: Uses RAG to retrieve relevant text chunks from the knowledge base before generating answers, ensuring responses are grounded in the provided content.
- **Source Tracking & Filtering**: Identifies the source of information (PDF, URL, YouTube, OCR) and allows users to filter which sources the chatbot should use during retrieval.
- **Fine-tuning Parameters**: Allows users to adjust LLM temperature, chunk size, and chunk overlap to optimize retrieval and response generation for different needs.

### 🎥 YouTube Processor (`2_YouTube_Processor.py`)
- **Automated Transcript Extraction**: Simply provide a YouTube URL, and the agent fetches the video transcript using the `youtube-transcript-api`.
- **Content Preparation**: Processes the extracted transcript, breaking it into manageable chunks suitable for the vector database, using the same core processing logic as other content types.
- **Knowledge Base Integration**: Embeds and stores the transcript chunks in the Pinecone vector database, making the video content searchable by the RAG Chatbot.
- **User Feedback**: Provides real-time status updates during processing and displays the video thumbnail for confirmation.

### 📧 Gmail Response Assistant (`3_Gmail_Agent.py`)
- **Secure Authentication**: Uses Google OAuth 2.0 for secure access to your Gmail account.
- **Intelligent Email Processing**: Fetches emails based on user queries (e.g., `is:unread`).
- **AI-Powered Analysis**: Uses the OpenAI API directly to:
    - **Classification**: Automatically categorize emails (e.g., Inquiry, Promotion, Personal).
    - **Summarization**: Generate concise summaries of long emails.
- **Contextual Reply Generation**: Drafts email replies using the OpenAI API based on the original email's content, subject, and sender, allowing users to specify tone, style, length, and provide additional context.
- **Direct Sending**: Sends the drafted (and potentially edited) replies directly from the interface using the Gmail API, automatically handling threading.
- **Label Management**: Marks replied-to emails as read automatically by removing the `UNREAD` label.

# New Feature: Mistral OCR Integration

COSMOS now integrates Mistral AI's OCR capabilities, allowing you to:

- Process images and extract text using state-of-the-art OCR technology
- Process PDFs with both text and images, combining regular PDF extraction with OCR for images
- Seamlessly add OCR-processed content to your knowledge base for RAG-powered chat

## Setup

To use the Mistral OCR feature, you need to:

1. Add your Mistral API key to your environment variables:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   ```

2. Select "Mistral OCR" from the input methods in the RAG Chatbot page
3. Upload an image or PDF with images to process

## Supported File Types

- Images: JPG, JPEG, PNG, BMP, TIFF, TIF
- PDFs with text and images

## Implementation Details

The Mistral OCR integration uses the Mistral AI REST API directly to ensure compatibility and reliability:

- Images are encoded to base64 and sent to the Mistral OCR API endpoint
- For PDFs, the text is extracted directly using PyMuPDF, and each image within the PDF is processed separately
- Combined results (extracted text + OCR results from images) are then processed through the standard RAG pipeline
- OCR-processed content is tagged with a special source type ("ocr") in the vector database, allowing for filtering in search

## Technical Architecture

COSMOS utilizes a modular Python architecture centered around Streamlit for the UI and LangChain for orchestrating AI components.

### Core Components (`core/`)

- **`data_extraction.py`**: Contains functions to extract raw text content from different sources (PDFs via `PyMuPDF`, URLs via `newspaper4k`, YouTube transcripts via `youtube-transcript-api`, images via Mistral OCR).
- **`processing.py`**: Takes raw text and a source identifier, performs text splitting (`RecursiveCharacterTextSplitter` from LangChain), enriches chunks with metadata (source type, URL, domain, timestamp, chunk sequence), and prepares them for embedding.
- **`vector_store.py`**: Handles interactions with the Pinecone vector database. It initializes the connection using environment variables and provides functions to add processed document chunks (with embeddings generated via `OpenAIEmbeddings` using the `text-embedding-3-large` model) to the specified Pinecone index.
- **`chain.py`**: Sets up the core LangChain sequence (LCEL) for the RAG functionality. It defines the prompt template (`ChatPromptTemplate`), initializes the selected ChatGroq LLM model with specific temperature settings, and includes the output parser (`StrOutputParser`).
- **`agents/gmail_logic.py`**: Encapsulates all logic related to the Gmail agent, including OAuth authentication (`google-auth-oauthlib`, `google-api-python-client`), fetching/sending emails, and interacting *directly* with the OpenAI API (`openai` library) for classification, summarization, and reply generation using predefined prompts.
- **`mistral_ocr.py`**: Handles OCR processing for images and PDFs with images using the Mistral AI API, with implementations for both SDK-based and REST API approaches.

### C++ Extensions (`cpp_extensions/`)

COSMOS leverages high-performance C++ modules to accelerate critical operations, with fallback to Python implementations when necessary.

- **`text_chunking/text_chunker.cpp`**: A C++ implementation of text chunking that significantly improves processing speed compared to the Python implementation. It mimics LangChain's RecursiveCharacterTextSplitter with optimized algorithms for splitting text by paragraphs, newlines, or characters.
- **`pdf_extraction/pdf_extractor.cpp`**: Combines PDF text extraction and hash generation into a single optimized C++ operation. Uses Poppler for fast PDF parsing with minimal overhead.
- **`hash_generation/hash_generator.cpp`**: Provides optimized SHA-256 hash generation for content identification and verification, with specialized versions for different Python data types.

These extensions are built using pybind11 for seamless Python integration and implement graceful fallback mechanisms. The system checks for availability of the C++ modules at runtime and defaults to pure Python implementations if needed, ensuring compatibility across different environments while providing substantial performance benefits when available.

### Configuration (`config/`)

- **`settings.py`**: Stores default configuration values like LLM temperature, chunk size, and overlap.
- **`prompts.py`**: Contains the various prompt templates used by the RAG chain and the Gmail agent for tasks like classification, summarization, and reply generation.

### User Interface (`Home.py`, `pages/`)

- **`Home.py`**: The main landing page of the Streamlit application.
- **`pages/`**: Contains individual Streamlit scripts for each major feature/agent (RAG Chatbot, YouTube Processor, Gmail Agent), providing the user interface and orchestrating calls to the core logic components.

### Data Flow (RAG Example)

1. **Upload/Input**: User provides a PDF, URL, or YouTube URL via the Streamlit UI.
2. **Extraction**: `data_extraction.py` extracts the text content.
3. **Processing**: `processing.py` splits text into chunks and adds metadata.
4. **Embedding & Storage**: `vector_store.py` uses `OpenAIEmbeddings` to create vector representations of the chunks and upserts them into Pinecone.
5. **Query**: User asks a question in the RAG Chatbot UI.
6. **Retrieval**: The Pinecone vector store is queried (using `vector_store.as_retriever`) to find the most relevant text chunks based on the question's embedding.
7. **Augmentation**: The retrieved chunks (context) are combined with the original question using the prompt template in `config/prompts.py`.
8. **Generation**: The augmented prompt is sent to the selected LLM (via `chain.py`) to generate a response.
9. **Display**: The LLM's response is displayed in the Streamlit UI.

## Getting Started

### Prerequisites

- Python 3.13.2+
- Pip (Python package installer)
- Git
- C++ compiler (GCC/Clang for Linux/macOS, MSVC for Windows)
- CMake 3.30.0+ (for building C++ extensions)
- OpenSSL development libraries (e.g., `openssl-devel` on Fedora/RHEL, `libssl-dev` on Debian/Ubuntu)
- Poppler development libraries (e.g., `poppler-cpp-devel` on Fedora/RHEL, `libpoppler-cpp-dev` on Debian/Ubuntu)
- Pinecone Account: Sign up at [Pinecone](https://www.pinecone.io/) and create an index. Note the API key and index name.
- OpenAI API Key: Obtain from [OpenAI](https://platform.openai.com/signup/) (used for embeddings and Gmail agent functions).
- Groq API Key (Optional but Recommended): Obtain from [Groq](https://groq.com/) for access to fast LLMs used in the RAG chatbot.
- Google Cloud Project with Gmail API Enabled:
    - Follow Google's documentation to create a project and enable the Gmail API.
    - Create OAuth 2.0 Client ID credentials (Desktop application type).
    - Download the `credentials.json` file.

### Installation

1.  **Clone the Repository**:
    ```bash
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

4.  **Configure Environment Variables**:
    Create a file named `.env` in the project root directory and add your API keys and Pinecone details. The application uses the `python-dotenv` library to load these variables.
    ```dotenv
    OPENAI_API_KEY="your_openai_api_key"
    GROQ_API_KEY="your_groq_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_INDEX_NAME="your_pinecone_index_name"
    MISTRAL_API_KEY="your_mistral_api_key"
    ```
    **Important**: Ensure your Pinecone index is configured with **3072 dimensions** to match the `text-embedding-3-large` model used for OpenAI embeddings.

5.  **Build C++ Extensions** (Optional but Recommended):
    These extensions improve performance for text chunking, PDF extraction, and hashing. Build them *after* installing system dependencies:
    ```bash
    cd cpp_extensions
    python setup.py build_ext --inplace
    cd .. 
    
    # Copy extensions to the correct location for the application to find
    # Ensure the target directory exists
    mkdir -p core/cpp_modules
    # Use find to copy all .so files, handling potential platform differences
    find cpp_extensions -name '*.so' -exec cp {} core/cpp_modules/ \;
    ```
    The application uses an `__init__.py` in `core/cpp_modules/` to dynamically load these extensions if available, falling back to pure Python implementations otherwise.

6.  **Set Up Gmail Credentials**:
    - Create a `credentials` directory in the project root if it doesn't exist: `mkdir -p credentials`
    - Place the `credentials.json` file you downloaded from Google Cloud inside this directory.
    - **Rename** the file to `.gmail_credentials.json`. (The leading dot helps keep it slightly hidden and matches the code).
    *(This file is ignored by `.gitignore` to prevent accidental commits)*

### Running the Application

1.  **Start Streamlit**:
    ```bash
    streamlit run Home.py
    ```

2.  **Access the App**: Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Gmail Authentication (First Run)**: When you navigate to the Gmail Agent page for the first time, you'll be prompted to authenticate. Click the "Connect to Gmail" button. This will open a browser window (or provide a URL) for you to log in to your Google account and grant permission. Follow the on-screen instructions. This process uses a local server (typically on port 8090) to receive the authorization code. A `.gmail_token.json` file will be created in the `credentials` directory to store your authorization token for future sessions.

## Usage Guide

### Working with the RAG Chatbot

1.  Navigate to the **RAG Chatbot** page via the sidebar.
2.  **Add Content**: 
    - Use the sidebar options to upload PDF files or enter web URLs
    - Upload images or PDFs with images to leverage OCR processing
    - Watch as the system processes your content and adds it to the Pinecone knowledge base
3.  **Configure Settings**: Adjust the LLM model, temperature, chunk size, and overlap in the sidebar to control how the chatbot retrieves information and generates responses.
4.  **Chat**: Type your questions about the ingested content into the chat input at the bottom.
5.  **Filter Sources**: Use the sidebar options to filter which types of sources (PDF, URL, YouTube, OCR) the chatbot should search when answering your questions.

### Processing YouTube Videos

1.  Go to the **YouTube Processor** page.
2.  Paste the full URL of the YouTube video you want to process.
3.  Click **Process YouTube Transcript**.
4.  Wait for the processing to complete (status updates are shown).
5.  Once successful, the video's transcript is added to the knowledge base and can be queried via the RAG Chatbot.

### Using the Gmail Assistant

1.  Navigate to the **Gmail Agent** page.
2.  **Authenticate**: If not already connected, click "Connect to Gmail" and follow the Google authentication flow.
3.  **Fetch Emails**: Use the search query input (default `is:unread`) and adjust the maximum results slider, then click "Fetch Emails".
4.  **Select Email**: Choose an email from the dropdown list in the sidebar.
5.  **Analyze**:
    - Click **Classify Email** to get an AI-determined category.
    - Click **Summarize Email** to generate a concise summary.
6.  **Generate Reply**:
    - Select the desired tone, style, and length for the reply.
    - Add any specific instructions or context in the "Optional Context" box.
    - Click **Generate Draft Reply**.
7.  **Edit & Send**: Review the generated draft in the text area. Make any necessary edits, then click **Send Reply**. The email will be sent using your Gmail account, and the original email will be marked as read.

## Customization

### Modifying LLM Settings

Default parameters like temperature and chunking settings can be adjusted directly in `config/settings.py`.

### Adding/Modifying Prompts

The prompts used for RAG, email classification, summarization, and reply generation are stored in `config/prompts.py`. You can edit these to change the AI's behavior or tailor its responses.

### Extending with New Agents

The modular structure (`core/agents/`) is designed for extension. To add a new agent (e.g., a Calendar Agent):
1. Create a new logic file (e.g., `core/agents/calendar_logic.py`) containing the core functionality (API interaction, processing logic).
2. Create a new Streamlit page (e.g., `pages/4_Calendar_Agent.py`) for the UI, importing functions from your new logic file.
3. Add necessary dependencies to `requirements.txt`.
4. Update environment variables (`.env`) and configuration (`config/`) if needed.

## Project Structure

```
COSMOS/
├── app.py                    # Main application script (if applicable, e.g., Flask/FastAPI)
├── config.yaml               # Main configuration file
├── Home.py                   # Main Streamlit app landing page
├── pages/
│   ├── 1_RAG_Chatbot.py     # UI for RAG chat functionality
│   ├── 2_YouTube_Processor.py # UI for YouTube transcript processing
│   └── 3_Gmail_Agent.py     # UI for Gmail integration and response
├── core/
│   ├── __init__.py
│   ├── chain.py             # LangChain setup for LLM interaction (RAG)
│   ├── data_extraction.py   # Functions for extracting text from sources
│   ├── mistral_ocr.py       # OCR implementation using Mistral AI
│   ├── processing.py        # Text chunking and metadata enrichment logic
│   ├── vector_store.py      # Pinecone connection and vector operations
│   ├── agents/
│   │   ├── __init__.py
│   │   └── gmail_logic.py   # Core logic for Gmail agent (API, OpenAI tasks)
│   └── cpp_modules/         # Compiled C++ extensions (loaded dynamically)
│       ├── __init__.py
│       └── *.so             # Platform-specific shared object files
├── cpp_extensions/
│   ├── text_chunking/       # C++ text chunking source and CMake
│   ├── pdf_extraction/      # C++ PDF extraction source and CMake
│   ├── hash_generation/     # C++ hash generation source and CMake
│   ├── setup.py             # Build script for C++ extensions using pybind11
│   └── CMakeLists.txt       # Top-level CMake configuration
├── config/
│   ├── __init__.py
│   ├── settings.py          # Default application settings (chunk size, temp)
│   └── prompts.py           # Prompt templates for LLM interactions
├── credentials/             # Directory for credentials (token, .json) - gitignored
│   └── .gitkeep             # Placeholder to keep directory in git if empty
├── requirements.txt         # Python package dependencies
├── .env                     # Environment variables (API keys, etc.) - gitignored
├── .gitignore               # Specifies intentionally untracked files for Git
├── LICENSE                  # Project License file
└── README.md               # This documentation file
```

### Development Guidelines

-   **Code Style**: Please follow PEP 8 guidelines for Python code.
-   **Documentation**: Add docstrings to new functions/classes and update this README if your changes affect usage or setup.
-   **Testing**: While formal tests aren't currently implemented, ensure your changes work as expected and don't break existing functionality.
-   **Dependencies**: Add any new Python dependencies to `requirements.txt` and ensure `python-dotenv` is included.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

-   Built with the amazing [Streamlit](https://streamlit.io/) framework.
-   Leverages the power of [LangChain](https://www.langchain.com/) for LLM application development.
-   Vector database capabilities provided by [Pinecone](https://www.pinecone.io/).
-   PDF parsing thanks to [PyMuPDF](https://pymupdf.readthedocs.io/).
-   Web scraping via [newspaper4k](https://github.com/funkeeler/newspaper4k).
-   YouTube transcriptions via [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api).
-   Gmail integration via the [Google API Python Client](https://github.com/googleapis/google-api-python-client).
-   C++ extensions built with [pybind11](https://github.com/pybind/pybind11) for Python-C++ interoperability.
-   PDF processing in C++ powered by [Poppler](https://poppler.freedesktop.org/).
-   Cryptographic operations via [OpenSSL](https://www.openssl.org/).
-   OCR capabilities powered by [Mistral AI](https://mistral.ai/).

---