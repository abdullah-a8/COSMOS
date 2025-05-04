import streamlit as st
from core.data_extraction import extract_text_from_pdf, extract_text_from_url, extract_content_with_ocr
from core.processing import process_content
from core.vector_store import get_pinecone_vector_store, add_chunks_to_vector_store
from core.chain import get_chain, ask_question, get_fast_chain
import config.settings as settings

from streamlit import cache_data, cache_resource
import time
import datetime
import concurrent.futures
import re
import os

# --- Authentication Check --- 
# Ensure the user is logged in, otherwise stop execution
if not st.session_state.get('authentication_status'):
    st.warning("Please log in to access this page.")
    st.stop() # Stop execution if not authenticated

# Streamlit App Configuration
st.set_page_config(page_title="Multi-Model RAG Chatbot 📄🔍", page_icon="🧬", layout="wide")
st.title("Multi-Model RAG-Powered Article Chatbot 📄🔍")

# Sidebar Settings
st.sidebar.title("Settings")

# Model selection options with performance characteristics
models = {
    # Meta's Latest Models
    "llama-3.3-70b-versatile": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 100_000,
        "advantages": "Latest versatile model with 128K context window and superior reasoning capabilities.",
        "disadvantages": "Limited daily requests compared to smaller models.",
    },
    "llama-4-maverick-17b-128e-instruct": {
        "requests_per_minute": 30,
        "requests_per_day": 7_000,
        "tokens_per_minute": 15_000,
        "tokens_per_day": 500_000,
        "advantages": "Latest Llama 4 model with superior reasoning despite smaller parameter count.",
        "disadvantages": "New model with potentially varying performance across different query types.",
    },
    "llama-3.1-8b-instant": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 20_000,
        "tokens_per_day": 500_000,
        "advantages": "High-speed processing with 128K context, great for real-time applications.",
        "disadvantages": "Less accurate for complex reasoning tasks compared to larger models.",
    },
    # Specialized Models
    "deepseek-r1-distill-llama-70b": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": None,  # Unlimited token capacity
        "advantages": "Specialized for mathematical and logical reasoning tasks with 128K context.",
        "disadvantages": "Limited daily requests compared to other models.",
    },
    "compound-beta": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 10_000,
        "tokens_per_day": 250_000,
        "advantages": "Groq's agentic system optimized for complex reasoning and tool use.",
        "disadvantages": "Beta status with potential variability in performance.",
    },
    # Additional Models
    "gemma2-9b-it": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 15_000,
        "tokens_per_day": 500_000,
        "advantages": "Google's efficient model with strong instruction-following capabilities.",
        "disadvantages": "Limited versatility compared to larger models.",
    },
    "qwen-qwq-32b": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 10_000,
        "tokens_per_day": 500_000,
        "advantages": "Strong multilingual capabilities and good knowledge representation.",
        "disadvantages": "May require more system resources than smaller models.",
    },
    # Previous Generation (Maintained for compatibility)
    "llama3-70b-8192": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 500_000,
        "advantages": "Long-context capabilities, well-tested for handling detailed research papers.",
        "disadvantages": "Older model compared to Llama 3.1/3.3 versions.",
    },
}

# Allow users to select multiple models
selected_models = st.sidebar.multiselect(
    "Choose Models",
    options=list(models.keys()),
    default=["llama-3.3-70b-versatile"],  # Updated default model selection
)

# Fallback default if no model is selected
if not selected_models:
    selected_models = ["llama-3.3-70b-versatile"]

# Display details for all selected models
st.sidebar.write("## Selected Model Details")
for model_name in selected_models:
    selected_model_details = models[model_name]
    st.sidebar.write(f"### {model_name}")
    for key, value in selected_model_details.items():
        if key not in ["advantages", "disadvantages"]:
            st.sidebar.write(f"- **{key.replace('_', ' ').title()}**: {value}")
    st.sidebar.write(f"- **Advantages**: {selected_model_details['advantages']}")
    st.sidebar.write(f"- **Disadvantages**: {selected_model_details['disadvantages']}")
    st.sidebar.write("---")  # Separator for clarity

# Temperature slider using default from settings
st.sidebar.markdown(
    """
    ### Temperature:
    Controls the randomness of the model's output. 
    - **Low Values (e.g., 0.1–0.3):** Makes the responses more deterministic and focused.  
    - **Medium Values (e.g., 0.7–1.0):** Balanced creativity and focus.  
    - **High Values (e.g., 1.5–2.0):** Increases creativity and variability, but may lead to less accurate or unpredictable responses.
    """
)
temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.0,
    max_value=2.0,
    value=settings.DEFAULT_TEMPERATURE,
    step=0.1,
    help="Adjust the randomness of the model's responses. Lower values = more focused; higher values = more creative."
)

# Chunk size slider using default from settings
st.sidebar.markdown(
    """
    ### Chunk Size:
    Defines the number of words in each content chunk. 
    - **Small Chunks (e.g., 100–300):** Ideal for highly specific queries or short articles.  
    - **Large Chunks (e.g., 1000–3000):** Better for summarization or broader context but may lose finer details.
    """
)
chunk_size = st.sidebar.slider(
    label="Chunk Size",
    min_value=100,
    max_value=3000,
    value=settings.DEFAULT_CHUNK_SIZE,
    step=50,
    help="Set the number of words in each content chunk. Choose smaller values for better specificity or larger for broader context."
)

# Chunk overlap slider using default from settings
st.sidebar.markdown(
    """
    ### Chunk Overlap:
    Specifies the overlap between consecutive content chunks.  
    - **Smaller Overlap (e.g., 10–50):** Reduces redundancy but may miss context in some queries.  
    - **Larger Overlap (e.g., 200–300):** Ensures more context but may increase processing time.
    """
)
chunk_overlap = st.sidebar.slider(
    label="Chunk Overlap",
    min_value=10,
    max_value=300,
    value=settings.DEFAULT_CHUNK_OVERLAP,
    step=10,
    help="Control the overlap of consecutive content chunks. Larger values improve context but may slow down processing."
)

# Session state initialization
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = [{"role": "assistant", "content": "Connecting to knowledge base... How can I help?"}]

if "rag_conversation_history" not in st.session_state:
    st.session_state["rag_conversation_history"] = ""

if "rag_vector_store_initialized" not in st.session_state:
    st.session_state["rag_vector_store_initialized"] = False

if "rag_current_source_id" not in st.session_state:
    st.session_state["rag_current_source_id"] = None

# Initialize the last processed ID tracker
if "rag_last_processed_source_id" not in st.session_state:
    st.session_state["rag_last_processed_source_id"] = None

# Initialize vector store once at page load or after reset
if "rag_vector_store" not in st.session_state or st.session_state.get("rag_vector_store") is None:
    try:
        with st.spinner("Connecting to knowledge base..."):
            # Direct initialization to avoid dependency on the cached function
            vector_store = get_pinecone_vector_store()
            if vector_store:
                st.session_state["rag_vector_store"] = vector_store
                st.session_state["rag_vector_store_initialized"] = True
                print("Pinecone connection initialized at page load.")
            else:
                st.session_state["rag_vector_store"] = None
                st.session_state["rag_vector_store_initialized"] = False
                print("Failed to initialize Pinecone connection at page load.")
                st.sidebar.warning("⚠️ Could not connect to the knowledge base. Some features may be limited.")
    except Exception as e:
        st.session_state["rag_vector_store"] = None
        st.session_state["rag_vector_store_initialized"] = False
        print(f"Error initializing Pinecone connection: {e}")
        st.sidebar.error(f"⚠️ Error connecting to knowledge base: {str(e)}")

# Reset App Function
def reset_app():
    """Set a reset flag to True and trigger app re-run."""
    st.session_state["reset"] = True
    st.rerun()

# Add Reset App button in the sidebar
st.sidebar.button("Reset Chat & Clear Upload", on_click=reset_app)

# Check if reset is flagged
if st.session_state.get("reset", False):
    # Identify keys to preserve (authentication-related)
    keys_to_keep = ['authentication_status', 'username', 'name', 'authenticator', 'logout']
    
    # Save values of keys we want to keep
    preserved_values = {}
    for key in keys_to_keep:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]
    
    # Clear all session state variables except those in keys_to_keep
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value
        
    # Clear caches but maintain authentication
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["reset"] = False
    
    # Explicitly flag that we need to reinitialize the vector store
    st.session_state["rag_vector_store_initialized"] = False
    st.session_state["rag_vector_store"] = None
    st.rerun()

# Content Upload Section
st.sidebar.title("Add Content to Knowledge Base")
input_method = st.sidebar.radio("Input Method", ["PDF File", "URL", "Image(s)"])

uploaded_file = None
url = None
content = None
source_id = None

# Cache extracted text
@cache_data
def cached_extract_text(input_method, uploaded_file_key=None, url=None):
    _content, _source_id = None, None
    if input_method == "PDF File" and uploaded_file_key:
        _content, _source_id = extract_text_from_pdf(uploaded_file_key)
    elif input_method == "URL" and url:
        _content, _source_id = extract_text_from_url(url)
    elif input_method == "Image(s)" and uploaded_file_key:
        _content, _source_id = extract_content_with_ocr(uploaded_file_key)

    if _source_id:
        st.session_state["rag_current_source_id"] = _source_id

    return _content, _source_id

# Cache processed content
@cache_data
def cached_process_content(_source_id, _content, _chunk_size, _chunk_overlap):
    print(f"Processing content for source: {_source_id}")
    return process_content(_content, _chunk_size, _chunk_overlap, _source_id)

# Cache vector store creation
@cache_resource
def cached_get_pinecone_vector_store():
    print("Attempting to initialize Pinecone connection (cached)...")
    vs = get_pinecone_vector_store()
    if vs:
        st.session_state["rag_vector_store_initialized"] = True
        print("Pinecone connection successful (cached).")
    else:
        print("Failed to initialize Pinecone connection (cached).")
        st.session_state["rag_vector_store_initialized"] = False
    return vs

def preprocess_query(query):
    """Extract key terms and concepts to make retrieval more efficient"""
    # Remove common stop words and keep meaningful terms
    stop_words = r'\b(the|a|an|in|on|at|to|for|with|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|shall|should|may|might|must|can|could)\b'
    cleaned_query = re.sub(stop_words, '', query.lower())
    # Remove extra spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    # If cleaning made the query too short, return original
    if len(cleaned_query.split()) < 3:
        return query
        
    return cleaned_query

# Process uploaded inputs
if input_method == "PDF File":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF File", key="pdf_uploader", type=["pdf"])
    if uploaded_file:
        content, source_id = cached_extract_text(input_method, uploaded_file_key=uploaded_file, url=None)
elif input_method == "URL":
    url = st.sidebar.text_input("Enter a News URL", key="url_input")
    if url:
        content, source_id = cached_extract_text(input_method, uploaded_file_key=None, url=url)
elif input_method == "Image(s)":
    st.sidebar.markdown("""
    ### Mistral OCR Processing
    Upload images or PDFs with images to extract text using Mistral's OCR capabilities.
    """)
    
    # Check if Mistral API key is available
    if not os.getenv("MISTRAL_API_KEY"):
        st.sidebar.error("""
        **Mistral API Key Missing!**
        
        Please add your Mistral API key to your environment variables:
        ```
        MISTRAL_API_KEY=your_mistral_api_key_here
        ```
        """)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image or PDF with Images", 
            key="ocr_uploader", 
            type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif"]
        )
        
        if uploaded_file:
            with st.sidebar.status("Processing with Mistral OCR...") as status:
                try:
                    content, source_id = cached_extract_text(input_method, uploaded_file_key=uploaded_file, url=None)
                    
                    if content and content.startswith("Error"):
                        status.update(label="OCR Processing Failed", state="error")
                        st.sidebar.error(content)
                    elif not source_id:
                        status.update(label="OCR Processing Failed", state="error")
                        st.sidebar.error("Failed to generate a valid source ID.")
                    else:
                        status.update(label="OCR Processing Complete", state="complete")
                except Exception as e:
                    status.update(label="OCR Processing Failed", state="error")
                    st.sidebar.error(f"Error processing with OCR: {str(e)}")

# Determine if processing should occur
last_processed_id = st.session_state.get("rag_last_processed_source_id")
# Process only if we have new content that wasn't processed before
should_process = content and source_id and (last_processed_id is None or source_id != last_processed_id)

# Process and add content only if conditions are met
if should_process:
    if content is not None and isinstance(content, str) and content.startswith("Error"):
        st.sidebar.error(f"Failed to extract content: {content}")
    elif content:
        st.sidebar.success(f"Extracted content from: {source_id[:50]}... ({len(content.split())} words)")
        
        # Create two columns for status and timer
        status_col, timer_col = st.sidebar.columns([3, 1])
        
        # Timer setup
        start_time = time.time()
        timer_placeholder = timer_col.empty()
        
        with status_col.status("Processing and adding to knowledge base...", expanded=True) as status:
            st.write("Splitting into chunks...")
            
            # Update timer in a loop while processing
            def update_timer():
                elapsed = time.time() - start_time
                timer_placeholder.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px; text-align: center;">
                    ⏱️ <span style="font-size: 1.1em; font-weight: bold;">{datetime.timedelta(seconds=int(elapsed))}</span>
                </div>
                """, unsafe_allow_html=True)
            
            update_timer()
            
            # Process content using cached function
            chunks, chunk_ids = cached_process_content(source_id, content, chunk_size, chunk_overlap)
            update_timer()
            
            if chunks:
                st.write(f"Created {len(chunks)} chunks.")
                st.write("Adding to Pinecone...")
                
                update_timer()

                # Get vector store from session state
                vector_store = st.session_state.get("rag_vector_store")
                
                if vector_store:
                    # Add chunks to the vector store
                    success = add_chunks_to_vector_store(vector_store, chunks, chunk_ids)
                    update_timer()
                    if success:
                        st.write("✅ Successfully added to knowledge base.")
                        st.session_state["rag_last_processed_source_id"] = source_id
                        
                        # Final timer update with status
                        elapsed = time.time() - start_time
                        total_time_str = str(datetime.timedelta(seconds=int(elapsed)))
                        timer_placeholder.markdown(f"""
                        <div style="background-color: #043927; padding: 10px; border-radius: 5px; text-align: center;">
                            ✅ <span style="font-size: 1.1em; font-weight: bold;">Done in {total_time_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("❌ Failed to add chunks to Pinecone.")
                        elapsed = time.time() - start_time
                        total_time_str = str(datetime.timedelta(seconds=int(elapsed)))
                        timer_placeholder.markdown(f"""
                        <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                            ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {total_time_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("❌ Failed to initialize Pinecone connection. Cannot add data.")
                    status.update(label="Error connecting to vector store!", state="error")
                    elapsed = time.time() - start_time
                    timer_placeholder.markdown(f"""
                    <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                        ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {str(datetime.timedelta(seconds=int(elapsed)))}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
            else:
                st.write("❌ No chunks created from the content.")
                elapsed = time.time() - start_time
                timer_placeholder.markdown(f"""
                <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                    ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {str(datetime.timedelta(seconds=int(elapsed)))}</span>
                </div>
                """, unsafe_allow_html=True)

            # Mark processing complete only if no Pinecone connection error occurred
            if vector_store: 
                status.update(label="Processing Complete!", state="complete")

# Handle cases where input was given but processing did not occur
elif uploaded_file or url:
    # Check if it was a duplicate within this session
    if source_id and last_processed_id and source_id == last_processed_id:
         st.sidebar.warning(
             f"Content from source '{source_id[:50]}...' was already processed in this session. The knowledge base is up-to-date with this source.",
             icon="♻️"
         )
    # Content extraction failed
    elif content and "Error" in content:
        st.sidebar.error(f"Failed to extract content: {content}")

# Chat Interface
st.write("---")
st.header("Chat with the Knowledge Base 🤖")

# Add source type filtering
with st.expander("Advanced Settings", expanded=False):
    st.subheader("Knowledge Source Filters")
    cols = st.columns(4)  # Expanded from 3 to 4 columns
    with cols[0]:
        include_pdf = st.checkbox("PDF Documents", value=True)
    with cols[1]:
        include_url = st.checkbox("Web Articles", value=True)
    with cols[2]:
        include_youtube = st.checkbox("YouTube Transcripts", value=True)
    with cols[3]:
        include_ocr = st.checkbox("OCR Processed Content", value=True)
    
    # Build filter based on selections
    source_filters = []
    if include_pdf:
        source_filters.append("pdf")
    if include_url:
        source_filters.append("url")
    if include_youtube:
        source_filters.append("youtube")
    if include_ocr:
        source_filters.append("ocr")
    
    if not source_filters:
        st.warning("Please select at least one knowledge source type", icon="⚠️")
        source_filters = ["pdf", "url", "youtube", "ocr"]  # Default to all if none selected

# Display conversation history
for message in st.session_state["rag_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the content in the knowledge base:"):
    # Add user message to state and display
    st.session_state["rag_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get the Pinecone vector store from session state
    vector_store = st.session_state.get("rag_vector_store")
    retriever = None
    
    if vector_store:
        # Create retriever with metadata filter for selected source types
        retriever = vector_store.as_retriever(
            search_kwargs={
                "filter": {"source_type": {"$in": source_filters}},
                "k": 3  # Reduced from 5 to 3 for faster retrieval
            }
        )
        
        if not st.session_state["rag_vector_store_initialized"]:
            st.warning("Error connecting to Pinecone. Please check API keys and index name.", icon="⚠️")
            st.stop()
    else:
        st.error("Failed to establish connection with Pinecone Vector Store. Cannot proceed.", icon="🚨")
        st.stop()

    # Get relevant documents 
    # Process the query for better retrieval
    processed_query = preprocess_query(prompt)
    print(f"Original query: '{prompt}'")
    print(f"Processed query: '{processed_query}'")
    
    # Get relevant documents without caching
    start_time = time.time()
    context_docs = retriever.invoke(processed_query) if retriever else []
    print(f"Context retrieval took {time.time() - start_time:.2f} seconds")
    
    # Format context with source information for better citation
    if context_docs:
        formatted_context = ""
        for i, doc in enumerate(context_docs):
            # Extract source metadata
            metadata = doc.metadata
            source_type = metadata.get("source_type", "unknown")
            display_name = metadata.get("display_name", "Unknown source")
            
            # Get additional details based on source type
            source_details = ""
            if source_type == "url":
                domain = metadata.get("domain", "")
                url = metadata.get("url", "")
                source_details = f"DOMAIN: {domain}\nURL: {url}"
            elif source_type == "youtube":
                video_id = metadata.get("source_id", "").replace("youtube_", "")
                url = metadata.get("url", "")
                source_details = f"VIDEO_ID: {video_id}\nURL: {url}"
            elif source_type == "pdf":
                doc_id = metadata.get("source_id", "")
                source_details = f"DOC_ID: {doc_id}"
            
            # Format each document chunk with structured source info
            formatted_context += f"\n--- BEGIN EXTRACT #{i+1} ---\n"
            formatted_context += doc.page_content
            formatted_context += f"\n--- END EXTRACT #{i+1} ---\n"
            formatted_context += f"SOURCE INFO FOR EXTRACT #{i+1}:\n"
            formatted_context += f"TYPE: {source_type}\n"
            formatted_context += f"NAME: {display_name}\n"
            formatted_context += source_details + "\n"
            
        context = formatted_context
    else:
        context = "No relevant information found in the knowledge base."
        
    # Prepare for multi-model response
    responses = {}
    assistant_response_display = ""

    with st.chat_message("assistant"):
        # Create placeholders for each model response
        model_placeholders = {}
        for model in selected_models:
            model_placeholders[model] = st.empty()
        
        initial_placeholder = st.empty()
        
        # Only use the fast model if we have multiple models (otherwise it's wasted work)
        if len(selected_models) > 1:
            # First generate a quick response with a fast model
            fast_chain = get_fast_chain(temperature)
            if fast_chain:
                try:
                    # Start with a thinking message
                    initial_placeholder.markdown("*Generating response...*")
                    
                    # Generate quick response with truncated context
                    # Use a subset of the context for faster response
                    truncated_context = context[:min(len(context), 4000)]  # Limit context size for faster response
                    response = ask_question(fast_chain, prompt, truncated_context)[0]
                    
                    # Show the quick response (but don't add it to the final message)
                    initial_placeholder.markdown(f"{response}")
                except Exception as e:
                    print(f"Error generating fast response: {e}")
                    initial_placeholder.markdown("*Generating detailed response...*")
        else:
            # For single model, just show a thinking message
            initial_placeholder.markdown("*Generating response...*")
        
        # Generate full responses with selected models in parallel
        for model in selected_models:
            model_placeholders[model] = st.empty()
            model_placeholders[model].markdown(f"*Processing with {model}...*")
            
        # Define a function to process each model concurrently
        def process_model(model_name, question, context, conversation_history):
            try:
                chain = get_chain(model_name, temperature)
                if not chain:
                    return model_name, f"Error: Could not initialize model {model_name}", []
                response, updated_history = ask_question(chain, question, context, conversation_history)
                return model_name, response, updated_history
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                return model_name, f"Error: {str(e)}", []
        
        # Process models concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Get current conversation history to pass to each process_model call
            current_history = st.session_state.get("rag_conversation_history", "")
            
            # Pass conversation_history to each process_model call
            future_to_model = {
                executor.submit(process_model, model, prompt, context, current_history): model 
                for model in selected_models
            }
            
            latest_history = current_history
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    model_name, response, updated_history = future.result()
                    responses[model] = response
                    
                    # If only one model is selected, replace the initial "Generating" message
                    # Otherwise show all model responses labeled
                    if len(selected_models) == 1:
                        # For single model, directly update the initial placeholder
                        initial_placeholder.markdown(f"{response}")
                        # Clear the "Processing" message
                        model_placeholders[model].empty()
                        assistant_response_display = response
                    else:
                        model_placeholders[model].markdown(f"**Response from {model}:**\n{response}")
                        assistant_response_display += f"**{model}**: {response}\n\n"
                    
                    # Save the most recent history update
                    latest_history = updated_history
                except Exception as e:
                    print(f"Exception processing result from {model}: {e}")
                    model_placeholders[model].markdown(f"**Response from {model}:**\nError generating response.")
                    assistant_response_display += f"**{model}**: Error generating response.\n\n"

    st.session_state["rag_messages"].append({"role": "assistant", "content": assistant_response_display.strip()})
    st.session_state["rag_conversation_history"] = latest_history