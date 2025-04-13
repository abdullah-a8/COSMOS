import streamlit as st
from core.data_extraction import extract_text_from_pdf, extract_text_from_url
from core.processing import process_content
from core.vector_store import get_pinecone_vector_store, add_chunks_to_vector_store
from core.chain import get_chain, ask_question
import config.settings as settings

from streamlit import cache_data, cache_resource
import time
import datetime

# Streamlit App Configuration
st.set_page_config(page_title="Multi-Model RAG Chatbot 📄🔍", page_icon="🧬", layout="wide")
st.title("Multi-Model RAG-Powered Article Chatbot 📄🔍")

# Sidebar Settings
st.sidebar.title("Settings")

# Model selection options with performance characteristics
models = {
    # DeepSeek Models
    "deepseek-r1-distill-llama-70b": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": None,  # Unlimited token capacity
        "advantages": "Highly optimized for low latency with no token limits, making it ideal for large-scale deployments.",
        "disadvantages": "Limited daily requests compared to other models.",
    },
    # Alibaba Cloud Qwen Models
    "qwen-2.5-32b": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 10_000,
        "tokens_per_day": 500_000,
        "advantages": "Powerful 32B model optimized for long-context comprehension and reasoning.",
        "disadvantages": "Requires more computational resources.",
    },
    # Google's Gemma Model
    "gemma2-9b-it": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 15_000,
        "tokens_per_day": 500_000,
        "advantages": "Higher token throughput, suitable for large-scale, fast inference.",
        "disadvantages": "Limited versatility compared to larger LLaMA3 models.",
    },
    # Meta's LLaMA 3 Models
    "llama-3.1-8b-instant": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 20_000,
        "tokens_per_day": 500_000,
        "advantages": "High-speed processing with large token capacity, great for real-time applications.",
        "disadvantages": "Less accurate for complex reasoning tasks compared to larger models.",
    },
    "llama-3.3-70b-versatile": {
        "requests_per_minute": 30,
        "requests_per_day": 1_000,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 100_000,
        "advantages": "Versatile model optimized for high accuracy in diverse scenarios.",
        "disadvantages": "Lower throughput compared to some smaller models.",
    },
    "llama3-70b-8192": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 6_000,
        "tokens_per_day": 500_000,
        "advantages": "Long-context capabilities, ideal for handling detailed research papers and articles.",
        "disadvantages": "Moderate speed and accuracy for shorter tasks.",
    },
    "llama3-8b-8192": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 20_000,
        "tokens_per_day": 500_000,
        "advantages": "Supports high-speed inference with long-context support.",
        "disadvantages": "Slightly less accurate for complex reasoning compared to larger models.",
    },
    # Mistral AI
    "mistral-saba-24b": {
        "requests_per_minute": 30,
        "requests_per_day": 7_000,
        "tokens_per_minute": 7_000,
        "tokens_per_day": 250_000,
        "advantages": "Strong multi-turn conversation capabilities and effective retrieval augmentation.",
        "disadvantages": "Limited token capacity compared to LLaMA-70B.",
    },
    "mixtral-8x7b-32768": {
        "requests_per_minute": 30,
        "requests_per_day": 14_400,
        "tokens_per_minute": 5_000,
        "tokens_per_day": 500_000,
        "advantages": "Supports long document processing for better contextual understanding.",
        "disadvantages": "Lower token throughput compared to some other models.",
    },
}

# Allow users to select multiple models
selected_models = st.sidebar.multiselect(
    "Choose Models",
    options=list(models.keys()),
    default=["llama3-70b-8192"],  # Default model selection
)

# Fallback default if no model is selected
if not selected_models:
    selected_models = ["llama3-70b-8192"]

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

# Reset App Function
def reset_app():
    """Set a reset flag to True and trigger app re-run."""
    st.session_state["reset"] = True
    st.rerun()

# Add Reset App button in the sidebar
st.sidebar.button("Reset Chat & Clear Upload", on_click=reset_app)

# Check if reset is flagged
if st.session_state.get("reset", False):
    # Clear all session state variables and cache
    keys_to_keep = []
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["reset"] = False
    st.rerun()

# Content Upload Section
st.sidebar.title("Add Content to Knowledge Base")
input_method = st.sidebar.radio("Input Method", ["PDF File", "URL"])

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

# Process uploaded inputs
if input_method == "PDF File":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF File", key="pdf_uploader")
    if uploaded_file:
        content, source_id = cached_extract_text(input_method, uploaded_file_key=uploaded_file, url=None)
elif input_method == "URL":
    url = st.sidebar.text_input("Enter a News URL", key="url_input")
    if url:
        content, source_id = cached_extract_text(input_method, uploaded_file_key=None, url=url)

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

                # Get vector store connection
                vector_store = cached_get_pinecone_vector_store()
                
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
    cols = st.columns(3)
    with cols[0]:
        include_pdf = st.checkbox("PDF Documents", value=True)
    with cols[1]:
        include_url = st.checkbox("Web Articles", value=True)
    with cols[2]:
        include_youtube = st.checkbox("YouTube Transcripts", value=True)
    
    # Build filter based on selections
    source_filters = []
    if include_pdf:
        source_filters.append("pdf")
    if include_url:
        source_filters.append("url")
    if include_youtube:
        source_filters.append("youtube")
    
    if not source_filters:
        st.warning("Please select at least one knowledge source type", icon="⚠️")
        source_filters = ["pdf", "url", "youtube"]  # Default to all if none selected

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

    # Get the Pinecone vector store with filter for source types
    vector_store = cached_get_pinecone_vector_store()
    retriever = None
    
    if vector_store:
        # Create retriever with metadata filter for selected source types
        retriever = vector_store.as_retriever(
            search_kwargs={
                "filter": {"source_type": {"$in": source_filters}},
                "k": 5  # Retrieve top 5 chunks
            }
        )
        
        if not st.session_state["rag_vector_store_initialized"]:
            st.warning("Error connecting to Pinecone. Please check API keys and index name.", icon="⚠️")
            st.stop()
    else:
        st.error("Failed to establish connection with Pinecone Vector Store. Cannot proceed.", icon="🚨")
        st.stop()

    # Get relevant documents
    context_docs = retriever.invoke(prompt) if retriever else []
    
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
        model_placeholders = {}
        for model in selected_models:
            model_placeholders[model] = st.empty()
            model_placeholders[model].markdown(f"*Thinking with {model}...*")

        for model in selected_models:
            chain = get_chain(model, temperature)
            response, updated_history = ask_question(chain, prompt, context, st.session_state["rag_conversation_history"])
            responses[model] = response
            model_placeholders[model].markdown(f"**Response from {model}:**\n{response}")
            assistant_response_display += f"**{model}**: {response}\n\n"

    st.session_state["rag_messages"].append({"role": "assistant", "content": assistant_response_display.strip()})
    st.session_state["rag_conversation_history"] = updated_history