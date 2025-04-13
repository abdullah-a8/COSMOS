import streamlit as st
import time
import datetime
from core.data_extraction import extract_transcript_details
from core.processing import process_content
from core.vector_store import get_pinecone_vector_store, add_chunks_to_vector_store
import config.settings as settings

from streamlit import cache_data, cache_resource

# Streamlit App Configuration
st.set_page_config(page_title="YouTube RAG Processor 🎥", page_icon="▶️", layout="wide")
st.title("YouTube Transcript to Knowledge Base Processor 🎥")
st.write("Extract, process, and add YouTube video transcripts to your knowledge base for later retrieval.")

# Sidebar Settings
st.sidebar.title("YouTube Settings")

# Chunk size slider using default from settings
st.sidebar.markdown(
    """
    ### Chunk Size:
    Defines the number of words in each content chunk. 
    - **Small Chunks (e.g., 100–300):** Ideal for highly specific queries or short videos.  
    - **Large Chunks (e.g., 1000–3000):** Better for summarization or broader context but may lose finer details.
    """
)
chunk_size = st.sidebar.slider(
    label="Chunk Size (Words)",
    min_value=100,
    max_value=3000,
    value=settings.DEFAULT_CHUNK_SIZE,
    step=50,
    help="Set the number of words in each transcript chunk. Choose smaller values for better specificity or larger for broader context."
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
    label="Chunk Overlap (Words)",
    min_value=10,
    max_value=300,
    value=settings.DEFAULT_CHUNK_OVERLAP,
    step=10,
    help="Control the overlap of consecutive content chunks. Larger values improve context but may slow down processing."
)

# Session State Initialization
if "youtube_last_processed_id" not in st.session_state:
    st.session_state["youtube_last_processed_id"] = None

# Reset button
def reset_youtube_processor():
    st.session_state["youtube_last_processed_id"] = None
    st.rerun()

st.sidebar.button("Reset YouTube Processor", on_click=reset_youtube_processor)

# Cache functions
@cache_data
def cached_extract_transcript(youtube_url):
    """Cache transcript extraction to avoid repeated API calls"""
    return extract_transcript_details(youtube_url)

@cache_data
def cached_process_youtube_content(video_id, content, _chunk_size, _chunk_overlap):
    """Cache content processing for YouTube transcripts"""
    print(f"Processing YouTube content for video ID: {video_id}")
    youtube_source_id = f"youtube_{video_id}"
    return process_content(content, _chunk_size, _chunk_overlap, youtube_source_id)

# Main UI for YouTube URL Input
youtube_url = st.text_input(
    "Enter YouTube Video URL:", 
    placeholder="https://www.youtube.com/watch?v=..."
)

# Process the URL when provided
if youtube_url:
    # Extract video_id for thumbnail
    video_id = None
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    
    # Display thumbnail if video_id was successfully extracted
    if video_id:
        thumbnail_container = st.container()
        with thumbnail_container:
            cols = st.columns([1, 2, 1])  # Create 3 columns with the middle one being wider
            with cols[1]:  # Use the middle column for the image
                st.image(
                    f"http://img.youtube.com/vi/{video_id}/0.jpg",
                    width=480,  # Standard YouTube thumbnail width
                    caption=f"Video ID: {video_id}"
                )
    
    # Check if this URL was already processed in this session
    if video_id and video_id == st.session_state["youtube_last_processed_id"]:
        st.warning(
            f"Video ID '{video_id}' was already processed in this session. The knowledge base is up-to-date with this transcript.",
            icon="♻️"
        )
    # Process button (only show if not already processed in this session)
    elif st.button("Process YouTube Transcript"):
        # Create columns for status and timer
        status_col, timer_col = st.columns([3, 1])
        
        # Timer setup
        start_time = time.time()
        timer_placeholder = timer_col.empty()
        
        # Function to update timer
        def update_timer():
            elapsed = time.time() - start_time
            timer_placeholder.markdown(f"""
            <div style="background-color: #262730; padding: 10px; border-radius: 5px; text-align: center;">
                ⏱️ <span style="font-size: 1.1em; font-weight: bold;">{datetime.timedelta(seconds=int(elapsed))}</span>
            </div>
            """, unsafe_allow_html=True)
        
        update_timer()
        
        with status_col.status("Processing YouTube transcript...", expanded=True):
            st.write("Extracting transcript...")
            transcript, source_id = cached_extract_transcript(youtube_url)
            
            update_timer()
            
            if (transcript is not None and isinstance(transcript, str) and transcript.startswith("Error")) or not source_id:
                error_msg = transcript if (transcript and "Error" in transcript) else "Failed to get source ID."
                st.error(error_msg)
                # Show failure in timer
                elapsed = time.time() - start_time
                timer_placeholder.markdown(f"""
                <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                    ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {str(datetime.timedelta(seconds=int(elapsed)))}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Extract the base video_id from the source_id ('youtube_xxxx')
                base_video_id = source_id.replace("youtube_", "")
                
                st.write(f"✅ Successfully extracted transcript ({len(transcript.split())} words)")
                st.write("Splitting into chunks...")
                
                update_timer()
                
                chunks, chunk_ids = cached_process_youtube_content(base_video_id, transcript, chunk_size, chunk_overlap)
                
                update_timer()
                
                if chunks:
                    st.write(f"Created {len(chunks)} chunks with YouTube-specific IDs")
                    st.write("Connecting to knowledge base...")
                    
                    update_timer()
                    
                    vector_store = get_pinecone_vector_store()
                    
                    if vector_store:
                        st.write("Adding chunks to knowledge base...")
                        success = add_chunks_to_vector_store(vector_store, chunks, chunk_ids)
                        
                        # Final timer update with status
                        elapsed = time.time() - start_time
                        total_time_str = str(datetime.timedelta(seconds=int(elapsed)))
                        
                        if success:
                            st.write("✅ Successfully added YouTube transcript to knowledge base!")
                            st.session_state["youtube_last_processed_id"] = base_video_id
                            st.success(f"The transcript for YouTube video '{base_video_id}' has been added to your knowledge base. You can now chat with it using the RAG Chatbot page.")
                            # Show success in timer
                            timer_placeholder.markdown(f"""
                            <div style="background-color: #043927; padding: 10px; border-radius: 5px; text-align: center;">
                                ✅ <span style="font-size: 1.1em; font-weight: bold;">Done in {total_time_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Failed to add chunks to knowledge base.")
                            # Show failure in timer
                            timer_placeholder.markdown(f"""
                            <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                                ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {total_time_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to connect to knowledge base.")
                        # Show failure in timer
                        elapsed = time.time() - start_time
                        timer_placeholder.markdown(f"""
                        <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                            ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {str(datetime.timedelta(seconds=int(elapsed)))}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Failed to process transcript into chunks.")
                    # Show failure in timer
                    elapsed = time.time() - start_time
                    timer_placeholder.markdown(f"""
                    <div style="background-color: #5e0808; padding: 10px; border-radius: 5px; text-align: center;">
                        ❌ <span style="font-size: 1.1em; font-weight: bold;">Failed after {str(datetime.timedelta(seconds=int(elapsed)))}</span>
                    </div>
                    """, unsafe_allow_html=True)

# Information about using the processed transcript
st.markdown("""
### How to Use the Processed Transcript

After processing a YouTube video transcript:

1. Navigate to the **RAG Chatbot** page from the sidebar.
2. Start asking questions about the video content.
3. The system will retrieve relevant portions of the transcript along with any other knowledge base content.

This integration allows you to query across all your sources (PDFs, URLs, and YouTube videos) in one place.
""")