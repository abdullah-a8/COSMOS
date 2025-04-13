import time
from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Function to split content into chunks and add metadata/IDs
def process_content(content, chunk_size, chunk_overlap, source_id):
    if not source_id:  # Don't process if we don't have a source identifier
        return [], []

    # Extract domain from URL if applicable
    domain_name = None
    if "://" in source_id:
        try:
            parsed_url = urlparse(source_id)
            domain_name = parsed_url.netloc
            if domain_name.startswith('www.'):
                domain_name = domain_name[4:]
        except:
            domain_name = source_id.split("://")[1].split("/")[0]

    # Determine source type and create rich metadata
    source_metadata = {
        "source_id": source_id,
        "source_type": "unknown",
        "title": "Untitled Document",
        "ingestion_timestamp": time.time(),  # timestamp for potential sorting/filtering later
    }
    
    # Set source_type and specific metadata based on source_id pattern
    if source_id.startswith("youtube_"):
        video_id = source_id.replace("youtube_", "")
        source_metadata.update({
            "source_type": "youtube",
            "title": f"YouTube Video: {video_id}",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "citation_text": f"YouTube video ({video_id})",
            "display_name": "YouTube"
        })
    elif domain_name:  # Handle URLs where a domain was successfully extracted
        source_metadata.update({
            "source_type": "url",
            "title": source_id.split("/")[-1] if "/" in source_id else source_id,
            "url": source_id,
            "domain": domain_name,
            "citation_text": f"Web article at {source_id}",
            "display_name": domain_name
        })
    else:  # Assume PDF (or other document types) identified by hash
        short_id = source_id[:8] + "..." if len(source_id) > 8 else source_id
        source_metadata.update({
            "source_type": "pdf",
            "title": f"PDF Document (ID: {short_id})",
            "citation_text": f"PDF document ({short_id})",
            "display_name": "PDF document"
        })

    base_document = Document(page_content=content, metadata=source_metadata)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split based on the base document to ensure all chunks inherit the initial metadata
    split_chunks = text_splitter.split_documents([base_document])

    # Process each chunk: add sequence info and generate final Pinecone IDs
    chunk_ids = []
    processed_chunks = []
    
    for i, chunk in enumerate(split_chunks):
        chunk.metadata["chunk_sequence"] = i
        chunk.metadata["chunk_total"] = len(split_chunks)
        
        # Augment citation text with chunk info for more precise referencing
        if "citation_text" in chunk.metadata:
            # Keep original citation, but add sequence info to a separate field for backend use
            chunk.metadata["citation_text_full"] = f"{chunk.metadata['citation_text']} (section {i+1} of {len(split_chunks)})"
        
        # Create a unique ID for Pinecone upsert (source + chunk index)
        pinecone_id = f"{source_id}_{i}"
        chunk_ids.append(pinecone_id)
        processed_chunks.append(chunk)

    return processed_chunks, chunk_ids