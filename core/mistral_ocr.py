import os
import base64
import hashlib
from typing import List,Tuple,Union
from mistralai import Mistral
import fitz
import config.settings as settings

# Try to import C++ accelerated components if available
try:
    from core.cpp_modules import hash_generator
    USE_CPP_HASH = True
except ImportError:
    USE_CPP_HASH = False

def get_mistral_client() -> Mistral:
    """
    Get or create a Mistral client instance with the API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not found")
    
    return Mistral(api_key=api_key)

def encode_image_to_base64(image_path_or_bytes: Union[str, bytes]) -> str:
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image_path_or_bytes: Path to image file or image bytes
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image_path_or_bytes, str):
        with open(image_path_or_bytes, "rb") as image_file:
            image_bytes = image_file.read()
    else:
        image_bytes = image_path_or_bytes
    
    return base64.b64encode(image_bytes).decode("utf-8")

def process_image_with_ocr(image_path_or_bytes: Union[str, bytes]) -> Tuple[str, str]:
    """
    Process an image with Mistral OCR and return extracted text and image hash.
    
    Args:
        image_path_or_bytes: Path to image file or image bytes
        
    Returns:
        Tuple of (extracted_text, image_hash)
    """
    try:
        # Get image bytes
        if isinstance(image_path_or_bytes, str):
            with open(image_path_or_bytes, "rb") as image_file:
                image_bytes = image_file.read()
        else:
            image_bytes = image_path_or_bytes
        
        # Hash the image
        if USE_CPP_HASH:
            image_hash = hash_generator.compute_sha256(image_bytes)
        else:
            image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_bytes)
        
        # Get Mistral client
        client = get_mistral_client()
        
        # First try with the latest SDK approach
        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            )
            
            # Extract plain text from response
            extracted_text = ocr_response.pages[0].markdown if ocr_response.pages else ""
            
            if not extracted_text:
                extracted_text = "No text extracted from image."
            
            # Prefix hash with 'ocr_' to identify it as OCR-processed content
            ocr_source_id = f"ocr_{image_hash}"
            
            return extracted_text, ocr_source_id
            
        except Exception as sdk_error:
            # Fall back to REST API approach
            try:
                api_key = os.getenv("MISTRAL_API_KEY")
                if not api_key:
                    raise ValueError("MISTRAL_API_KEY environment variable not found")
                
                # Direct API call to Mistral OCR endpoint
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Using the correct payload structure for the latest OCR API
                payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                
                # Get OCR API endpoint from settings
                ocr_endpoint = getattr(settings, "MISTRAL_OCR_API_ENDPOINT", "https://api.mistral.ai/v1/ocr")
                
                import requests
                response = requests.post(
                    ocr_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60  # Set a reasonable timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"OCR API Error: {response.status_code} - {response.text}"
                    return f"Error processing image with OCR: {error_msg}", None
                
                # Extract text from response
                result = response.json()
                
                # The latest OCR API returns structured data with pages
                if "pages" in result and result["pages"]:
                    extracted_text = result["pages"][0].get("markdown", "")
                else:
                    extracted_text = result.get("text", "")
                
                if not extracted_text:
                    extracted_text = "No text extracted from image."
                
                # Prefix hash with 'ocr_' to identify it as OCR-processed content
                ocr_source_id = f"ocr_{image_hash}"
                
                return extracted_text, ocr_source_id
                
            except Exception as fallback_error:
                fallback_error_msg = str(fallback_error)
                return f"Error processing image with OCR: Original error: {str(sdk_error)}, Fallback error: {fallback_error_msg}", None
                
    except Exception as e:
        error_msg = str(e)
        return f"Error processing image with OCR: {error_msg}", None

def extract_images_from_pdf(pdf_bytes: bytes) -> List[bytes]:
    """
    Extract all images from a PDF file.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of image byte arrays
    """
    try:
        # Open PDF document
        pdf_document = fitz.open("pdf", pdf_bytes)
        
        # Extract images
        images = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
                
        pdf_document.close()
        return images
    
    except Exception as e:
        return []

def process_pdf_with_ocr(pdf_path_or_bytes: Union[str, bytes]) -> Tuple[str, str]:
    """
    Process a PDF with Mistral OCR and return extracted text and document hash.
    This uses the Mistral OCR API directly instead of extracting and processing images separately.
    
    Args:
        pdf_path_or_bytes: Path to PDF file or PDF bytes
        
    Returns:
        Tuple of (extracted_text, document_hash)
    """
    try:
        # Get PDF bytes
        if isinstance(pdf_path_or_bytes, str):
            with open(pdf_path_or_bytes, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
        else:
            pdf_bytes = pdf_path_or_bytes
        
        # Hash the PDF bytes
        if USE_CPP_HASH:
            pdf_hash = hash_generator.compute_sha256(pdf_bytes)
        else:
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        
        # Encode PDF to base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Get Mistral client
        client = get_mistral_client()
        
        # Try with SDK approach first
        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                }
            )
            
            # Extract markdown text from all pages
            all_pages_text = []
            for page in ocr_response.pages:
                if page.markdown:
                    all_pages_text.append(page.markdown)
            
            extracted_text = "\n\n".join(all_pages_text)
            
            if not extracted_text:
                extracted_text = "No text extracted from PDF."
            
            # Prefix hash with 'ocr_pdf_' to identify it as OCR-processed PDF content
            ocr_source_id = f"ocr_pdf_{pdf_hash}"
            
            return extracted_text, ocr_source_id
            
        except Exception as sdk_error:
            # Fall back to REST API approach
            try:
                api_key = os.getenv("MISTRAL_API_KEY")
                if not api_key:
                    raise ValueError("MISTRAL_API_KEY environment variable not found")
                
                # Direct API call to Mistral OCR endpoint
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Use the correct payload structure for PDFs with the latest OCR API
                payload = {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    }
                }
                
                # Get OCR API endpoint from settings
                ocr_endpoint = getattr(settings, "MISTRAL_OCR_API_ENDPOINT", "https://api.mistral.ai/v1/ocr")
                
                import requests
                response = requests.post(
                    ocr_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120  # Longer timeout for PDFs
                )
                
                if response.status_code != 200:
                    error_msg = f"OCR API Error for PDF: {response.status_code} - {response.text}"
                    return f"Error processing PDF with OCR: {error_msg}", None
                
                # Extract text from response
                result = response.json()
                
                # The latest OCR API returns structured data with pages
                all_pages_text = []
                if "pages" in result and result["pages"]:
                    for page in result["pages"]:
                        if "markdown" in page:
                            all_pages_text.append(page["markdown"])
                
                extracted_text = "\n\n".join(all_pages_text)
                
                if not extracted_text:
                    extracted_text = "No text extracted from PDF."
                
                # Prefix hash with 'ocr_pdf_' to identify it as OCR-processed PDF content
                ocr_source_id = f"ocr_pdf_{pdf_hash}"
                
                return extracted_text, ocr_source_id
                
            except Exception as fallback_error:
                fallback_error_msg = str(fallback_error)
                return f"Error processing PDF with OCR: Original error: {str(sdk_error)}, Fallback error: {fallback_error_msg}", None
    
    except Exception as e:
        error_msg = str(e)
        return f"Error processing PDF with OCR: {error_msg}", None

def process_file_with_ocr(file) -> Tuple[str, str]:
    """
    Process a file (PDF or image) with OCR based on file type.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (extracted_content, file_hash)
    """
    try:
        file_content = file.read()
        file_extension = file.name.split('.')[-1].lower()
        
        # Reset file pointer for further operations
        file.seek(0)
        
        # Check if API key is available
        if not os.getenv("MISTRAL_API_KEY"):
            return "Error: MISTRAL_API_KEY environment variable not found", None
            
        if file_extension in ('pdf'):
            return process_pdf_with_ocr(file_content)
        elif file_extension in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'):
            return process_image_with_ocr(file_content)
        else:
            return f"Error: Unsupported file type: {file_extension}", None
            
    except Exception as e:
        error_msg = str(e)
        return f"Error processing file with OCR: {error_msg}", None 