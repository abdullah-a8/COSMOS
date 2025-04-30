import os
import json
import base64
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import openai
import config.prompts as prompts
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config.settings as settings

# Define constants within this module as they are only used here
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
# Make paths relative to the project root
TOKEN_FILE = 'credentials/.gmail_token.json'
CREDENTIALS_FILE = 'credentials/.gmail_credentials.json'

# --- Authentication Functions ---

def gmail_authenticate():
    """Handles the Google OAuth 2.0 flow for Gmail API access."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'r') as token:
                creds = Credentials.from_authorized_user_info(json.load(token), SCOPES)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {TOKEN_FILE}. Attempting re-authentication.")
            creds = None # Force re-authentication
        except Exception as e:
            print(f"Error loading token file {TOKEN_FILE}: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Proceeding to full authentication.")
                # Fall through to full authentication if refresh fails
        # If no valid creds, attempt full authentication flow
        if not creds or not creds.valid: # Check again in case refresh failed
            if not os.path.exists(CREDENTIALS_FILE):
                st.error(f"Credentials file not found at {CREDENTIALS_FILE}. Cannot authenticate.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                # Consider adding a timeout or better error handling for run_local_server
                creds = flow.run_local_server(port=8090) # Use a specific port
            except Exception as e:
                st.error(f"Authentication flow failed: {e}")
                return None
        
        # Save the new/refreshed credentials
        try:
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            st.warning(f"Could not save token file {TOKEN_FILE}: {e}")
            
    # Build and return the service object
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Failed to build Gmail service: {e}")
        return None

@st.cache_resource(ttl=3600) # Keep caching for the service object
def get_gmail_service():
    """Gets the authenticated Gmail service, potentially from cache."""
    return gmail_authenticate()

def save_credentials(uploaded_file):
    """Saves the uploaded credentials.json file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
        with open(CREDENTIALS_FILE, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving credentials file: {e}")
        return False

# --- Email Interaction Functions ---

def get_emails(service, max_results=10, query=None):
    """Fetches emails from the authenticated Gmail account."""
    if not service:
        st.error("Gmail service not available. Cannot fetch emails.")
        return []
    if not query:
        query = "is:unread"
    
    try:
        result = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = result.get('messages', [])
        if not messages:
            return []
        
        emails = []
        for message in messages:
            try:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
                headers = {header['name'].lower(): header['value'] for header in msg['payload'].get('headers', [])}
                body = ''
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            body_data = part.get('body', {}).get('data')
                            if body_data:
                                body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='replace')
                                break
                elif 'body' in msg['payload']:
                    body_data = msg['payload'].get('body', {}).get('data')
                    if body_data:
                        body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='replace')
                
                email = {
                    'id': message['id'],
                    'thread_id': msg['threadId'],
                    'from': headers.get('from', 'Unknown'),
                    'to': headers.get('to', 'Unknown'),
                    'subject': headers.get('subject', '(No Subject)'),
                    'date': headers.get('date', ''),
                    'body': body,
                    'labels': msg.get('labelIds', [])
                }
                emails.append(email)
            except HttpError as inner_error:
                 print(f"Error fetching details for message {message['id']}: {inner_error}") # Log non-critical error
                 continue # Skip this email
            except Exception as inner_e:
                 print(f"Unexpected error processing message {message['id']}: {inner_e}")
                 continue # Skip this email
        
        return emails
    except HttpError as error:
        st.error(f"An error occurred fetching email list: {error}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred fetching emails: {e}")
        return []

def send_email(service, to, subject, body, thread_id=None, original_message_id=None, custom_signature=None):
    """Sends an email using the authenticated Gmail service, adding threading headers if possible."""
    if not service:
        st.error("Gmail service not available. Cannot send email.")
        return None
    try:
        # Add signature (custom or default)
        default_signature = "\n\n--\nSent by COSMOS AI Assistant\nPowered by OpenAI GPT"
        signature = custom_signature if custom_signature is not None else default_signature
        body_with_signature = f"{body}{signature}"
        
        message_body = (
            f"To: {to}\r\n"
            f"Subject: {subject}\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            # Initial newline before body
            f"\r\n{body_with_signature}"
        )
        
        # --- Add Threading Headers (In-Reply-To / References) ---
        message_id_header_value = None
        if original_message_id:
            try:
                # Fetch only the Message-ID header of the original message
                original_msg_metadata = service.users().messages().get(
                    userId='me', 
                    id=original_message_id, 
                    format='metadata', 
                    metadataHeaders=['Message-ID']
                ).execute()
                
                if original_msg_metadata and 'payload' in original_msg_metadata and 'headers' in original_msg_metadata['payload']:
                    message_id_header = next((h['value'] for h in original_msg_metadata['payload']['headers'] if h['name'] == 'Message-ID'), None)
                    if message_id_header:
                         message_id_header_value = message_id_header
                         # Prepend headers to the message body *before* encoding
                         message_body = (
                             f"In-Reply-To: {message_id_header_value}\r\n"
                             f"References: {message_id_header_value}\r\n"
                             f"{message_body}" # Append original message body
                         )
                    else:
                        print(f"Warning: Message-ID header not found for message {original_message_id}")
                else:
                    print(f"Warning: Could not retrieve headers needed for threading for message {original_message_id}")
            except HttpError as header_error:
                # Non-critical error, just log and continue without headers
                print(f"Warning: HttpError fetching headers for threading on message {original_message_id}: {header_error}")
            except Exception as e:
                # Catch any other unexpected error during header processing
                print(f"Warning: Unexpected error fetching headers for threading: {e}")
        # --- End Threading Headers ---

        message = {
            'raw': base64.urlsafe_b64encode(message_body.encode('utf-8')).decode('utf-8')
        }
        
        # Use thread_id if provided (ensures it stays in the same Gmail thread visually)
        if thread_id:
            message['threadId'] = thread_id
        
        sent_message = service.users().messages().send(userId='me', body=message).execute()
        print(f"Email sent. Added threading headers: {bool(message_id_header_value)}")
        return sent_message
        
    except HttpError as error:
        st.error(f"An error occurred while sending the email: {error}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred sending email: {e}")
        return None

def modify_email_labels(service, message_id, labels_to_add=None, labels_to_remove=None):
    """Adds or removes labels from a specific email message."""
    if not service:
        st.error("Gmail service not available. Cannot modify labels.")
        return False
    try:
        modify_body = {}
        if labels_to_add:
            modify_body['addLabelIds'] = labels_to_add
        if labels_to_remove:
            modify_body['removeLabelIds'] = labels_to_remove
            
        if not modify_body:
            return True # Nothing to do
            
        service.users().messages().modify(
            userId='me', 
            id=message_id, 
            body=modify_body
        ).execute()
        print(f"Successfully modified labels for message {message_id}. Added: {labels_to_add}, Removed: {labels_to_remove}")
        return True
    except HttpError as error:
        st.warning(f"Could not modify labels for message {message_id}: {error}")
        return False
    except Exception as e:
        st.warning(f"Unexpected error modifying labels for message {message_id}: {e}")
        return False

# --- OpenAI Interaction Functions ---

def classify_email(email_body, email_subject, use_fallback=True, fallback_model=None):
    """Classifies email content using OpenAI with LangChain fallback."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Try OpenAI first
    if openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            prompt_content = prompts.GMAIL_CLASSIFY_PROMPT.format(
                email_subject=email_subject,
                email_body=email_body[:1000] # Trim for safety
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=15,
                temperature=0.3
            )
            category = response.choices[0].message.content.strip()
            return category
        except Exception as e:
            print(f"OpenAI error: {str(e)}. Attempting fallback if enabled.")
            if not use_fallback:
                st.error(f"Error classifying email with OpenAI: {str(e)}")
                return "Unknown"
    
    # Use LangChain fallback if OpenAI fails or no API key
    if use_fallback:
        return classify_with_langchain(
            email_body, 
            email_subject, 
            prompts.GMAIL_CLASSIFY_PROMPT,
            model_name=fallback_model
        )
    else:
        st.error("OpenAI API key not found and fallback not enabled")
        return "Unknown"

def generate_reply(email_body, email_subject, sender_name, tone, style, length, 
                  user_context="N/A", use_fallback=True, fallback_model=None):
    """Generates an email reply using OpenAI with LangChain fallback."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Try OpenAI first
    if openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Ensure user_context has a default value if empty or None
            if not user_context:
                user_context = "N/A"
                
            prompt_content = prompts.GMAIL_GENERATE_REPLY_PROMPT.format(
                sender_name=sender_name,
                subject=email_subject,
                email_body=email_body[:1500],
                tone=tone,
                style=style,
                length=length,
                user_context=user_context
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=1000,
                temperature=0.7
            )
            generated_reply = response.choices[0].message.content.strip()
            return generated_reply
        except Exception as e:
            print(f"OpenAI error: {str(e)}. Attempting fallback if enabled.")
            if not use_fallback:
                st.error(f"Error generating reply with OpenAI: {str(e)}")
                return ""
    
    # Use LangChain fallback if OpenAI fails or no API key
    if use_fallback:
        return generate_reply_with_langchain(
            email_body,
            email_subject,
            sender_name,
            tone,
            style,
            length,
            prompts.GMAIL_GENERATE_REPLY_PROMPT,
            model_name=fallback_model,
            user_context=user_context
        )
    else:
        st.error("OpenAI API key not found and fallback not enabled")
        return ""

def summarize_email(email_body, recipient_info=None, use_fallback=True, fallback_model=None):
    """Summarizes email content using OpenAI with LangChain fallback."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Set default recipient info if not provided
    if not recipient_info:
        recipient_info = {
            'name': 'User',
            'email': 'you@example.com'
        }
    
    # Try OpenAI first
    if openai_api_key:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            prompt_content = prompts.GMAIL_SUMMARIZE_PROMPT.format(
                email_content=email_body[:4000],
                recipient_name=recipient_info.get('name', 'User'),
                recipient_email=recipient_info.get('email', 'you@example.com')
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=250,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"OpenAI error: {str(e)}. Attempting fallback if enabled.")
            if not use_fallback:
                st.error(f"Error summarizing email with OpenAI: {str(e)}")
                return f"Error during summarization. Please check API key and connection."
    
    # Use LangChain fallback if OpenAI fails or no API key
    if use_fallback:
        # Create a properly formatted prompt with recipient info
        formatted_prompt = prompts.GMAIL_SUMMARIZE_PROMPT.format(
            email_content="{email_content}",
            recipient_name=recipient_info.get('name', 'User'),
            recipient_email=recipient_info.get('email', 'you@example.com')
        )
        
        return summarize_with_langchain(
            email_body,
            formatted_prompt,
            model_name=fallback_model
        )
    else:
        st.error("OpenAI API key not found and fallback not enabled")
        return f"Error during summarization. Please check API key and connection."

# --- LangChain Fallback Functions ---

def get_langchain_model(model_name=None, temperature=None):
    """Creates a LangChain model with the specified model and temperature."""
    # Use provided values or fall back to defaults from settings
    effective_model_name = model_name if model_name is not None else settings.DEFAULT_MODEL_NAME
    effective_temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
    
    if not settings.GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in settings.")
        return None
        
    try:
        model = ChatGroq(
            temperature=effective_temperature,
            groq_api_key=settings.GROQ_API_KEY,
            model_name=effective_model_name,
        )
        return model
    except Exception as e:
        print(f"Error initializing ChatGroq: {e}")
        return None

def summarize_with_langchain(email_body, prompt_template, model_name=None, temperature=None):
    """Summarizes email content using LangChain as a fallback."""
    model = get_langchain_model(model_name, temperature)
    if not model:
        return "Error: Could not initialize LangChain fallback model."
        
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | parser
    
    try:
        summary = chain.invoke({"email_content": email_body[:4000]})
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing with LangChain: {e}")
        return f"Error during LangChain summarization: {str(e)}"

def classify_with_langchain(email_body, email_subject, prompt_template, model_name=None, temperature=None):
    """Classifies email content using LangChain as a fallback."""
    model = get_langchain_model(model_name, temperature)
    if not model:
        return "Unknown"
        
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | parser
    
    try:
        category = chain.invoke({
            "email_subject": email_subject,
            "email_body": email_body[:1000]
        })
        return category.strip()
    except Exception as e:
        print(f"Error classifying with LangChain: {e}")
        return "Unknown"
        
def generate_reply_with_langchain(email_body, email_subject, sender_name, tone, style, length, 
                                  prompt_template, model_name=None, temperature=None, user_context="N/A"):
    """Generates an email reply using LangChain as a fallback."""
    model = get_langchain_model(model_name, temperature)
    if not model:
        return "Error: Could not initialize LangChain fallback model."
        
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | parser
    
    try:
        # Ensure user_context has a default value if empty or None
        if not user_context:
            user_context = "N/A"
            
        reply = chain.invoke({
            "sender_name": sender_name,
            "subject": email_subject,
            "email_body": email_body[:1500],
            "tone": tone,
            "style": style,
            "length": length,
            "user_context": user_context
        })
        return reply.strip()
    except Exception as e:
        print(f"Error generating reply with LangChain: {e}")
        return f"Error during LangChain reply generation: {str(e)}" 