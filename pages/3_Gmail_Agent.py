import streamlit as st
import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    # Insert at the beginning of the path list
    sys.path.insert(0, project_root)

# --- Authentication Check --- 
# Ensure the user is logged in, otherwise stop execution
if not st.session_state.get('authentication_status'):
    st.warning("Please log in to access this page.")
    st.stop() # Stop execution if not authenticated

# --- Original Page Code Starts Here --- 
import time
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
from core.agents.gmail_logic import (
    gmail_authenticate,
    get_gmail_service,
    save_credentials,
    get_emails,
    classify_email,
    generate_reply,
    send_email,
    summarize_email,
    modify_email_labels,
    TOKEN_FILE
)
load_dotenv()

# App configuration
st.set_page_config(
    page_title="Gmail Response Assistant",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Gmail Response Assistant")
st.write("Automate email responses with customized AI-generated replies")

# Sidebar authentication
with st.sidebar:
    st.header("Authentication")
    
    # Check for credentials file
    CREDENTIALS_FILE_UI_CHECK = 'credentials/.gmail_credentials.json'
    if not os.path.exists(CREDENTIALS_FILE_UI_CHECK):
        st.error("Gmail API credentials file not found. Please upload your credentials.json file.")
        uploaded_file = st.file_uploader("Upload your credentials.json file", type=['json'])
        if uploaded_file is not None:
            if save_credentials(uploaded_file):
                st.success("Credentials file uploaded successfully!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Failed to save credentials file.")
    
    # Authentication status
    service = get_gmail_service()
    if service:
        st.success("✅ Connected to Gmail")
        
        # Logout option
        if st.button("Logout from Gmail"):
            if os.path.exists(TOKEN_FILE):
                try:
                    os.remove(TOKEN_FILE)
                    st.cache_resource.clear()
                    st.success("Logged out successfully!")
                    st.rerun()
                except OSError as e:
                    st.error(f"Error removing token file: {e}")
            else:
                 st.info("Already logged out (token file not found).")
                 st.cache_resource.clear()
                 st.rerun()
    else:
        st.warning("⚠️ Not connected to Gmail")
        st.info("Please authenticate to access your Gmail account")
        
        if st.button("Connect to Gmail"):
            with st.spinner("Attempting to authenticate..."):
                st.cache_resource.clear()
                new_service = get_gmail_service()
                if new_service:
                    st.success("Authentication successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Authentication failed. Check terminal/logs if needed.")
    
    st.markdown("---")
    
    # Email search settings
    st.header("Email Settings")
    email_query = st.text_input("Search Query", value="is:unread", 
                               help="Gmail search query (e.g., is:unread, from:example@gmail.com)")
    max_results = st.slider("Max Emails to Load", min_value=1, max_value=50, value=10)
    
    # Add signature configuration
    st.markdown("---")
    st.header("Signature Settings")
    default_signature = "\n--\nSent by COSMOS AI Assistant"
    
    if 'custom_signature' not in st.session_state:
        st.session_state['custom_signature'] = default_signature
        
    custom_signature = st.text_area(
        "Email Signature", 
        value=st.session_state['custom_signature'],
        help="Customize your email signature. Leave blank to use default signature.",
        height=100
    )
    
    if custom_signature != st.session_state['custom_signature']:
        st.session_state['custom_signature'] = custom_signature
    
    if st.button("Reset to Default"):
        st.session_state['custom_signature'] = default_signature
        st.rerun()
    
    st.markdown("---")
    
    # Add model settings for fallback
    st.header("Model Settings")
    
    # LLM models available for fallback - Updated with latest Groq models
    models = {
        # Latest Meta Models
        "llama-3.1-8b-instant": "Fast & efficient, best for simple email tasks (4x faster than previous gen)",
        "llama-3.3-70b-versatile": "Powerful versatile model with excellent comprehension (recommended)",
        "meta-llama/llama-4-maverick-17b-128e-instruct": "Latest Llama 4 model with superior reasoning",
        
        # Specialized Models
        "deepseek-r1-distill-llama-70b": "Superior reasoning for complex emails and analytical tasks",
        "compound-beta": "Groq's agentic system for sophisticated email handling",
        
        # Additional Models
        "gemma2-9b-it": "Fast with good reasoning, efficient for straightforward emails",
        "qwen-qwq-32b": "Strong multilingual capabilities for international correspondence",
        
        # Previous Generation (Maintained for compatibility)
        "llama3-70b-8192": "Previous generation Llama model",
    }
    
    # Configuration for OpenAI and fallback
    use_openai = st.checkbox("Use OpenAI GPT", value=True, 
                           help="Use OpenAI GPT as the primary model. Requires API key.")
    
    use_fallback = st.checkbox("Enable Fallback Models", value=True,
                             help="If OpenAI fails or is unavailable, use LangChain models as fallback")
    
    if use_fallback:
        fallback_model = st.selectbox(
            "Fallback Model",
            options=list(models.keys()),
            index=1,  # Default to llama-3.3-70b-versatile
            format_func=lambda x: f"{x} - {models[x]}",
            help="Select which model to use as fallback if OpenAI is unavailable"
        )
    else:
        fallback_model = None
        
    # Store settings in session state
    if 'use_openai' not in st.session_state:
        st.session_state['use_openai'] = use_openai
    if use_openai != st.session_state['use_openai']:
        st.session_state['use_openai'] = use_openai
        
    if 'use_fallback' not in st.session_state:
        st.session_state['use_fallback'] = use_fallback
    if use_fallback != st.session_state['use_fallback']:
        st.session_state['use_fallback'] = use_fallback
        
    if 'fallback_model' not in st.session_state:
        st.session_state['fallback_model'] = fallback_model if fallback_model else "llama-3.3-70b-versatile"
    if fallback_model != st.session_state.get('fallback_model') and fallback_model:
        st.session_state['fallback_model'] = fallback_model
    
    # Store which model was last used for each operation
    if 'last_summary_model' not in st.session_state:
        st.session_state['last_summary_model'] = None
    if 'last_classify_model' not in st.session_state:
        st.session_state['last_classify_model'] = None
    if 'last_reply_model' not in st.session_state:
        st.session_state['last_reply_model'] = None
        
    st.markdown("---")
    
    if st.button("Fetch Emails"):
        if service:
            with st.spinner("Fetching emails..."):
                st.session_state['emails'] = get_emails(service, max_results=max_results, query=email_query or None)
            st.session_state['selected_email_id'] = None
            st.session_state['generated_reply'] = ""
            st.session_state['email_summary'] = ""
            st.session_state['last_summary_model'] = None
            st.session_state['last_classify_model'] = None
            st.session_state['last_reply_model'] = None
            st.rerun()
        else:
            st.warning("Please connect to Gmail first.")

# Email listing and processing
if 'emails' in st.session_state and st.session_state['emails']:
    emails = st.session_state['emails']
    
    # Email selection sidebar
    email_options = {f"{email['subject']} (From: {email['from']})": email['id'] for email in emails}
    selected_email_key = st.sidebar.selectbox("Select Email to Process", options=list(email_options.keys()))
    
    if selected_email_key:
        selected_id_candidate = email_options.get(selected_email_key)
        if selected_id_candidate != st.session_state.get('selected_email_id'):
            st.session_state['selected_email_id'] = selected_id_candidate
            st.session_state['generated_reply'] = ""
            st.session_state['email_summary'] = ""
            st.session_state['last_summary_model'] = None
            
            # Auto-generate summary for the newly selected email
            if service:
                try:
                    # Get user email information for personalization
                    user_info = service.users().getProfile(userId='me').execute()
                    recipient_info = {
                        'name': user_info.get('emailAddress', '').split('@')[0],
                        'email': user_info.get('emailAddress', '')
                    }
                except:
                    recipient_info = None
                
                selected_email = next((email for email in emails if email['id'] == selected_id_candidate), None)
                if selected_email:
                    with st.spinner("Generating summary..."):
                        summary, model_used = summarize_email(
                            selected_email['body'], 
                            recipient_info,
                            use_openai=st.session_state.get('use_openai', True),
                            use_fallback=st.session_state.get('use_fallback', True),
                            fallback_model=st.session_state.get('fallback_model')
                        )
                        st.session_state['email_summary'] = summary
                        st.session_state['last_summary_model'] = model_used

    # Email details and actions
    selected_id = st.session_state.get('selected_email_id')
    if selected_id:
        selected_email = next((email for email in emails if email['id'] == selected_id), None)
        
        if selected_email:
            st.subheader(f"Subject: {selected_email['subject']}")
            st.caption(f"From: {selected_email['from']} | To: {selected_email['to']} | Date: {selected_email['date']}")
            
            with st.expander("View Full Email Body", expanded=False):
                st.text(selected_email['body'])
                
            # Email summary display
            with st.expander("View Email Summary", expanded=False):
                if st.session_state.get('email_summary'):
                    st.markdown(st.session_state['email_summary'])
                    if st.session_state.get('last_summary_model'):
                        st.caption(f"⚙️ Summary generated using: {st.session_state['last_summary_model']}")
                else:
                    st.info("Summary not available. Please try regenerating the summary.")
                
            # Email analysis actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Classify Email"):
                    with st.spinner("Classifying..."):
                        category, model_used = classify_email(
                            selected_email['body'], 
                            selected_email['subject'],
                            use_openai=st.session_state.get('use_openai', True),
                            use_fallback=st.session_state.get('use_fallback', True),
                            fallback_model=st.session_state.get('fallback_model')
                        )
                    st.session_state['last_classify_model'] = model_used
                    st.info(f"Email classified as: **{category}**")
                    st.caption(f"⚙️ Classification using: {model_used}")
            
            with col2:
                 if st.button("Regenerate Summary"):
                    with st.spinner("Generating summary..."):
                        # Get user's email from Gmail service
                        try:
                            user_info = service.users().getProfile(userId='me').execute()
                            recipient_info = {
                                'name': user_info.get('emailAddress', '').split('@')[0],  # Use username part of email
                                'email': user_info.get('emailAddress', '')
                            }
                        except:
                            recipient_info = None
                            
                        summary, model_used = summarize_email(
                            selected_email['body'], 
                            recipient_info,
                            use_openai=st.session_state.get('use_openai', True),
                            use_fallback=st.session_state.get('use_fallback', True),
                            fallback_model=st.session_state.get('fallback_model')
                        )
                    st.session_state['email_summary'] = summary
                    st.session_state['last_summary_model'] = model_used
                    st.success("Summary regenerated successfully")
                    st.caption(f"⚙️ Generated using: {model_used}")

            st.divider()
            st.subheader("Generate Reply")
            
            # Reply configuration options
            reply_col1, reply_col2, reply_col3 = st.columns(3)
            with reply_col1:
                tone = st.selectbox("Tone", ["Friendly", "Formal", "Direct", "Empathetic"], index=0)
            with reply_col2:
                style = st.selectbox("Style", ["Concise", "Detailed", "Professional", "Casual"], index=0)
            with reply_col3:
                length = st.selectbox("Length", ["Brief", "Standard", "Comprehensive"], index=1)

            # Context for reply generation
            user_context_input = st.text_area(
                "Optional Context for Reply:", 
                placeholder="e.g., I had a fever; Please reschedule the meeting; Ask for clarification on point 3.",
                help="Provide brief context, keywords, or sentences to guide the reply generation. Leave blank if not needed."
            )
            
            if st.button("Generate Draft Reply"):
                with st.spinner("Generating draft reply..."):
                    generated_reply, model_used = generate_reply(
                        selected_email['body'],
                        selected_email['subject'],
                        selected_email['from'],
                        tone, 
                        style, 
                        length,
                        user_context=user_context_input,
                        use_openai=st.session_state.get('use_openai', True),
                        use_fallback=st.session_state.get('use_fallback', True),
                        fallback_model=st.session_state.get('fallback_model')
                    )
                st.session_state['generated_reply'] = generated_reply
                st.session_state['last_reply_model'] = model_used
                st.success("Draft reply generated")
                st.caption(f"⚙️ Generated using: {model_used}")
            
            # Edit and send reply
            if 'generated_reply' in st.session_state and st.session_state['generated_reply']:
                edited_reply = st.text_area("Edit Reply:", value=st.session_state['generated_reply'], height=250)
                
                # Display which model was used for generation
                if st.session_state.get('last_reply_model'):
                    st.caption(f"⚙️ Draft generated using: {st.session_state['last_reply_model']}")
                
                if st.button("Send Reply"):
                    if service:
                        reply_subject = selected_email['subject']
                        if not reply_subject.lower().startswith("re:"):
                            reply_subject = "Re: " + reply_subject
                        
                        send_status = send_email(
                            service, 
                            selected_email['from'],
                            reply_subject,
                            edited_reply,
                            thread_id=selected_email['thread_id'],
                            original_message_id=selected_email['id'],
                            custom_signature=st.session_state.get('custom_signature')
                        )
                        
                        if send_status:
                            st.success(f"Reply sent successfully to {selected_email['from']}!")
                            modify_success = modify_email_labels(service, selected_id, labels_to_remove=['UNREAD'])
                            if modify_success:
                                st.info("Marked email as read.")
                            else:
                                st.warning(f"Could not mark email as read.")
                                
                            st.session_state['generated_reply'] = ""
                            st.session_state['email_summary'] = ""
                            st.session_state['selected_email_id'] = None
                            st.session_state['emails'] = None
                            st.session_state['last_summary_model'] = None
                            st.session_state['last_classify_model'] = None
                            st.session_state['last_reply_model'] = None
                            st.rerun()
                        else:
                            st.error("Failed to send the reply.")
                    else:
                        st.error("Authentication error. Cannot send email.")

# No emails message
elif 'emails' in st.session_state and not st.session_state['emails']:
    st.info("No emails found matching your query or no unread emails.")

# Footer
st.divider()
st.caption("Gmail Response Assistant • Powered by OpenAI GPT • Built with Streamlit")