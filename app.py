import streamlit as st
import yaml
import streamlit_authenticator as authenticator
import os
from yaml.loader import SafeLoader
import Home
from dotenv import load_dotenv

load_dotenv()

# Initialize session state keys used by authenticator and logic
for key in ['authentication_status', 'username', 'name', 'logout']:
    if key not in st.session_state:
        st.session_state[key] = None

# Initial page config for the login screen
st.set_page_config(
    page_title="COSMOS | Login",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed" # Start with sidebar closed for login
)

# Load credentials from YAML
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Process environment variables in config
def replace_env_variables(config_dict):
    """Replace environment variables in the config dictionary"""
    if isinstance(config_dict, dict):
        for key, value in list(config_dict.items()):
            if isinstance(value, (dict, list)):
                replace_env_variables(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    # Convert numeric values to integers
                    if key == 'expiry_days' and env_value.isdigit():
                        config_dict[key] = int(env_value)
                    # Handle boolean values
                    elif env_value.lower() in ['true', 'false']:
                        config_dict[key] = env_value.lower() == 'true'
                    else:
                        config_dict[key] = env_value
                    
            # Special case for the username which is an environment variable itself
            if key.startswith('${') and key.endswith('}'):
                env_var = key[2:-1]
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    # Replace the env var key with its value
                    config_dict[env_value] = config_dict.pop(key)

replace_env_variables(config)

# Ensure cookie configuration values have the correct types
if 'cookie' in config:
    # Ensure expiry_days is an integer
    if not isinstance(config['cookie'].get('expiry_days'), int):
        config['cookie']['expiry_days'] = 30
    
    # Ensure secure and other boolean flags are boolean
    if not isinstance(config['cookie'].get('secure'), bool):
        config['cookie']['secure'] = True if config['cookie'].get('secure', 'true').lower() == 'true' else False

# Get signature key from environment variable (required for production)
signature_key = os.environ.get('SIGNATURE_KEY', 'this_is_a_default_key_only_for_local_dev')
if signature_key == 'this_is_a_default_key_only_for_local_dev':
    # Display warning only locally if default key is used
    st.warning("Using default signature key. Set SIGNATURE_KEY environment variable for production!", icon="⚠️")

# Initialize authenticator
authenticator = authenticator.Authenticate(
    config['credentials'],
    config['cookie']['name'], 
    signature_key, 
    int(config['cookie']['expiry_days']),  # Ensure this is an integer
    cookie_kwargs={
        'secure': bool(config['cookie'].get('secure', True)),
        'samesite': config['cookie'].get('samesite', 'Lax')
    }
)

# Custom CSS for login page styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Style the div containing the authenticator's form */
    div[data-testid="stForm"] {
        background-color: #1A1E2E; 
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 500px; 
        margin: 40px auto;
    }
    
    /* Style the title *inside* the login form */
    div[data-testid="stForm"] h1 {
        text-align: center !important;
        padding-top: 0px; 
        padding-bottom: 20px;
        color: #D1D5DB;
        font-size: 1.5rem; 
    }
    
    /* Input fields */
    div[data-testid="stTextInput"] input {
        background-color: #252A3B;
        color: #E0E0E0;
        border: 1px solid #3B4255;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border: none;
        padding: 10px 15px;
        font-weight: bold;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px; 
        color: #6B7280;
        font-size: 12px;
        padding-bottom: 20px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Displays the login form
def display_login_form():
    st.title("COSMOS Login")
    
    # Placeholder for status messages above the form
    message_placeholder = st.empty()

    # Render the login widget (centered via CSS)
    authenticator.login(location='main')

    # Update placeholder with status messages
    if st.session_state['authentication_status'] == False:
        message_placeholder.error("Username/password is incorrect")
    elif st.session_state['authentication_status'] is None:
        # Show initial prompt only if 'name' isn't set (first load/after logout)
        if st.session_state.get('name') is None:
             message_placeholder.warning("Please enter your username and password")
    else:
        # On successful login, clear the message placeholder
        message_placeholder.empty()

    # Footer
    st.markdown(
        '<div class="footer">© 2025 COSMOS • Built with Streamlit & advanced AI models</div>',
        unsafe_allow_html=True
    )

# Main application flow / Router
if st.session_state.get('authentication_status'):
    # --- Authenticated Flow --- 
    
    # Render the main app content (including its sidebar elements) first
    Home.main()
    
    # Add Welcome title and Logout button to the bottom of the sidebar
    with st.sidebar:
        # Check auth status again before adding logout button
        if st.session_state.get('authentication_status'): 
            st.title(f"Welcome, {st.session_state['name']}")
            st.divider() 
            authenticator.logout(location='sidebar')

    # If logout button was clicked in the sidebar rendering above, 
    # the status is now False. Rerun to show login page immediately.
    if not st.session_state.get('authentication_status'):
        st.rerun() 

else:
    # --- Unauthenticated Flow --- 
    display_login_form() 