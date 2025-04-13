import streamlit as st

# Page config
st.set_page_config(
    page_title="COSMOS | Collaborative Organized System for Multiple Operating Specialists",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* General background and text */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }

    /* Centered layout container */
    .centered-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 70vh;
        text-align: center;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #D1D5DB;
        margin-bottom: 1rem;
    }

    /* Subtitle */
    .subtitle {
        font-size: 1.25rem;
        color: #9CA3AF;
        margin-bottom: 3rem;
    }

    /* Footer */
    .page-info {
        font-size: 0.875rem;
        text-align: center;
        margin: 3rem 0 1rem;
        color: #6B7280;
    }

    /* Sidebar text contrast */
    .css-1d391kg h2, .css-1d391kg p, .css-1d391kg li {
        color: #E5E7EB !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.write(
        """  
        COSMOS is your team of AI assistants - each one great at a specific task. Need to chat about documents? Process YouTube videos? Handle emails? We've got you covered, all in one place.
        """
    )
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Main Content - Centered
st.markdown(
    """
    <div class="centered-container">
        <div class="main-title">COSMOS</div>
        <div class="subtitle">Collaborative Organized System for Multiple Operating Specialists</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown(
    '<div class="page-info">© 2025 COSMOS • Built with Streamlit & advanced AI models</div>',
    unsafe_allow_html=True
)