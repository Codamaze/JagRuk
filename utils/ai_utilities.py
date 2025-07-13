import streamlit as st
import google.generativeai as genai
import os
import time
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# --- Configuration & Initialization ---
def initialize_gemini_api() -> Optional[Any]:
    """Initialize Google Gemini API with proper error handling."""
    api_key = None
    # Prioritize Streamlit secrets, then environment variables
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ:
        api_key = os.environ["GEMINI_API_KEY"]

    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Use 'gemini-1.5-flash' for faster generation, or 'gemini-1.5-pro' for higher quality if needed
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            logger.info("Gemini API initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            st.error(f"Failed to load Gemini model. Check API key and model name. Error: {e}")
            return None
    else:
        logger.warning("Google Gemini API key not found.")
        st.warning("Google Gemini API key not found. GenAI features will be unavailable.")
        return None

# Initialize model globally once
GEMINI_MODEL = initialize_gemini_api()

# --- Enhanced GenAI Explanations Function ---
@st.cache_data(ttl=1800)  # Cache results for 30 minutes
def get_gemini_explanation(prompt: str, cache_key: Optional[str] = None, max_retries: int = 3) -> str:
    """Generate explanation using Gemini API with caching and retry logic."""
    if GEMINI_MODEL is None:
        return "*(AI explanation unavailable: Gemini API not configured)*"
    
    # Define safety settings to avoid blocking responses unnecessarily (adjust based on policy)
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'block_none',
        'HARM_CATEGORY_HATE_SPEECH': 'block_none',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'block_none',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'block_none'
    }

    for attempt in range(max_retries):
        try:
            response = GEMINI_MODEL.generate_content(prompt, safety_settings=safety_settings)
            return response.text
        except Exception as e:
            logger.warning(f"Gemini generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return f"Error generating explanation: {e}. Please try again later."
            time.sleep(1) # Brief delay before retry

    return "Error generating explanation after multiple retries."