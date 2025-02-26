# This file will handle environment variables and configuration
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Email for PubMed from environment variable
    PUBMED_EMAIL = os.getenv('PUBMED_EMAIL')
    
    # Secret key for session state encryption
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    @staticmethod
    def validate():
        """Validate required configuration variables."""
        missing = []
        if not Config.OPENAI_API_KEY:
            missing.append('OPENAI_API_KEY')
        if not Config.PUBMED_EMAIL:
            missing.append('PUBMED_EMAIL')
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")