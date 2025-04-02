import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """
    Configuration class for the Financial Profiler application.
    Loads settings from environment variables with sensible defaults.
    """
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Database settings
    DB_PATH = os.environ.get('DB_PATH') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'profiles.db')
    
    # Use a more specific path for the database to avoid permission issues
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'profiles.db')
    
    # LLM settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
    
    # Allow explicit disabling of LLM via environment variable
    LLM_ENABLED_ENV = os.environ.get('LLM_ENABLED', 'True')
    LLM_ENABLED = LLM_ENABLED_ENV.lower() not in ('false', '0', 'f', 'no') and bool(OPENAI_API_KEY)
    
    # Profile settings
    DATA_DIRECTORY = os.environ.get('DATA_DIRECTORY') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PROFILES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'profiles')
    
    # Admin settings
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin')
    
    @classmethod
    def get_llm_status_message(cls):
        """
        Get a user-friendly message about the LLM service status
        """
        if cls.LLM_ENABLED:
            return {
                "status": "enabled",
                "message": f"LLM service is enabled using model: {cls.OPENAI_MODEL}",
                "model": cls.OPENAI_MODEL
            }
        else:
            return {
                "status": "disabled",
                "message": "LLM service is disabled. Set OPENAI_API_KEY environment variable to enable AI-powered question generation and response analysis.",
                "model": None
            }
    
    @classmethod
    def init_app(cls, app):
        """
        Initialize Flask app with configuration
        """
        # Ensure data directories exist
        os.makedirs(cls.DATA_DIRECTORY, exist_ok=True)
        os.makedirs(cls.PROFILES_DIRECTORY, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO if cls.DEBUG else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Log LLM status
        if cls.LLM_ENABLED:
            logging.info(f"LLM service is enabled using model: {cls.OPENAI_MODEL}")
        else:
            logging.warning("LLM service is disabled. Set OPENAI_API_KEY environment variable to enable advanced features.")