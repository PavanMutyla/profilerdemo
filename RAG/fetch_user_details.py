import os 
print(os.getcwd())
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.database_profile_manager import DatabaseProfileManager
from services.question_service import QuestionService, QuestionLogger
from services.llm_service import LLMService

def fetch_user_data(profile_id):
    """Fetch user data based on profile ID."""
    db_manager = DatabaseProfileManager()  # Initialize the database manager
    profile = db_manager.get_profile(profile_id)  # Fetch the profile
    
    if profile:
        print("User Profile Data:")
        print(profile)  # Print or process the profile data as needed
    else:
        print("Profile not found.")
