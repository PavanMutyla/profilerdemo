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

if __name__ == "__main__":

   db_manager = DatabaseProfileManager()
   all_profiles = db_manager.get_all_profiles()
   if all_profiles:
        print("All User Profile Data:")
        for profile_summary in all_profiles:
            profile_id = profile_summary['id']
            profile = db_manager.get_profile(profile_id)
            if profile:
              print(f"\nProfile ID: {profile_id}")
              print(profile)
            else:
               print(f"Profile not found for ID: {profile_id}")
else:
        print("No profiles found in the database.")