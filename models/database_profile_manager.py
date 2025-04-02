import os
import json
import uuid
import logging
import copy
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from config import Config  # Import the Config class

class DatabaseProfileManager:
    """
    Database-backed Profile Management System for creating, loading, updating, and versioning user profiles.
    Uses SQLite for persistent storage with JSON serialization of profile data.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the DatabaseProfileManager with SQLite database.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        if db_path is None:
            db_path = Config.DB_PATH  # Use the DB_PATH from Config if not provided
        self.db_path = db_path
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Memory cache for profiles to ensure single instance
        self.cache = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize database tables
        self._initialize_database()
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for getting a database connection.
        Handles transaction management and connection closing.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            # Enables returning rows as dictionaries
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self):
        """
        Initialize database tables if they don't exist.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create profiles table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Create profile_versions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS profile_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES profiles (id) ON DELETE CASCADE
                )
                ''')
                
                # Create goal categories table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS goal_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    order_index INTEGER NOT NULL DEFAULT 0,
                    is_foundation BOOLEAN NOT NULL DEFAULT 0
                )
                ''')
                
                # Create goals table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    user_profile_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    target_amount REAL,
                    timeframe TEXT,
                    current_amount REAL DEFAULT 0,
                    importance TEXT CHECK(importance IN ('high', 'medium', 'low')) DEFAULT 'medium',
                    flexibility TEXT CHECK(flexibility IN ('fixed', 'somewhat_flexible', 'very_flexible')) DEFAULT 'somewhat_flexible',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_profile_id) REFERENCES profiles (id) ON DELETE CASCADE
                )
                ''')
                
                # Insert predefined goal categories if they don't exist yet
                self._initialize_goal_categories(cursor)
                
                conn.commit()
                logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize database: {str(e)}")
            raise RuntimeError(f"Database initialization failed: {str(e)}")
            
    def _initialize_goal_categories(self, cursor):
        """
        Initialize predefined goal categories in the database.
        
        Args:
            cursor: Database cursor
        """
        # Check if categories already exist
        cursor.execute("SELECT COUNT(*) FROM goal_categories")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logging.info("Goal categories already initialized")
            return
            
        # Define predefined categories based on goals framework
        categories = [
            # Security goals (foundation)
            {"name": "emergency_fund", "description": "Emergency fund for unexpected expenses", "order_index": 1, "is_foundation": 1},
            {"name": "insurance", "description": "Insurance coverage for protection", "order_index": 2, "is_foundation": 1},
            
            # Essential goals
            {"name": "home_purchase", "description": "Saving for home purchase or down payment", "order_index": 3, "is_foundation": 0},
            {"name": "education", "description": "Education funding for self or family", "order_index": 4, "is_foundation": 0},
            {"name": "debt_elimination", "description": "Paying off existing debts", "order_index": 5, "is_foundation": 0},
            
            # Retirement goals
            {"name": "early_retirement", "description": "Saving for early retirement", "order_index": 6, "is_foundation": 0},
            {"name": "traditional_retirement", "description": "Saving for traditional retirement age", "order_index": 7, "is_foundation": 0},
            
            # Lifestyle goals
            {"name": "travel", "description": "Saving for travel experiences", "order_index": 8, "is_foundation": 0},
            {"name": "vehicle", "description": "Saving for vehicle purchase", "order_index": 9, "is_foundation": 0},
            {"name": "home_improvement", "description": "Saving for home improvements or renovations", "order_index": 10, "is_foundation": 0},
            
            # Legacy goals
            {"name": "estate_planning", "description": "Planning for wealth transfer and estate", "order_index": 11, "is_foundation": 0},
            {"name": "charitable_giving", "description": "Saving for charitable donations or giving", "order_index": 12, "is_foundation": 0},
            
            # Custom goals
            {"name": "custom", "description": "User-defined custom goal", "order_index": 13, "is_foundation": 0}
        ]
        
        # Insert categories
        for category in categories:
            cursor.execute(
                "INSERT INTO goal_categories (name, description, order_index, is_foundation) VALUES (?, ?, ?, ?)",
                (category["name"], category["description"], category["order_index"], category["is_foundation"])
            )
            
        logging.info(f"Initialized {len(categories)} goal categories")
    
    def create_profile(self, name, email):
        """
        Create a new user profile with basic info.
        
        Args:
            name (str): User's name
            email (str): User's email address
            
        Returns:
            dict: Newly created profile object
        """
        profile_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        profile = {
            "id": profile_id,
            "name": name,
            "email": email,
            "answers": [],
            "created_at": current_time,
            "updated_at": current_time,
            "versions": [
                {
                    "version_id": 1,
                    "timestamp": current_time,
                    "reason": "initial_creation"
                }
            ],
            "_debug_marker": f"created_{uuid.uuid4().hex[:6]}"
        }
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert profile into database
                cursor.execute(
                    "INSERT INTO profiles (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (profile_id, json.dumps(profile), current_time, current_time)
                )
                
                # Insert initial version
                cursor.execute(
                    "INSERT INTO profile_versions (profile_id, data, version, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                    (profile_id, json.dumps(profile), 1, "initial_creation", current_time)
                )
                
                conn.commit()
                
                # Store in cache
                self.cache[profile_id] = profile
                
                logging.info(f"Created new profile with ID: {profile_id}")
                return profile
                
        except Exception as e:
            logging.error(f"Failed to create profile: {str(e)}")
            raise ValueError(f"Failed to create profile: {str(e)}")
    
    def get_profile(self, profile_id):
        """
        Load a profile by ID, prioritizing cache for reference consistency.
        
        Args:
            profile_id (str): Unique ID of the profile to load
            
        Returns:
            dict: Profile object or None if not found
        """
        # First check the cache for existing reference
        if profile_id in self.cache:
            logging.info(f"Retrieved profile {profile_id} from cache (id: {id(self.cache[profile_id])})")
            return self.cache[profile_id]
        
        # Otherwise load from database
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT data FROM profiles WHERE id = ?", (profile_id,))
                result = cursor.fetchone()
                
                if not result:
                    logging.error(f"Profile {profile_id} not found in database")
                    return None
                
                # Parse JSON data
                profile = json.loads(result['data'])
                
                # Store in cache for future reference consistency
                self.cache[profile_id] = profile
                
                logging.info(f"Loaded profile {profile_id} from database (id: {id(profile)})")
                return profile
                
        except Exception as e:
            logging.error(f"Failed to load profile {profile_id}: {str(e)}")
            return None
    
    def save_profile(self, profile):
        """
        Update profile in both cache and database.
        
        Args:
            profile (dict): Profile object to save
            
        Returns:
            dict: Updated profile object
        """
        # Verify we have a valid profile
        if not profile or 'id' not in profile:
            logging.error(f"Cannot save invalid profile: {profile}")
            raise ValueError("Invalid profile object")
        
        profile_id = profile['id']
        
        # Make sure this exact object reference is in our cache
        if profile_id in self.cache and id(profile) != id(self.cache[profile_id]):
            logging.warning(f"Profile reference mismatch: {id(profile)} vs {id(self.cache[profile_id])}")
            # Update our cache to use this reference, to maintain consistency
            self.cache[profile_id] = profile
        
        # Update timestamp
        profile['updated_at'] = datetime.now().isoformat()
        
        try:
            # Verify structure before saving
            if 'answers' not in profile:
                profile['answers'] = []
            
            # Ensure required fields are present
            required_fields = ['id', 'created_at', 'updated_at']
            for field in required_fields:
                if field not in profile:
                    if field == 'created_at' or field == 'updated_at':
                        profile[field] = datetime.now().isoformat()
                    else:
                        logging.error(f"Required field {field} missing from profile")
                        raise ValueError(f"Required field {field} missing from profile")
            
            # Check answers count before serialization
            answers_count = len(profile.get('answers', []))
            logging.info(f"Before save: Profile has {answers_count} answers")
            
            # Create a copy for serialization to avoid reference issues
            profile_copy = copy.deepcopy(profile)
            
            # Remove debug fields for serialization
            if '_object_id' in profile_copy:
                del profile_copy['_object_id']
            
            # Serialize profile
            json_string = json.dumps(profile_copy)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if profile exists
                cursor.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing profile
                    cursor.execute(
                        "UPDATE profiles SET data = ?, updated_at = ? WHERE id = ?",
                        (json_string, profile['updated_at'], profile_id)
                    )
                else:
                    # Insert new profile (shouldn't normally happen through this method)
                    cursor.execute(
                        "INSERT INTO profiles (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (profile_id, json_string, profile['created_at'], profile['updated_at'])
                    )
                
                conn.commit()
            
            # Ensure it's in our cache
            self.cache[profile_id] = profile
            
            logging.info(f"Saved profile {profile_id} (answers: {len(profile.get('answers', []))})")
            return profile
            
        except Exception as e:
            logging.error(f"Failed to save profile {profile_id}: {str(e)}")
            raise ValueError(f"Failed to save profile: {str(e)}")
    
    def add_answer(self, profile, question_id, answer_value):
        """
        Add a new answer to a profile or update existing answer.
        
        Args:
            profile (dict): Profile to update
            question_id (str): ID of the question being answered
            answer_value: The answer value (could be string, number, boolean, etc.)
            
        Returns:
            dict: Updated profile with new answer
        """
        if not profile or 'id' not in profile:
            logging.error("Invalid profile for adding answer")
            raise ValueError("Invalid profile")
        
        # Ensure this is the cached reference
        profile_id = profile['id']
        if profile_id in self.cache and id(profile) != id(self.cache[profile_id]):
            logging.warning(f"Using different profile reference for {profile_id}")
            profile = self.cache[profile_id]  # Use cached reference
        
        # Ensure answers array exists
        if 'answers' not in profile:
            profile['answers'] = []
        
        # Check if this question already has an answer
        answer_updated = False
        for answer in profile['answers']:
            if answer['question_id'] == question_id:
                answer['answer'] = answer_value
                answer['timestamp'] = datetime.now().isoformat()
                answer_updated = True
                break
        
        # If not found, create new answer
        if not answer_updated:
            new_answer = {
                "id": str(uuid.uuid4()),
                "question_id": question_id,
                "answer": answer_value,
                "timestamp": datetime.now().isoformat()
            }
            profile['answers'].append(new_answer)
        
        # Save changes
        self.save_profile(profile)
        
        # Verify answers are still there
        cache_profile = self.cache[profile_id]
        logging.info(f"After add_answer: Profile has {len(cache_profile.get('answers', []))} answers")
        
        return profile
    
    def create_version(self, profile, reason):
        """
        Create a new version of the profile for tracking major changes.
        
        Args:
            profile (dict): Profile to version
            reason (str): Reason for creating new version
            
        Returns:
            dict: Updated profile with new version
        """
        if not profile or 'id' not in profile:
            logging.error("Invalid profile for versioning")
            raise ValueError("Invalid profile")
        
        # Ensure profile is the cached reference
        profile_id = profile['id']
        if profile_id in self.cache and id(profile) != id(self.cache[profile_id]):
            logging.warning(f"Using different profile reference for versioning {profile_id}")
            profile = self.cache[profile_id]
        
        # Initialize versions array if needed
        if 'versions' not in profile:
            profile['versions'] = []
        
        # Get latest version ID
        latest_version = max([v['version_id'] for v in profile['versions']]) if profile['versions'] else 0
        
        # Create new version entry
        current_time = datetime.now().isoformat()
        new_version = {
            "version_id": latest_version + 1,
            "timestamp": current_time,
            "reason": reason
        }
        
        # Add version to profile
        profile['versions'].append(new_version)
        
        try:
            # Create a deep copy for the version snapshot
            version_snapshot = copy.deepcopy(profile)
            
            # Remove any debug fields
            if '_object_id' in version_snapshot:
                del version_snapshot['_object_id']
            
            # Serialize version
            version_json = json.dumps(version_snapshot)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert into profile_versions table
                cursor.execute(
                    "INSERT INTO profile_versions (profile_id, data, version, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                    (profile_id, version_json, new_version['version_id'], reason, current_time)
                )
                
                conn.commit()
                
            logging.info(f"Created version {new_version['version_id']} for profile {profile_id}")
            
        except Exception as e:
            logging.error(f"Failed to save version snapshot: {str(e)}")
            # Continue without failing - version record in profile object still exists
        
        # Save the updated profile
        self.save_profile(profile)
        
        return profile
    
    def get_profile_versions(self, profile_id):
        """
        Get all versions of a profile.
        
        Args:
            profile_id (str): ID of the profile
            
        Returns:
            list: List of profile versions
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT data, version, reason, created_at FROM profile_versions WHERE profile_id = ? ORDER BY version",
                    (profile_id,)
                )
                
                results = cursor.fetchall()
                versions = []
                
                for result in results:
                    version_data = json.loads(result['data'])
                    versions.append({
                        'version': result['version'],
                        'reason': result['reason'],
                        'created_at': result['created_at'],
                        'data': version_data
                    })
                
                return versions
                
        except Exception as e:
            logging.error(f"Failed to get profile versions for {profile_id}: {str(e)}")
            return []
    
    def get_version(self, profile_id, version_number):
        """
        Get a specific version of a profile.
        
        Args:
            profile_id (str): ID of the profile
            version_number (int): Version number to retrieve
            
        Returns:
            dict: Profile version or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT data FROM profile_versions WHERE profile_id = ? AND version = ?",
                    (profile_id, version_number)
                )
                
                result = cursor.fetchone()
                
                if not result:
                    logging.error(f"Version {version_number} for profile {profile_id} not found")
                    return None
                
                return json.loads(result['data'])
                
        except Exception as e:
            logging.error(f"Failed to get profile version {version_number} for {profile_id}: {str(e)}")
            return None
    
    def delete_profile(self, profile_id):
        """
        Delete a profile and all its versions.
        
        Args:
            profile_id (str): ID of the profile to delete
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Foreign key constraint will delete related versions
                cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
                
                conn.commit()
                
                # Remove from cache
                if profile_id in self.cache:
                    del self.cache[profile_id]
                
                logging.info(f"Deleted profile {profile_id}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to delete profile {profile_id}: {str(e)}")
            return False
    
    def get_all_profiles(self):
        """
        Get all profiles (basic info only).
        
        Returns:
            list: List of profile summaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT id, data, created_at, updated_at FROM profiles")
                
                results = cursor.fetchall()
                profiles = []
                
                for result in results:
                    profile_data = json.loads(result['data'])
                    profiles.append({
                        'id': result['id'],
                        'name': profile_data.get('name', 'Unknown'),
                        'email': profile_data.get('email', 'Unknown'),
                        'created_at': result['created_at'],
                        'updated_at': result['updated_at'],
                        'answers_count': len(profile_data.get('answers', []))
                    })
                
                return profiles
                
        except Exception as e:
            logging.error(f"Failed to get all profiles: {str(e)}")
            return []
            
    def _export_profile_to_json(self, profile_id, output_path):
        """
        Export a profile to a JSON file (utility method for migration).
        
        Args:
            profile_id (str): ID of the profile to export
            output_path (str): Path to save the JSON file
            
        Returns:
            bool: Success status
        """
        profile = self.get_profile(profile_id)
        if not profile:
            return False
            
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as file:
                json.dump(profile, file, indent=2)
                
            return True
        except Exception as e:
            logging.error(f"Failed to export profile {profile_id}: {str(e)}")
            return False
            
    def _import_profile_from_json(self, input_path):
        """
        Import a profile from a JSON file (utility method for migration).
        
        Args:
            input_path (str): Path to the JSON file
            
        Returns:
            str: Profile ID if successful, None otherwise
        """
        try:
            with open(input_path, 'r') as file:
                profile = json.load(file)
                
            profile_id = profile.get('id')
            if not profile_id:
                return None
                
            # Ensure timestamps are present
            if 'created_at' not in profile:
                profile['created_at'] = datetime.now().isoformat()
            if 'updated_at' not in profile:
                profile['updated_at'] = datetime.now().isoformat()
                
            # Save to database
            self.cache[profile_id] = profile
            self.save_profile(profile)
            
            return profile_id
            
        except Exception as e:
            logging.error(f"Failed to import profile from {input_path}: {str(e)}")
            return None