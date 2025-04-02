#!/usr/bin/env python3
"""
Migration utility to move profiles from file-based storage to SQLite database.
"""

import os
import json
import logging
import argparse
from models.profile_manager import ProfileManager
from models.database_profile_manager import DatabaseProfileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def find_profile_files(profiles_dir):
    """Find all JSON profile files in the directory."""
    profile_files = []
    try:
        for filename in os.listdir(profiles_dir):
            if filename.endswith('.json') and not filename.endswith('.bak') and not filename.endswith('.tmp'):
                profile_files.append(os.path.join(profiles_dir, filename))
        return profile_files
    except Exception as e:
        logging.error(f"Error finding profile files: {str(e)}")
        return []

def migrate_profiles(source_dir, db_path):
    """Migrate profiles from files to SQLite database."""
    # Initialize managers
    file_manager = ProfileManager(data_directory=source_dir)
    db_manager = DatabaseProfileManager(db_path=db_path)
    
    # Find all profile files
    profile_files = find_profile_files(source_dir)
    logging.info(f"Found {len(profile_files)} profile files to migrate")
    
    successful = 0
    failed = 0
    
    for file_path in profile_files:
        try:
            # Extract profile ID from filename
            filename = os.path.basename(file_path)
            profile_id = filename.replace('.json', '')
            
            logging.info(f"Migrating profile {profile_id}")
            
            # Load profile from file
            profile = file_manager.get_profile(profile_id)
            if not profile:
                logging.error(f"Failed to load profile {profile_id}")
                failed += 1
                continue
            
            # Save to database
            db_manager.cache[profile_id] = profile
            db_manager.save_profile(profile)
            
            # Migrate versions if available
            versions_dir = os.path.join(source_dir, profile_id, 'versions')
            if os.path.exists(versions_dir):
                for version_file in os.listdir(versions_dir):
                    if version_file.endswith('.json'):
                        try:
                            version_path = os.path.join(versions_dir, version_file)
                            with open(version_path, 'r') as f:
                                version_data = json.load(f)
                            
                            # Extract version info
                            version_parts = version_file.split('_')
                            if len(version_parts) >= 2:
                                version_num = int(version_parts[0].replace('v', ''))
                                
                                # Insert directly into versions table
                                reason = "migrated_from_file"
                                for v in profile.get('versions', []):
                                    if v.get('version_id') == version_num:
                                        reason = v.get('reason', reason)
                                
                                with db_manager._get_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "INSERT INTO profile_versions (profile_id, data, version, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                                        (profile_id, json.dumps(version_data), version_num, reason, version_data.get('updated_at', version_data.get('created_at')))
                                    )
                                    conn.commit()
                        except Exception as ve:
                            logging.error(f"Failed to migrate version file {version_file}: {str(ve)}")
            
            successful += 1
            logging.info(f"Successfully migrated profile {profile_id}")
            
        except Exception as e:
            logging.error(f"Failed to migrate {file_path}: {str(e)}")
            failed += 1
    
    logging.info(f"Migration complete: {successful} successful, {failed} failed")
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description="Migrate profiles from file-based storage to SQLite database")
    parser.add_argument('--source', default="/Users/coddiwomplers/Desktop/Python/Profiler4/data/profiles", 
                        help="Source directory containing profile JSON files")
    parser.add_argument('--db', default="/Users/coddiwomplers/Desktop/Python/Profiler4/data/profiles.db",
                        help="Target SQLite database path")
    args = parser.parse_args()
    
    print(f"Starting migration from {args.source} to {args.db}")
    print("This will preserve all profiles and their version history.")
    print("The original files will not be modified.")
    
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Migration cancelled.")
        return
    
    successful, failed = migrate_profiles(args.source, args.db)
    print(f"Migration complete: {successful} profiles migrated successfully, {failed} failed")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Migration utility to move profiles from file-based storage to SQLite database.
"""

import os
import json
import logging
import argparse
from models.profile_manager import ProfileManager
from models.database_profile_manager import DatabaseProfileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def find_profile_files(profiles_dir):
    """Find all JSON profile files in the directory."""
    profile_files = []
    try:
        for filename in os.listdir(profiles_dir):
            if filename.endswith('.json') and not filename.endswith('.bak') and not filename.endswith('.tmp'):
                profile_files.append(os.path.join(profiles_dir, filename))
        return profile_files
    except Exception as e:
        logging.error(f"Error finding profile files: {str(e)}")
        return []

def migrate_profiles(source_dir, db_path):
    """Migrate profiles from files to SQLite database."""
    # Initialize managers
    file_manager = ProfileManager(data_directory=source_dir)
    db_manager = DatabaseProfileManager()
    
    # Find all profile files
    profile_files = find_profile_files(source_dir)
    logging.info(f"Found {len(profile_files)} profile files to migrate")
    
    successful = 0
    failed = 0
    
    for file_path in profile_files:
        try:
            # Extract profile ID from filename
            filename = os.path.basename(file_path)
            profile_id = filename.replace('.json', '')
            
            logging.info(f"Migrating profile {profile_id}")
            
            # Load profile from file
            profile = file_manager.get_profile(profile_id)
            if not profile:
                logging.error(f"Failed to load profile {profile_id}")
                failed += 1
                continue
            
            # Save to database
            db_manager.cache[profile_id] = profile
            db_manager.save_profile(profile)
            
            # Migrate versions if available
            versions_dir = os.path.join(source_dir, profile_id, 'versions')
            if os.path.exists(versions_dir):
                for version_file in os.listdir(versions_dir):
                    if version_file.endswith('.json'):
                        try:
                            version_path = os.path.join(versions_dir, version_file)
                            with open(version_path, 'r') as f:
                                version_data = json.load(f)
                            
                            # Extract version info
                            version_parts = version_file.split('_')
                            if len(version_parts) >= 2:
                                version_num = int(version_parts[0].replace('v', ''))
                                
                                # Insert directly into versions table
                                reason = "migrated_from_file"
                                for v in profile.get('versions', []):
                                    if v.get('version_id') == version_num:
                                        reason = v.get('reason', reason)
                                
                                with db_manager._get_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "INSERT INTO profile_versions (profile_id, data, version, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                                        (profile_id, json.dumps(version_data), version_num, reason, version_data.get('updated_at', version_data.get('created_at')))
                                    )
                                    conn.commit()
                        except Exception as ve:
                            logging.error(f"Failed to migrate version file {version_file}: {str(ve)}")
            
            successful += 1
            logging.info(f"Successfully migrated profile {profile_id}")
            
        except Exception as e:
            logging.error(f"Failed to migrate {file_path}: {str(e)}")
            failed += 1
    
    logging.info(f"Migration complete: {successful} successful, {failed} failed")
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description="Migrate profiles from file-based storage to SQLite database")
    parser.add_argument('--source', default="/Users/coddiwomplers/Desktop/Python/Profiler4/data/profiles", 
                        help="Source directory containing profile JSON files")
    parser.add_argument('--db', default="/Users/coddiwomplers/Desktop/Python/Profiler4/data/profiles.db",
                        help="Target SQLite database path")
    args = parser.parse_args()
    
    print(f"Starting migration from {args.source} to {args.db}")
    print("This will preserve all profiles and their version history.")
    print("The original files will not be modified.")
    
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Migration cancelled.")
        return
    
    successful, failed = migrate_profiles(args.source, args.db)
    print(f"Migration complete: {successful} profiles migrated successfully, {failed} failed")

if __name__ == "__main__":
    main()
