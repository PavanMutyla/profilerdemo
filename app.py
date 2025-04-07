from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_httpauth import HTTPBasicAuth
import os
import logging
import json
from datetime import datetime
import uuid
import functools
import sqlite3

# Import our custom modules
from models.database_profile_manager import DatabaseProfileManager
from models.question_repository import QuestionRepository
from models.goal_models import GoalManager
from services.question_service import QuestionService
from services.llm_service import LLMService
from services.profile_analytics_service import ProfileAnalyticsService
from config import Config
from RAG.fetch_user_details import fetch_user_data  # Import the function
from RAG.rag import graph  # Import the graph from rag.py

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Initialize configuration for the app
Config.init_app(app)

# Set up HTTP Basic Auth for admin routes
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    """Verify admin credentials"""
    if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
        return username
    return None

# Initialize our components
profile_manager = DatabaseProfileManager(db_path=Config.DB_PATH)
question_repository = QuestionRepository()
llm_service = LLMService(api_key=Config.OPENAI_API_KEY, model=Config.OPENAI_MODEL)
question_service = QuestionService(question_repository, profile_manager, llm_service)
profile_analytics_service = ProfileAnalyticsService(profile_manager)
goal_manager = GoalManager(db_path=Config.DB_PATH)

DATABASE = 'data/users.db'  # Separate database for user authentication

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def create_user(username, password, email):
    """Create a new user in the database."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, password, email)  # Hash the password in a real application!
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "username": username, "email": email}
    except sqlite3.IntegrityError:
        # Handle username already exists error
        return None
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user against the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",  # Hash the password in a real application!
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    return user

def update_user_profile_id(user, profile_id):
    """Update user's profile_id in the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET profile_id = ? WHERE id = ?",
        (profile_id, user['id'])
    )
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')
@app.route('/profile/create', methods=['GET', 'POST'])
def create_profile():
    """Create a new profile"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        
        if not name or not email:
            flash('Name and email are required')
            return render_template('profile_create.html')
        
        # Create new profile
        profile = profile_manager.create_profile(name, email)
        
        # Store profile ID in session
        session['profile_id'] = profile['id']
        
        # Redirect to questions
        return redirect(url_for('questions'))
    
    return render_template('profile_create.html')

@app.route('/questions', methods=['GET'])
def questions():
    """Main questions interface"""
    # Check if we have an active profile
    profile_id = session.get('profile_id')
    if not profile_id:
        flash('No active profile. Please create a profile first.')
        return redirect(url_for('create_profile'))
    
    # Get the profile
    profile = profile_manager.get_profile(profile_id)
    if not profile:
        flash('Profile not found. Please create a new profile.')
        session.pop('profile_id', None)
        return redirect(url_for('create_profile'))
    
    # Check if we should skip the completion redirect
    skip_completion = request.args.get('skip_completion', '0') == '1'
    
    # Detect and prevent redirect loops
    from_complete = request.referrer and 'profile/complete' in request.referrer
    if from_complete:
        # If we're coming from the complete page, don't redirect back there
        skip_completion = True
        
    # Get next question and completion metrics
    next_question, _ = question_service.get_next_question(profile_id)
    completion = question_service.get_profile_completion(profile_id)
    
    # Log when a question is displayed to the user
    if next_question:
        question_id = next_question.get('id')
        question_text = next_question.get('text', '')
        logging.info(f"Displaying question to user: {question_id} - {question_text[:50]}...")
        
        # Ensure this question is in the cache before showing it to the user
        # This is a critical safety check to prevent questions appearing that aren't in the cache
        if question_id and question_id.startswith(('llm_next_level_', 'gen_question_', 'fallback_')):
            cached_questions = question_service.dynamic_questions_cache.get(profile_id, [])
            question_in_cache = any(q.get('id') == question_id for q in cached_questions)
            
            if not question_in_cache:
                logging.info(f"Adding displayed question to cache for safety: {question_id}")
                question_copy = next_question.copy()
                if profile_id not in question_service.dynamic_questions_cache:
                    question_service.dynamic_questions_cache[profile_id] = []
                question_service.dynamic_questions_cache[profile_id].append(question_copy)
        
        # Log the question display event
        question_service.question_logger.log_question_displayed(profile_id, question_id, next_question)
    
    # Ensure question ID is consistent - this can help with fallback questions
    if next_question and 'fallback_' in next_question.get('id', ''):
        # Log the exact question ID being presented to the user
        logging.info(f"Presenting fallback question to user: {next_question.get('id')}")
    
    # Count next-level questions that have been answered
    answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
    next_level_answered_count = 0
    
    for q_id in answered_ids:
        if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
            next_level_answered_count += 1
        # Count dynamic/LLM generated questions
        elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
            next_level_answered_count += 1
            
    logging.info(f"Profile {profile_id} has answered {next_level_answered_count}/5 required next-level questions")
    
    # Check if profile is complete
    is_complete = question_service.is_profile_complete(profile_id)
    
    # If we're below the threshold for next-level questions (5) but there's no next question,
    # force questions to continue rather than marking complete
    force_continue = next_level_answered_count < 5 and not next_question
    
    # Count goal and behavioral questions
    goal_questions_count = len([a.get('question_id') for a in profile.get('answers', []) 
                             if a.get('question_id', '').startswith('goals_') 
                             and not a.get('question_id', '').endswith('_insights')])
                             
    behavioral_questions_answered = len([a.get('question_id') for a in profile.get('answers', []) 
                                     if a.get('question_id', '').startswith('behavioral_') 
                                     and not a.get('question_id', '').endswith('_insights')])
    
    # Check if all minimum requirements are met
    meets_minimum_requirements = (
        is_complete and 
        goal_questions_count >= 7 and
        next_level_answered_count >= 5 and
        behavioral_questions_answered >= 3
    )
    
    # If profile is complete and not skipping completion, redirect to completion page
    # Only redirect if ALL required question types have been answered (core, goals, next-level, behavioral)
    if not skip_completion and not force_continue and meets_minimum_requirements:
        # Check if we were already on profile/complete to avoid redirect loops
        if not from_complete:
            logging.info(f"Redirecting profile {profile_id} to completion page with " +
                        f"{goal_questions_count} goal questions, " +
                        f"{next_level_answered_count} next-level questions, " +
                        f"{behavioral_questions_answered} behavioral questions")
            return redirect(url_for('profile_complete'))
    
    # If there's no next question to show but we're skipping completion redirect,
    # show a special message in the template
    no_questions = next_question is None
    
    # Get all answered questions for display
    answered_questions = question_service.get_answered_questions(profile_id)
    
    return render_template(
        'questions.html',
        profile=profile,
        next_question=next_question,
        completion=completion,
        answered_questions=answered_questions,
        no_questions=no_questions,
        next_level_count=next_level_answered_count
    )

@app.route('/answer/submit', methods=['POST'])
def submit_answer():
    """Process an answer submission"""
    profile_id = session.get('profile_id')
    if not profile_id:
        return jsonify({'success': False, 'error': 'No active profile'})
    
    question_id = request.form.get('question_id')
    logging.info(f"Received answer submission for question ID: {question_id}")
    
    # Detailed logging of the question submission
    logging.info(f"QUESTION SUBMISSION: Profile={profile_id}, Question ID={question_id}")
    logging.info(f"FORM DATA: {dict(request.form)}")
    
    # Skip repository validation for LLM-generated questions - expanded patterns
    is_llm_question = (question_id and 
                      ('llm_next_level' in question_id or 
                       'gen_question' in question_id or 
                       'fallback' in question_id or
                       'emergency' in question_id or
                       question_id == 'next_level_question_1' or 
                       question_id == 'next_level_question_2' or
                       question_id.startswith('question_')))
    
    # Special logging for LLM questions
    if is_llm_question:
        logging.info(f"DETECTED LLM QUESTION: {question_id}")
    
    # For non-LLM questions, check repository
    if not is_llm_question:
        question = question_repository.get_question(question_id)
        if not question:
            logging.error(f"❌ INVALID QUESTION ID: {question_id} - Not found in repository")
            
            # Special handling - if this looks like a generic LLM question ID that wasn't caught above
            if (question_id.startswith('next_level_question_') or 
                '_question_' in question_id or 
                question_id.startswith('mock_nl_')):  # Added check for mock_nl_ prefix
                logging.info(f"🛠️ TREATING AS LLM OR MOCK QUESTION DESPITE UNEXPECTED FORMAT: {question_id}")
                is_llm_question = True
                input_type = request.form.get('input_type', 'text')
                logging.info(f"✅ PROCESSING GENERIC QUESTION AS LLM/MOCK: {question_id} with input_type={input_type}")
                question = {
                    'input_type': input_type,
                    'id': question_id,
                    'question_id': question_id,
                    'type': 'next_level'
                }
            else:
                # It's a true invalid ID
                return jsonify({'success': False, 'error': 'Invalid question ID'})
        else:
            logging.info(f"✅ VALID REPOSITORY QUESTION: {question_id}, type={question.get('type')}")
    else:
        # For LLM questions, get input_type from form
        input_type = request.form.get('input_type', 'text')
        logging.info(f"✅ PROCESSING LLM QUESTION: {question_id} with input_type={input_type}")
        question = {
            'input_type': input_type,
            'id': question_id,
            'question_id': question_id,
            'type': 'next_level'
        }
    
    # Get appropriate answer value based on input type
    if 'is_multiselect' in request.form or question.get('input_type') == 'multiselect':
        # For multiselect, get all selected values as a list
        answer_value = request.form.getlist('answer')
        app.logger.info(f"Processing multiselect answer for question {question_id}: {answer_value}")
    else:
        # For regular inputs, get single value
        answer_value = request.form.get('answer')
        
        # Convert to appropriate type for number inputs
        if question.get('input_type') == 'number':
            try:
                answer_value = float(answer_value)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid number format'})
    
    # Submit the answer
    success, updated_profile = question_service.submit_answer(
        profile_id, question_id, answer_value
    )
    
    if not success:
        logging.error(f"Failed to save answer: {question_id}. Profile: {profile_id}")
        return jsonify({'success': False, 'error': 'Failed to save answer'})
    
    logging.info(f"Successfully saved answer for question: {question_id}. Profile: {profile_id}")
    
    # Log when a question is answered, including the question data if available
    question_data = None
    if not is_llm_question:
        question_data = question
    question_service.question_logger.log_question_answered(profile_id, question_id, answer_value, question_data)
    
    # Get updated completion and next question
    completion = question_service.get_profile_completion(profile_id)
    next_question, _ = question_service.get_next_question(profile_id)
    
    return jsonify({
        'success': True, 
        'completion': completion,
        'has_next_question': next_question is not None,
        'next_url': url_for('questions')
    })

@app.route('/profile/complete')
def profile_complete():
    """Profile completion summary page"""
    profile_id = session.get('profile_id')
    if not profile_id:
        return redirect(url_for('create_profile'))
    
    profile = profile_manager.get_profile(profile_id)
    if not profile:
        return redirect(url_for('create_profile'))
    
    # Get completion metrics and answered questions for rendering
    completion = question_service.get_profile_completion(profile_id)
    answered_questions = question_service.get_answered_questions(profile_id)
    
    # Render the completion page even if is_profile_complete would return false
    # This prevents redirect loops between questions and profile_complete
    return render_template(
        'profile_complete.html',
        profile=profile,
        completion=completion,
        answered_questions=answered_questions
    )

@app.route('/profile/switch', methods=['GET'])
def switch_profile():
    """Clear current profile from session"""
    session.pop('profile_id', None)
    return redirect(url_for('index'))

@app.route('/profile/data/<profile_id>')
def profile_data(profile_id):
    """API endpoint to view profile data (for debugging)"""
    # Verify profile ID matches session for security
    if profile_id != session.get('profile_id'):
        return jsonify({'error': 'Unauthorized access'})
    
    profile = profile_manager.get_profile(profile_id)
    if not profile:
        return jsonify({'error': 'Profile not found'})
    
    return jsonify(profile)

@app.route('/categories')
def list_categories():
    """List all question categories (for debugging)"""
    categories = {
        'demographics': question_repository.get_questions_by_category('demographics'),
        'financial_basics': question_repository.get_questions_by_category('financial_basics'),
        'assets_and_debts': question_repository.get_questions_by_category('assets_and_debts'),
        'special_cases': question_repository.get_questions_by_category('special_cases')
    }
    
    return jsonify({
        cat: [q['id'] for q in questions] 
        for cat, questions in categories.items()
    })

@app.route('/profiles', methods=['GET'])
def list_profiles():
    """List all profiles (admin view)"""
    profiles = profile_manager.get_all_profiles()
    return jsonify(profiles)

@app.route('/profile/versions/<profile_id>', methods=['GET'])
def list_profile_versions(profile_id):
    """List all versions of a profile (for debugging)"""
    # Verify profile ID matches session for security
    if profile_id != session.get('profile_id'):
        return jsonify({'error': 'Unauthorized access'})
        
    versions = profile_manager.get_profile_versions(profile_id)
    return jsonify({
        'profile_id': profile_id,
        'versions_count': len(versions),
        'versions': [{'version': v['version'], 'reason': v['reason'], 'created_at': v['created_at']} for v in versions]
    })

@app.route('/profile/analytics/<profile_id>', methods=['GET'])
def profile_analytics(profile_id):
    """Generate analytics for a profile"""
    # Verify profile ID matches session for security
    if profile_id != session.get('profile_id'):
        return jsonify({'error': 'Unauthorized access'})
    
    analytics = profile_analytics_service.generate_profile_analytics(profile_id)
    return jsonify(analytics)

@app.route('/profile/analytics/view/<profile_id>', methods=['GET'])
def view_profile_analytics(profile_id):
    """View analytics page for a profile"""
    # Verify profile ID matches session for security
    if profile_id != session.get('profile_id'):
        flash('Unauthorized access')
        return redirect(url_for('index'))
    
    profile = profile_manager.get_profile(profile_id)
    if not profile:
        flash('Profile not found')
        return redirect(url_for('index'))
    
    return render_template('profile_analytics.html', profile_id=profile_id)

@app.route('/profile/analytics/summary/<profile_id>', methods=['GET'])
def profile_analytics_summary(profile_id):
    """Get a summarized version of profile analytics for UI display"""
    # Verify profile ID matches session for security
    if profile_id != session.get('profile_id'):
        return jsonify({'error': 'Unauthorized access'})
    
    analytics = profile_analytics_service.generate_profile_analytics(profile_id)
    
    # Create a simplified version for UI display
    summary = {
        'profile_name': analytics.get('profile_name', 'Unknown'),
        'dimensions': analytics.get('dimensions', {}),
        'investment_profile': {
            'type': analytics.get('investment_profile', {}).get('type', 'Balanced'),
            'description': analytics.get('investment_profile', {}).get('description', '')
        },
        'financial_health': {
            'score': analytics.get('financial_health_score', {}).get('score', 0),
            'status': analytics.get('financial_health_score', {}).get('status', 'Unknown')
        },
        'behavioral_profile': analytics.get('behavioral_profile', {}),
        'key_insights': analytics.get('key_insights', [])[:3],
        'recommendations': analytics.get('recommendations', [])[:3]
    }
    
    # Log what we're sending to help with debugging
    logging.info(f"Sending analytics summary for profile {profile_id}")
    if 'behavioral_profile' in analytics:
        logging.info(f"Behavioral profile data: {analytics['behavioral_profile']}")
    
    return jsonify(summary)

@app.route('/profile/compare', methods=['POST'])
def compare_profiles():
    """Compare multiple profiles"""
    # This should be an admin/researcher only endpoint in production
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.json
    profile_ids = data.get('profile_ids', [])
    
    if not profile_ids or len(profile_ids) < 2:
        return jsonify({'error': 'Must provide at least two profile IDs to compare'}), 400
    
    comparison = profile_analytics_service.compare_profiles(profile_ids)
    return jsonify(comparison)

@app.route('/api/llm-status', methods=['GET'])
def llm_status():
    """Check the status of the LLM service"""
    return jsonify(Config.get_llm_status_message())

@app.context_processor
def inject_llm_status():
    """Inject LLM status into all templates"""
    return {'llm_status': Config.get_llm_status_message()}

# ----- Admin Dashboard Routes -----

@app.route('/admin')
@auth.login_required
def admin_dashboard():
    """Admin dashboard home page"""
    return render_template('admin/dashboard.html')

@app.route('/admin/profiles')
@auth.login_required
def admin_profiles():
    """Admin profiles overview"""
    profiles = profile_manager.get_all_profiles()
    return render_template('admin/profiles.html', profiles=profiles)

@app.route('/admin/profile/<profile_id>')
@auth.login_required
def admin_profile_detail(profile_id):
    """Admin view of a specific profile"""
    profile = profile_manager.get_profile(profile_id)
    if not profile:
        flash('Profile not found')
        return redirect(url_for('admin_profiles'))
    
    analytics = profile_analytics_service.generate_profile_analytics(profile_id)
    return render_template('admin/profile_detail.html', profile=profile, analytics=analytics)

@app.route('/admin/insights')
@auth.login_required
def admin_insights():
    """View of extracted insights across profiles"""
    profiles = profile_manager.get_all_profiles()
    
    # Collect all LLM-extracted insights
    all_insights = []
    for profile in profiles:
        profile_id = profile.get('id')
        profile_name = profile.get('name', 'Unknown')
        
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        # Look for insight answers (those ending with _insights)
        for question_id, answer in answers.items():
            if question_id.endswith('_insights') and isinstance(answer, dict):
                insight_entry = {
                    'profile_id': profile_id,
                    'profile_name': profile_name,
                    'question_id': question_id,
                    'related_question_id': question_id.replace('_insights', ''),
                    'timestamp': answer.get('timestamp', 'Unknown'),
                    'data': answer
                }
                all_insights.append(insight_entry)
    
    return render_template('admin/insights.html', insights=all_insights)

@app.route('/admin/metrics')
@auth.login_required
def admin_metrics():
    """System-wide metrics dashboard"""
    profiles = profile_manager.get_all_profiles()
    
    # Calculate overall metrics
    metrics = {
        'total_profiles': len(profiles),
        'total_answers': 0,
        'core_questions_answered': 0,
        'next_level_questions_answered': 0,
        'profiles_by_completion': {
            'not_started': 0,      # 0-20%
            'early_stage': 0,      # 20-40%
            'mid_stage': 0,        # 40-60%
            'late_stage': 0,       # 60-80%
            'completed': 0         # 80-100%
        },
        'category_completion': {
            'demographics': 0,
            'financial_basics': 0,
            'assets_and_debts': 0,
            'special_cases': 0
        },
        'investment_profiles': {
            'Conservative': 0,
            'Moderately Conservative': 0,
            'Balanced': 0,
            'Moderately Aggressive': 0,
            'Aggressive': 0
        },
        'llm_usage': {
            'total_insights_generated': 0,
            'total_next_level_questions': 0
        }
    }
    
    # Process each profile
    for profile in profiles:
        # Count total answers
        answers = profile.get('answers', [])
        metrics['total_answers'] += len(answers)
        
        # Separate core vs. next-level questions
        for answer in answers:
            question_id = answer.get('question_id', '')
            if question_id.endswith('_insights'):
                metrics['llm_usage']['total_insights_generated'] += 1
            elif '_next_level_' in question_id or 'next_level_' in question_id:
                metrics['next_level_questions_answered'] += 1
            else:
                metrics['core_questions_answered'] += 1
        
        # Generate analytics and classify profile
        try:
            analytics = profile_analytics_service.generate_profile_analytics(profile['id'])
            
            # Categorize by completion percentage
            completion = analytics.get('financial_health_score', {}).get('score', 0)
            if completion < 20:
                metrics['profiles_by_completion']['not_started'] += 1
            elif completion < 40:
                metrics['profiles_by_completion']['early_stage'] += 1
            elif completion < 60:
                metrics['profiles_by_completion']['mid_stage'] += 1
            elif completion < 80:
                metrics['profiles_by_completion']['late_stage'] += 1
            else:
                metrics['profiles_by_completion']['completed'] += 1
                
            # Count investment profiles
            profile_type = analytics.get('investment_profile', {}).get('type', 'Balanced')
            metrics['investment_profiles'][profile_type] = metrics['investment_profiles'].get(profile_type, 0) + 1
                
            # Add to category completion totals
            for category, completion in analytics.get('core', {}).get('by_category', {}).items():
                metrics['category_completion'][category] = metrics['category_completion'].get(category, 0) + completion
        except Exception as e:
            logging.error(f"Error processing analytics for profile {profile['id']}: {str(e)}")
    
    # Calculate averages for category completion
    if profiles:
        for category in metrics['category_completion']:
            metrics['category_completion'][category] /= len(profiles)
            metrics['category_completion'][category] = round(metrics['category_completion'][category], 1)
    
    return render_template('admin/metrics.html', metrics=metrics)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and user['password'] == password:  # WARNING: Never store passwords in plain text
            session['username'] = username
            session['profile_id'] = user['profile_id']  # Assuming you have a profile_id in the users table

            # Fetch user data here
            user_data = fetch_user_data(user['profile_id'])  # Use the function from RAG/fetch_user_details.py
            session['user_data'] = user_data

            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))  # Redirect to the main page
        else:
            flash('Invalid username or password', 'danger')
            return render_template('login.html')

    # Render login form (for GET requests)
    return render_template('login.html')

@app.route('/profile')
def profile():
    """User profile route."""
    if 'profile_id' in session:
        profile_id = session['profile_id']
        db_manager = DatabaseProfileManager()
        profile = db_manager.get_profile(profile_id)

        if profile:
            return render_template('profile.html', profile=profile)  # Pass profile to template
        else:
            flash('Profile not found.')
            return redirect(url_for('login'))
    else:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        # *** Replace with your actual user creation logic ***
        user = create_user(username, password, email)  # Example user creation function

        if user:
            # User creation successful, now create a profile
            db_manager = DatabaseProfileManager()
            profile = db_manager.create_profile(username, email)  # Create profile

            # *** Store profile_id in your user table ***
            user.profile_id = profile['id']
            update_user_profile_id(user, profile['id'])  # Example update function

            flash('Registration successful. Please log in.')
            return redirect(url_for('login'))
        else:
            # User creation failed
            flash('Registration failed. Please try again.')
            return render_template('register.html')

    # Render registration form (for GET requests)
    return render_template('register.html')

@app.route('/rag', methods=['GET'])
def rag_page():
    """Render the RAG query page."""
    return render_template('rag_query.html')

@app.route('/rag/query', methods=['POST'])
def rag_query():
    """Handle user queries using the RAG model."""
    user_query = request.json.get('query')  # Get the query from the request
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    # Get user data from the session
    user_data = session.get('user_data', {})  # Default to an empty dictionary if no user data

    # Prepare the input for the graph
    inputs = {"query": user_query, "user_data": user_data} # Pass user_data to the graph
    
    # Invoke the graph with the user's query
    result = graph.invoke(inputs)

    # Return the generated response
    return jsonify({
        'generation': result.get('generation', 'No response generated.'),
        'question': result.get('question', user_query)  # Return the original question if needed
    })
    # Return the generated response
 

if __name__ == '__main__':
    # Run the app with the configured debug setting
    app.run(debug=Config.DEBUG)