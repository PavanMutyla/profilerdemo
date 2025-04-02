import logging
from datetime import datetime
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from services.llm_service import LLMService
from models.profile_understanding import ProfileUnderstandingCalculator

class QuestionLogger:
    """
    Dedicated logger for tracking question lifecycle events.
    Creates and maintains logs of all question events for each profile.
    """
    
    def __init__(self, log_dir=None):
        """Initialize the question logger with the specified log directory"""
        # Use a relative path or user's home directory by default
        self.log_dir = log_dir or os.path.join(os.path.expanduser("~"), "profiler_question_logs")
        
        # Create the log directory if it doesn't exist
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        except Exception as e:
            # Fallback to current directory if we can't create the preferred directory
            self.log_dir = "question_logs"
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            logging.warning(f"Could not create log directory at {log_dir}, using {self.log_dir} instead: {str(e)}")
        
        # Dictionary to store in-memory logs before writing to file
        self.question_logs = {}
    
    def _ensure_profile_log(self, profile_id):
        """Ensure the profile's log file exists and is properly initialized"""
        profile_log_dir = os.path.join(self.log_dir, profile_id)
        
        if not os.path.exists(profile_log_dir):
            os.makedirs(profile_log_dir)
            
        return profile_log_dir
    
    def log_question_generated(self, profile_id, question_id, question_data):
        """Log when a question is generated (by system or LLM)"""
        log_entry = {
            "event": "question_generated",
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "question_type": question_data.get('type', 'unknown'),
            "question_category": question_data.get('category', 'unknown'),
            "question_text": question_data.get('text', ''),
            "input_type": question_data.get('input_type', 'unknown'),
            "is_llm_generated": bool('llm_next_level' in question_id or 'gen_question' in question_id),
            "is_fallback": bool('fallback' in question_id),
            "is_required": question_data.get('required', False),
            "order": question_data.get('order', None),
        }
        
        self._append_to_question_log(profile_id, question_id, log_entry)
    
    def log_question_displayed(self, profile_id, question_id, question_data):
        """Log when a question is displayed to the user"""
        log_entry = {
            "event": "question_displayed",
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "question_text": question_data.get('text', ''),
        }
        
        self._append_to_question_log(profile_id, question_id, log_entry)
    
    def log_question_answered(self, profile_id, question_id, answer_value, question_data=None):
        """Log when a question is answered by the user"""
        log_entry = {
            "event": "question_answered",
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "answer_value": self._format_answer_for_log(answer_value),
        }
        
        if question_data:
            log_entry["question_text"] = question_data.get('text', '')
            log_entry["question_type"] = question_data.get('type', 'unknown')
            log_entry["question_category"] = question_data.get('category', 'unknown')
        
        self._append_to_question_log(profile_id, question_id, log_entry)
    
    def _format_answer_for_log(self, answer_value):
        """Format the answer value to ensure it can be properly serialized"""
        # If the answer is an iterable but not a string or dict, convert to list
        if hasattr(answer_value, '__iter__') and not isinstance(answer_value, (str, dict)):
            return list(answer_value)
        return answer_value
    
    def _append_to_question_log(self, profile_id, question_id, log_entry):
        """Append a log entry to the appropriate question log"""
        # Initialize profile logs if not exist
        if profile_id not in self.question_logs:
            self.question_logs[profile_id] = {}
            
        # Initialize question log if not exist
        if question_id not in self.question_logs[profile_id]:
            self.question_logs[profile_id][question_id] = []
            
        # Add the log entry
        self.question_logs[profile_id][question_id].append(log_entry)
        
        # Write the updated logs to disk
        self._write_logs_to_disk(profile_id)
    
    def _write_logs_to_disk(self, profile_id):
        """Write the question logs for a profile to disk"""
        profile_log_dir = self._ensure_profile_log(profile_id)
        
        # Write a comprehensive log file with all questions
        all_questions_log_path = os.path.join(profile_log_dir, "all_questions.json")
        
        try:
            with open(all_questions_log_path, 'w') as f:
                json.dump(self.question_logs[profile_id], f, indent=2)
        except Exception as e:
            logging.error(f"Error writing question logs for profile {profile_id}: {str(e)}")
            
        # Write a summary file for easier analysis
        summary_log_path = os.path.join(profile_log_dir, "question_summary.json")
        
        try:
            summary = {}
            for question_id, events in self.question_logs[profile_id].items():
                # Get the most recent version of each event type
                latest_events = {}
                for event in events:
                    event_type = event["event"]
                    latest_events[event_type] = event
                
                # Create the summary entry
                summary[question_id] = {
                    "question_id": question_id,
                    "question_text": (latest_events.get("question_generated", {}).get("question_text") or
                                     latest_events.get("question_displayed", {}).get("question_text") or
                                     latest_events.get("question_answered", {}).get("question_text") or ""),
                    "question_type": latest_events.get("question_generated", {}).get("question_type", "unknown"),
                    "category": latest_events.get("question_generated", {}).get("question_category", "unknown"),
                    "is_llm_generated": latest_events.get("question_generated", {}).get("is_llm_generated", False),
                    "is_fallback": latest_events.get("question_generated", {}).get("is_fallback", False),
                    "generated_at": latest_events.get("question_generated", {}).get("timestamp", None),
                    "displayed_at": latest_events.get("question_displayed", {}).get("timestamp", None),
                    "answered_at": latest_events.get("question_answered", {}).get("timestamp", None),
                    "answer_value": latest_events.get("question_answered", {}).get("answer_value", None),
                    "lifecycle_complete": all(e in latest_events for e in ["question_generated", "question_displayed", "question_answered"]),
                }
            
            with open(summary_log_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Generate an HTML summary for easier viewing
            self._generate_html_summary(profile_id, summary)
                
        except Exception as e:
            logging.error(f"Error writing question summary log for profile {profile_id}: {str(e)}")
            
    def _generate_html_summary(self, profile_id, summary):
        """Generate an HTML summary of questions for easier analysis"""
        profile_log_dir = self._ensure_profile_log(profile_id)
        html_path = os.path.join(profile_log_dir, "question_report.html")
        
        try:
            # Sort questions by types and the time they were first displayed
            core_questions = []
            goal_questions = []
            next_level_questions = []
            behavioral_questions = []
            other_questions = []
            
            for qid, q_data in summary.items():
                if q_data["question_type"] == "core":
                    core_questions.append(q_data)
                elif q_data["question_type"] == "goal":
                    goal_questions.append(q_data)
                elif q_data["question_type"] == "next_level":
                    next_level_questions.append(q_data)
                elif q_data["question_type"] == "behavioral":
                    behavioral_questions.append(q_data)
                else:
                    other_questions.append(q_data)
            
            # Sort each group by displayed_at timestamp
            for q_list in [core_questions, goal_questions, next_level_questions, behavioral_questions, other_questions]:
                q_list.sort(key=lambda q: q.get("displayed_at", "9999-99-99") or "9999-99-99")
            
            # Generate HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Question Report for Profile {profile_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .timestamp {{ font-size: 0.8em; color: #666; }}
                    .duplicated {{ background-color: #ffe6e6; }}
                    .section {{ margin-bottom: 40px; }}
                    .question-text {{ font-weight: bold; }}
                    .llm-generated {{ background-color: #e6f7ff; }}
                    .fallback {{ background-color: #fff2e6; }}
                    .lifecycle-incomplete {{ background-color: #fffde6; }}
                    .durations {{ font-size: 0.85em; color: #444; }}
                </style>
            </head>
            <body>
                <h1>Question Report for Profile {profile_id}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Count</th>
                        </tr>
                        <tr>
                            <td>Total Questions</td>
                            <td>{len(summary)}</td>
                        </tr>
                        <tr>
                            <td>Core Questions</td>
                            <td>{len(core_questions)}</td>
                        </tr>
                        <tr>
                            <td>Goal Questions</td>
                            <td>{len(goal_questions)}</td>
                        </tr>
                        <tr>
                            <td>Next-Level Questions</td>
                            <td>{len(next_level_questions)}</td>
                        </tr>
                        <tr>
                            <td>Behavioral Questions</td>
                            <td>{len(behavioral_questions)}</td>
                        </tr>
                        <tr>
                            <td>LLM-Generated Questions</td>
                            <td>{len([q for q in summary.values() if q.get("is_llm_generated", False)])}</td>
                        </tr>
                        <tr>
                            <td>Fallback Questions</td>
                            <td>{len([q for q in summary.values() if q.get("is_fallback", False)])}</td>
                        </tr>
                    </table>
                </div>
            """
            
            # Function to generate a table for a question set
            def generate_question_table(title, questions):
                if not questions:
                    return f"<div class='section'><h2>{title}</h2><p>No questions in this category.</p></div>"
                
                # Find duplicate questions (by text)
                text_count = {}
                for q in questions:
                    text = q.get("question_text", "")
                    text_count[text] = text_count.get(text, 0) + 1
                
                html = f"""
                <div class="section">
                    <h2>{title} ({len(questions)})</h2>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Question</th>
                            <th>Answer</th>
                            <th>Timestamps</th>
                            <th>Details</th>
                        </tr>
                """
                
                for q in questions:
                    # Calculate time between events
                    generated_time = None
                    displayed_time = None
                    answered_time = None
                    
                    if q.get("generated_at"):
                        generated_time = datetime.fromisoformat(q["generated_at"])
                    if q.get("displayed_at"):
                        displayed_time = datetime.fromisoformat(q["displayed_at"])
                    if q.get("answered_at"):
                        answered_time = datetime.fromisoformat(q["answered_at"])
                    
                    durations = []
                    if generated_time and displayed_time:
                        gen_to_display = (displayed_time - generated_time).total_seconds()
                        durations.append(f"Gen→Display: {gen_to_display:.1f}s")
                    
                    if displayed_time and answered_time:
                        display_to_answer = (answered_time - displayed_time).total_seconds()
                        durations.append(f"Display→Answer: {display_to_answer:.1f}s")
                    
                    # Check if this question text is duplicated
                    is_duplicated = text_count.get(q.get("question_text", ""), 0) > 1
                    is_llm = q.get("is_llm_generated", False)
                    is_fallback = q.get("is_fallback", False)
                    is_complete = q.get("lifecycle_complete", False)
                    
                    # Determine row CSS classes
                    row_classes = []
                    if is_duplicated:
                        row_classes.append("duplicated")
                    if is_llm:
                        row_classes.append("llm-generated")
                    if is_fallback:
                        row_classes.append("fallback")
                    if not is_complete:
                        row_classes.append("lifecycle-incomplete")
                    
                    row_class = f"class='{' '.join(row_classes)}'" if row_classes else ""
                    
                    # Format the answer value for display
                    answer_value = q.get("answer_value")
                    if isinstance(answer_value, list):
                        answer_display = ", ".join(str(item) for item in answer_value)
                    elif answer_value is None:
                        answer_display = "<em>Not answered</em>"
                    else:
                        answer_display = str(answer_value)
                    
                    html += f"""
                        <tr {row_class}>
                            <td>{q["question_id"]}</td>
                            <td class="question-text">{q["question_text"]}</td>
                            <td>{answer_display}</td>
                            <td>
                                <div class="timestamp">Generated: {q.get("generated_at", "N/A")}</div>
                                <div class="timestamp">Displayed: {q.get("displayed_at", "N/A")}</div>
                                <div class="timestamp">Answered: {q.get("answered_at", "N/A")}</div>
                                <div class="durations">{" | ".join(durations)}</div>
                            </td>
                            <td>
                                <div>Type: {q["question_type"]}</div>
                                <div>Category: {q["category"]}</div>
                                <div>LLM Generated: {str(is_llm)}</div>
                                <div>Fallback: {str(is_fallback)}</div>
                                <div>Lifecycle Complete: {str(is_complete)}</div>
                            </td>
                        </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
                return html
            
            # Generate tables for each question type
            html_content += generate_question_table("Core Questions", core_questions)
            html_content += generate_question_table("Goal Questions", goal_questions)
            html_content += generate_question_table("Next-Level Questions", next_level_questions)
            html_content += generate_question_table("Behavioral Questions", behavioral_questions)
            html_content += generate_question_table("Other Questions", other_questions)
            
            # Finish the HTML document
            html_content += """
            </body>
            </html>
            """
            
            # Write to file
            with open(html_path, 'w') as f:
                f.write(html_content)
                
            logging.info(f"Generated HTML question report for profile {profile_id}: {html_path}")
            
        except Exception as e:
            logging.error(f"Error generating HTML question report for profile {profile_id}: {str(e)}")

class QuestionService:
    """
    Service for handling question flow logic and answer management.
    Determines next questions and processes answer submissions.
    """
    
    def __init__(self, question_repository, profile_manager, llm_service=None):
        """
        Initialize the question service with required dependencies.
        
        Args:
            question_repository: Repository for question definitions
            profile_manager: Manager for profile operations
            llm_service: Optional LLMService for generating dynamic questions
        """
        self.question_repository = question_repository
        self.profile_manager = profile_manager
        self.llm_service = llm_service or LLMService()
        
        # Cache for dynamically generated questions
        self.dynamic_questions_cache = {}
        
        # Understanding level calculator
        self.understanding_calculator = ProfileUnderstandingCalculator()
        
        # Initialize the question logger
        self.question_logger = QuestionLogger()
        
        logging.basicConfig(level=logging.INFO)
        
    def get_next_question(self, profile_id):
        """
        Get the next appropriate question for a profile.
        Prioritizes core questions, followed by goal-setting questions, then next-level and behavioral questions.
        Special priority for business value questions for business owners.
        
        Args:
            profile_id (str): ID of the profile
            
        Returns:
            dict: Question definition or None if all required questions are complete
            dict: Profile object for context
        """
        # Load profile
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            logging.error(f"Profile {profile_id} not found")
            return None, None
        
        # Check for special cases (business owner and real estate value)
        special_case_question = self._check_special_case_questions(profile)
        if special_case_question:
            question_id = special_case_question.get('id')
            logging.info(f"Prioritizing special case question ({question_id}) in profile {profile_id}")
            # Log the question generation
            self._log_question_generation(profile_id, special_case_question)
            return special_case_question, profile
            
        # Check for unanswered core questions from repository
        next_core_question = self.question_repository.get_next_question(profile)
        
        # Log core question generation if present
        if next_core_question:
            self._log_question_generation(profile_id, next_core_question)
        
        # If all core questions are answered, check for goal questions
        if not next_core_question and self._is_ready_for_goals(profile):
            goal_question = self._get_goal_question(profile)
            
            if goal_question:
                # If this is the emergency fund calculation question, personalize it with the actual calculation
                if goal_question.get('id') == 'goals_emergency_fund_calculation':
                    logging.info(f"EMERGENCY FUND CALCULATION: About to calculate for question {goal_question.get('id')}")
                    
                    # Calculate emergency fund recommendations
                    emergency_fund_data = self._calculate_emergency_fund_recommendations(profile)
                    
                    # Debug: Log raw calculation result to make sure we're getting data
                    logging.info(f"DEBUG: Emergency fund calculation raw result: {emergency_fund_data}")
                    
                    if emergency_fund_data:
                        # Format the numbers with commas for better readability
                        monthly_expenses = f"₹{emergency_fund_data['monthly_expenses']:,.0f}"
                        minimum_fund = f"₹{emergency_fund_data['minimum_fund']:,.0f}"
                        recommended_fund = f"₹{emergency_fund_data['recommended_fund']:,.0f}"
                        
                        # Check if we're using default values
                        is_default = emergency_fund_data.get('is_default', False)
                        data_source = emergency_fund_data.get('data_source', "Your Data" if not is_default else "Example")
                        
                        # Create enhanced educational content with the actual calculations
                        if is_default:
                            educational_content = f"""
                                <h3>Example Emergency Fund Calculation</h3>
                                
                                <p>Since you haven't provided your monthly expenses yet, here's an example calculation 
                                based on a monthly expense of {monthly_expenses}. This demonstrates how emergency funds 
                                are calculated in India:</p>
                            """
                        else:
                            educational_content = f"""
                                <h3>Your Personalized Emergency Fund Calculation</h3>
                                
                                <p>Based on the financial guidelines for India and your reported monthly expenses, 
                                we've calculated recommended emergency fund targets for your specific situation:</p>
                            """
                        
                        # Create calculation_details for the calculation box - using properly formatted HTML with data source
                        calculation_details = f"""
                            <div class="calculation-item">
                                <div class="calculation-label">Your Monthly Expenses:</div>
                                <div class="calculation-value">{monthly_expenses}</div>
                            </div>
                            <div class="calculation-item">
                                <div class="calculation-label">Minimum Recommended (6 months):</div>
                                <div class="calculation-value">{minimum_fund}</div>
                            </div>
                            <div class="calculation-item">
                                <div class="calculation-label">Ideal Recommended (9 months):</div>
                                <div class="calculation-value">{recommended_fund}</div>
                            </div>
                        """
                        
                        # Log the calculation details to help with debugging
                        logging.info(f"CALCULATION DETAILS ADDED: Monthly Expenses={monthly_expenses}, " +
                                    f"Minimum={minimum_fund}, Recommended={recommended_fund}, " +
                                    f"Is Default={is_default}")
                        
                        # Explicitly add these properties to the question
                        goal_question['educational_content'] = educational_content
                        goal_question['calculation_details'] = calculation_details
                        
                        # Debug: Log that we've successfully added calculation_details
                        logging.info(f"DEBUG: Successfully added calculation_details to question {goal_question.get('id')}")
                        logging.info(f"DEBUG: Question now has keys: {goal_question.keys()}")
                        
                        # We'll still update the help_text as a fallback
                        help_text = (
                            f"Current Monthly Expenses: {monthly_expenses}\n"
                            f"Minimum Recommended (6 months): {minimum_fund}\n"
                            f"Ideal Recommended (9 months): {recommended_fund}"
                        )
                        
                        goal_question['help_text'] = help_text
                    else:
                        # Log error if calculation returned None
                        logging.error(f"ERROR: Emergency fund calculation returned None for profile {profile_id}")
                        # Provide default value since calculation failed
                        goal_question['calculation_details'] = """
                            <div class="calculation-item">
                                <div class="calculation-label">Your Monthly Expenses (Example):</div>
                                <div class="calculation-value">₹50,000</div>
                            </div>
                            <div class="calculation-item">
                                <div class="calculation-label">Minimum Recommended (6 months):</div>
                                <div class="calculation-value">₹300,000</div>
                            </div>
                            <div class="calculation-item">
                                <div class="calculation-label">Ideal Recommended (9 months):</div>
                                <div class="calculation-value">₹450,000</div>
                            </div>
                        """
                
                # If this is the emergency fund amount question, set a default value based on calculations
                elif goal_question.get('id') == 'goals_emergency_fund_amount':
                    # Calculate emergency fund recommendations
                    emergency_fund_data = self._calculate_emergency_fund_recommendations(profile)
                    
                    if emergency_fund_data:
                        # Add a suggested value (the ideal 9-month amount)
                        goal_question['suggested_value'] = emergency_fund_data['recommended_fund']
                        
                        # Format amount with commas
                        formatted_amount = f"₹{emergency_fund_data['recommended_fund']:,.0f}"
                        
                        # Check if we're using default values
                        is_default = emergency_fund_data.get('is_default', False)
                        
                        # Update help text to include the suggestion
                        if is_default:
                            help_text = (
                                f"In India, 6-9 months of expenses is recommended. For demonstration purposes, "
                                f"we've suggested {formatted_amount} based on a sample monthly expense. "
                                f"Please enter your actual target amount."
                            )
                        else:
                            help_text = (
                                f"In India, 6-9 months of expenses is recommended. Based on your monthly expenses, "
                                f"we've suggested {formatted_amount} (9 months of expenses). "
                                f"You can override this with your own target amount that suits your specific situation."
                            )
                        
                        # Log the suggestion
                        logging.info(f"Suggesting emergency fund amount: {formatted_amount}, Is Default: {is_default}")
                        
                        goal_question['help_text'] = help_text
                
                logging.info(f"Presenting goal question to profile {profile_id}: {goal_question.get('id')}")
                # Log the goal question generation
                self._log_question_generation(profile_id, goal_question)
                return goal_question, profile
        
        # If all core questions are answered, check for pending goal questions first
        goal_question = None
        if not next_core_question:
            if self._has_pending_goal_questions(profile):
                goal_question = self._get_goal_question(profile)
                if goal_question:
                    logging.info(f"Found pending goal question {goal_question.get('id')} for profile {profile_id}")
                    # Log the goal question generation
                    self._log_question_generation(profile_id, goal_question)
                    return goal_question, profile
            
            # Count how many next-level questions have been answered
            answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
            next_level_answered_count = 0
            for q_id in answered_ids:
                if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                    next_level_answered_count += 1
                # Count dynamic/LLM generated questions
                elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                    next_level_answered_count += 1
                
            # Log how many next-level questions have been answered
            logging.info(f"Profile {profile_id} has answered {next_level_answered_count}/5 required next-level questions")
                    
            # If no pending goal questions, check for next-level questions
            if self._is_ready_for_next_level(profile):
                next_level_question = self._get_next_level_question(profile)
                
                # Personalize the question if it contains placeholders
                if next_level_question and "text" in next_level_question:
                    next_level_question["text"] = self.llm_service.personalize_question_text(
                        next_level_question["text"], profile
                    )
                    
                # If we have a next-level question, return it
                if next_level_question:
                    logging.info(f"Presenting next-level question ({next_level_question.get('id')}) to profile {profile_id}")
                    # Log the next-level question generation
                    self._log_question_generation(profile_id, next_level_question)
                    return next_level_question, profile
                
                # If no next-level questions, check if there are any new goal questions to ask
                if self._is_ready_for_goals(profile):
                    goal_question = self._get_goal_question(profile)
                    if goal_question:
                        logging.info(f"Returning to goal questions after next-level questions for profile {profile_id}")
                        # Log the goal question generation
                        self._log_question_generation(profile_id, goal_question)
                        return goal_question, profile
                
                # If no next-level or goal questions are available and we've answered enough questions,
                # check if we should transition to behavioral questions
                if self._is_ready_for_behavioral(profile):
                    logging.info(f"Profile {profile_id} is ready for behavioral questions")
                    behavioral_question = self._get_behavioral_question(profile)
                    
                    if behavioral_question:
                        logging.info(f"Presenting behavioral question ({behavioral_question.get('id')}) to profile {profile_id}")
                        # Log the behavioral question generation
                        self._log_question_generation(profile_id, behavioral_question)
                        return behavioral_question, profile
        
        return next_core_question, profile
        
    def _check_special_case_questions(self, profile):
        """
        Check for special case questions based on user information.
        Prioritizes business value and real estate value questions for specific user types.
        
        Args:
            profile (dict): User profile
            
        Returns:
            dict: Special case question or None
        """
        # Get answers
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        answered_ids = set(answers.keys())
        
        # Check if user is a business owner
        if 'demographics_employment_type' in answers and answers['demographics_employment_type'] == 'Business owner':
            # Check if business value question has been answered already
            if 'special_cases_business_value' not in answered_ids:
                # Get the business value question
                business_value_question = self.question_repository.get_question('special_cases_business_value')
                if business_value_question:
                    logging.info(f"Business owner detected in profile {profile.get('id')}, triggering business value question")
                    return business_value_question
                else:
                    logging.warning(f"Business value question not found in repository for profile {profile.get('id')}")
        
        # Check for real estate value question if the user has a housing loan
        if 'assets_debts_housing_loan' in answers and answers['assets_debts_housing_loan'] == 'Yes':
            # Check if real estate value question has been answered already
            if 'special_cases_real_estate_value' not in answered_ids:
                # Get the real estate value question
                real_estate_question = self.question_repository.get_question('special_cases_real_estate_value')
                if real_estate_question:
                    logging.info(f"Housing loan detected in profile {profile.get('id')}, triggering real estate value question")
                    return real_estate_question
                else:
                    logging.warning(f"Real estate value question not found in repository for profile {profile.get('id')}")
        
        return None
        
    def _is_ready_for_goals(self, profile):
        """
        Check if the profile is ready for goal questions.
        This happens after completing most core questions.
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if ready for goal questions
        """
        profile_id = profile.get('id')
        
        # Get answered question IDs
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        
        # Check if we've already started goals questions
        goal_questions_started = any(q_id.startswith("goals_") for q_id in answered_ids)
        if goal_questions_started:
            logging.info(f"Goal questions already started for profile {profile_id}")
            return True
            
        # Calculate core completion percentage
        core_questions = self.question_repository.get_core_questions()
        required_core = [q for q in core_questions if q.get('required', False)]
        
        if not required_core:
            logging.warning(f"No required core questions found for profile {profile_id}")
            return False
            
        answered_required = [q for q in required_core if q.get('id') in answered_ids]
        core_completion = (len(answered_required) / len(required_core)) * 100
        
        # Ready for goals when core is mostly complete 
        is_ready = core_completion >= 80
        
        if is_ready:
            logging.info(f"QUESTION TRANSITION: Profile {profile_id} is ready for goal questions (core completion: {core_completion:.1f}%)")
        else:
            logging.info(f"GOALS CHECK: Profile {profile_id} not ready for goals yet (core completion: {core_completion:.1f}%)")
            
        return is_ready
    
    def _has_pending_goal_questions(self, profile):
        """
        Check if there are pending goal questions that need to be answered.
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if there are pending goal questions
        """
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        # Check if we have the goals confirmation answer
        if 'goals_confirmation' in answers:
            return False
            
        # Check if we've started on goal questions
        goal_questions_started = any(q_id.startswith("goals_") for q_id in answers.keys())
        
        # If we've started and not finished, we have pending questions
        return goal_questions_started
    
    def _get_goal_question(self, profile):
        """
        Get the next appropriate goal question.
        Follows a progressive flow through financial security hierarchy:
        1. Emergency fund
        2. Insurance
        3. Other financial goals
        
        Args:
            profile (dict): User profile
            
        Returns:
            dict: Next goal question or None
        """
        profile_id = profile.get('id')
        logging.info(f"DEBUG: Getting next goal question for profile {profile_id}")
        
        # Get answers
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        answered_ids = set(answers.keys())
        
        # Debug: log answered questions
        logging.info(f"DEBUG: User has answered these goal questions: {[qid for qid in answered_ids if qid.startswith('goals_')]}")
        
        # Check if we need to handle emergency fund questions specially
        should_skip_emergency_fund = False
        has_adequate_fund = False
        
        # Only check for emergency fund adequacy if we haven't already asked about it
        if 'goals_emergency_fund_exists' not in answered_ids:
            # Check if user already has adequate emergency fund (using savings/expenses calculation)
            has_adequate_fund = self._has_adequate_emergency_fund(profile)
            if has_adequate_fund:
                logging.info(f"EMERGENCY FUND: User already has adequate emergency fund, will skip related questions")
                should_skip_emergency_fund = True
        
        # List of emergency fund related question IDs to skip if user has adequate funds
        emergency_fund_question_ids = [
            'goals_emergency_fund_exists', 
            'goals_emergency_fund_months',
            'goals_emergency_fund_target', 
            'goals_emergency_fund_amount',
            'goals_emergency_fund_timeframe',
            'goals_emergency_fund_education',
            'goals_emergency_fund_calculation'
        ]
        
        # Check for dependent questions based on previous answers
        dependent_questions = self.question_repository.get_dependent_questions(profile)
        
        # Filter out emergency fund questions if needed
        if should_skip_emergency_fund:
            goal_dependent_questions = [q for q in dependent_questions 
                                      if q.get('type') == 'goal' 
                                      and q.get('id') not in answered_ids
                                      and q.get('id') not in emergency_fund_question_ids]
            logging.info(f"DEBUG: Filtered out emergency fund dependent questions, {len(dependent_questions) - len(goal_dependent_questions)} questions removed")
        else:
            goal_dependent_questions = [q for q in dependent_questions 
                                      if q.get('type') == 'goal' 
                                      and q.get('id') not in answered_ids]
        
        if goal_dependent_questions:
            # Sort by the dependency order to maintain proper flow
            sorted_dependent = sorted(goal_dependent_questions, key=lambda q: q.get('order', 999))
            next_q = sorted_dependent[0]
            logging.info(f"DEBUG: Next dependent question: {next_q.get('id')}")
            return next_q
        
        # If no dependent questions, check for the next goal question in sequence
        goal_questions = self.question_repository.get_questions_by_type('goal')
        
        # Filter out emergency fund questions if needed
        if should_skip_emergency_fund:
            unanswered_goal_questions = [q for q in goal_questions 
                                        if q.get('id') not in answered_ids
                                        and 'depends_on' not in q
                                        and q.get('id') not in emergency_fund_question_ids]
            logging.info(f"DEBUG: Filtered out emergency fund questions from main sequence, skipping related questions")
        else:
            unanswered_goal_questions = [q for q in goal_questions 
                                        if q.get('id') not in answered_ids
                                        and 'depends_on' not in q]
        
        if unanswered_goal_questions:
            # Sort by the question order to maintain proper flow
            sorted_questions = sorted(unanswered_goal_questions, key=lambda q: q.get('order', 999))
            next_q = sorted_questions[0]
            logging.info(f"DEBUG: Next regular goal question: {next_q.get('id')}")
            return next_q
            
        # No more goal questions to ask
        logging.info(f"DEBUG: No more goal questions to ask for profile {profile_id}")
        return None
        
    def _has_adequate_emergency_fund(self, profile):
        """
        Determine if the user has an adequate emergency fund based on answers.
        For India, the standard is 6-9 months (vs 3-6 months globally).
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if emergency fund is adequate
        """
        # Debug: Log checking emergency fund for profile
        profile_id = profile.get('id')
        logging.info(f"DEBUG: Checking if profile {profile_id} has adequate emergency fund")
        
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        # Log all emergency fund related answers to debug
        for key in ['goals_emergency_fund_exists', 'goals_emergency_fund_months']:
            if key in answers:
                logging.info(f"DEBUG: Found answer for {key}: {answers[key]}")
        
        # ONLY check direct question about emergency fund existence and adequacy
        if 'goals_emergency_fund_exists' in answers:
            fund_exists = answers['goals_emergency_fund_exists']
            logging.info(f"DEBUG: Emergency fund exists? {fund_exists}")
            
            if fund_exists == 'Yes':
                # Check months of coverage
                if 'goals_emergency_fund_months' in answers:
                    months = answers['goals_emergency_fund_months']
                    logging.info(f"DEBUG: Emergency fund covers {months} of expenses")
                    
                    # Using India standard: 5+ months is considered adequate (including 5-6 months option)
                    is_adequate = months in ['5-6 months', '6-9 months', 'More than 9 months']
                    logging.info(f"EMERGENCY FUND ADEQUACY CHECK: {months} coverage is {'adequate' if is_adequate else 'inadequate'} for India standards")
                    return is_adequate
            
            logging.info("DEBUG: Emergency fund exists but coverage is inadequate or not specified")
            return False
            
        # If we don't have an explicit answer about emergency fund existence,
        # assume the user needs to build one (don't try to calculate from savings)
        logging.info("DEBUG: No explicit answer about emergency fund existence, assuming user needs to establish one")
        return False
        
    def _calculate_emergency_fund_recommendations(self, profile):
        """
        Calculate recommended emergency fund amounts based on monthly expenses.
        
        Args:
            profile (dict): User profile
            
        Returns:
            dict: Minimum and recommended emergency fund amounts
        """
        # Debug: Log the profile ID 
        profile_id = profile.get('id')
        logging.info(f"DEBUG: Calculate emergency fund for profile: {profile_id}")
        
        # CRITICAL: Dump the complete profile structure to understand the exact format
        logging.info(f"DEBUG: Profile content: {profile}")
        
        # Get answers and dump all answers for debugging
        answers_raw = profile.get('answers', [])
        logging.info(f"DEBUG: Raw answers count: {len(answers_raw)}")
        
        # Log each answer separately for easier inspection
        for idx, answer in enumerate(answers_raw):
            logging.info(f"DEBUG: Answer {idx}: {answer}")
        
        # Get all answers as dictionary
        answers = {a.get('question_id'): a.get('answer') for a in answers_raw}
        
        # Debug: Log all available answer keys to see what we're working with
        logging.info(f"DEBUG: Available answer keys: {sorted(list(answers.keys()))}")
        
        # Define all possible field names for monthly expenses
        monthly_expense_fields = [
            'financial_basics_monthly_expenses',
            'financial_basics_expenses',
            'monthly_expenses'
        ]
        
        # Try each potential field name
        found_field = None
        raw_expense = None
        
        for field in monthly_expense_fields:
            if field in answers:
                found_field = field
                raw_expense = answers[field]
                logging.info(f"DEBUG: Found expense field '{field}' with value: {raw_expense} (type: {type(raw_expense).__name__})")
                break
        
        # If we found a value, try to parse it
        if raw_expense is not None:
            # Log the exact raw value for debugging
            logging.info(f"DEBUG: Raw expense value from field '{found_field}': '{raw_expense}'")
            
            # Try multiple parsing approaches
            try:
                # First, try direct conversion if it's already a number
                if isinstance(raw_expense, (int, float)):
                    monthly_expenses = float(raw_expense)
                    logging.info(f"DEBUG: Direct number conversion successful: {monthly_expenses}")
                
                # If it's a string, try different cleaning approaches
                elif isinstance(raw_expense, str):
                    # Strip all whitespace
                    raw_expense = raw_expense.strip()
                    logging.info(f"DEBUG: After whitespace strip: '{raw_expense}'")
                    
                    # Try different parsing techniques
                    # 1. Try direct conversion first
                    try:
                        monthly_expenses = float(raw_expense)
                        logging.info(f"DEBUG: Direct string conversion successful: {monthly_expenses}")
                    except ValueError:
                        # 2. Try removing currency symbol and commas
                        clean_expense = raw_expense.replace('₹', '').replace(',', '').replace('Rs', '').replace('INR', '').strip()
                        logging.info(f"DEBUG: After currency/commas removal: '{clean_expense}'")
                        
                        try:
                            monthly_expenses = float(clean_expense)
                            logging.info(f"DEBUG: Cleaned string conversion successful: {monthly_expenses}")
                        except ValueError:
                            # 3. Try extracting digits only
                            import re
                            digits_only = re.sub(r'[^\d.]', '', raw_expense)
                            logging.info(f"DEBUG: Digits only: '{digits_only}'")
                            
                            try:
                                monthly_expenses = float(digits_only)
                                logging.info(f"DEBUG: Digits-only conversion successful: {monthly_expenses}")
                            except ValueError:
                                # If all attempts fail, raise the exception to be caught by outer handler
                                raise ValueError(f"Could not convert '{raw_expense}' to a number after multiple attempts")
                
                # Success! We have a valid monthly expense value
                logging.info(f"FOUND VALID MONTHLY EXPENSES: ₹{monthly_expenses:,.0f}")
                
                # Validate the value
                if monthly_expenses <= 0:
                    logging.warning(f"Expense value is too low: {monthly_expenses}, using default")
                    monthly_expenses = 50000  # Default value
                    is_default = True
                    data_source = "Example"
                elif monthly_expenses < 5000:  # Sanity check for reasonable minimum
                    logging.warning(f"Expense value suspiciously low: {monthly_expenses}, might be incorrectly entered")
                    is_default = False
                    data_source = "Your Data (Low)"
                elif monthly_expenses > 1000000:  # Sanity check for reasonable maximum (₹10 lakhs monthly)
                    logging.warning(f"Expense value suspiciously high: {monthly_expenses}, might be incorrectly entered")
                    is_default = False
                    data_source = "Your Data (High)"
                else:
                    is_default = False
                    data_source = "Your Data"
                
                # Calculate based on India-specific guidelines (6-9 months)
                minimum_fund = monthly_expenses * 6
                recommended_fund = monthly_expenses * 9
                
                # Log the calculated values
                logging.info(f"CALCULATION RESULTS: Monthly Expenses=₹{monthly_expenses:,.0f} ({data_source}), "
                            f"Minimum Fund (6 months)=₹{minimum_fund:,.0f}, "
                            f"Recommended Fund (9 months)=₹{recommended_fund:,.0f}")
                
                result = {
                    'monthly_expenses': monthly_expenses,
                    'minimum_fund': minimum_fund,
                    'recommended_fund': recommended_fund,
                    'data_source': data_source
                }
                
                if is_default:
                    result['is_default'] = True
                
                return result
                
            except Exception as e:
                # Log the error in great detail
                import traceback
                logging.error(f"ERROR converting '{raw_expense}' to float: {str(e)}")
                logging.error(f"ERROR details: {traceback.format_exc()}")
        else:
            logging.warning(f"No monthly expense fields found. Available fields: {list(answers.keys())}")
        
        # No valid monthly expenses found, use default
        logging.warning("Using default monthly expense value of ₹50,000")
        monthly_expenses = 50000  # Using standard default value
        
        # Calculate based on India-specific guidelines (6-9 months)
        minimum_fund = monthly_expenses * 6
        recommended_fund = monthly_expenses * 9
        
        # Log using default values clearly
        logging.info(f"USING DEFAULT VALUES: Monthly Expenses=₹{monthly_expenses:,.0f} (Example), "
                    f"Minimum Fund=₹{minimum_fund:,.0f}, "
                    f"Recommended Fund=₹{recommended_fund:,.0f}")
        
        return {
            'monthly_expenses': monthly_expenses,
            'minimum_fund': minimum_fund,
            'recommended_fund': recommended_fund,
            'is_default': True,  # Flag to indicate this is default data
            'data_source': "Example"  # Indicate this is example data
        }
    
    def _has_adequate_insurance(self, profile):
        """
        Determine if the user has adequate insurance coverage based on answers.
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if insurance coverage is adequate
        """
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        # Check direct question about insurance adequacy
        if 'goals_insurance_adequacy' in answers:
            adequacy = answers['goals_insurance_adequacy']
            return adequacy == 'Yes, fully adequate'
            
        # Check insurance types
        if 'goals_insurance_types' in answers:
            insurance_types = answers['goals_insurance_types']
            if isinstance(insurance_types, list):
                # Consider adequate if they have at least health insurance and one other type
                has_health = "Health Insurance" in insurance_types
                return has_health and len(insurance_types) >= 2
            
        # Default to not adequate if we can't determine
        return False
        
    def _is_ready_for_next_level(self, profile):
        """
        Check if the profile is ready for next-level questions.
        Requires at least 70% of core questions to be answered.
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if ready for next-level questions
        """
        # Count answered core questions
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        core_questions = self.question_repository.get_core_questions()
        required_core = [q for q in core_questions if q.get('required', False)]
        
        if not required_core:
            return False
            
        # Calculate core completion percentage
        answered_required = [q for q in required_core if q.get('id') in answered_ids]
        core_completion = (len(answered_required) / len(required_core)) * 100
        
        # Count goal questions
        goal_questions_count = len([a for a in profile.get('answers', []) 
                                 if a.get('question_id', '').startswith('goals_')
                                 and not a.get('question_id', '').endswith('_insights')])
        
        # Ready for next-level when core is mostly complete AND at least 3 goal questions answered
        is_ready = core_completion >= 70 and goal_questions_count >= 3
        
        if not is_ready and goal_questions_count < 3:
            logging.info(f"Profile {profile.get('id')} has only answered {goal_questions_count}/3 required goal questions")
            
        return is_ready
        
    def _get_next_level_question(self, profile):
        """
        Get a next-level question for the profile.
        Uses LLM service to generate questions based on profile data.
        
        Args:
            profile (dict): User profile
            
        Returns:
            dict: Next-level question definition or None
        """
        profile_id = profile.get('id')
        
        # First check if there are predefined next-level questions that haven't been answered yet
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        next_level_repo_questions = self.question_repository.get_questions_by_type('next_level')
        
        # Count how many next-level questions have been answered
        next_level_answered_count = 0
        for q_id in answered_ids:
            if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                next_level_answered_count += 1
            # Count dynamic/LLM generated questions
            elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                next_level_answered_count += 1
                
        # Check if we've reached the minimum threshold for behavioral questions
        behavioral_questions_min = 5
        need_more_next_level = next_level_answered_count < behavioral_questions_min
        
        if need_more_next_level:
            logging.info(f"Profile {profile_id} has only answered {next_level_answered_count}/{behavioral_questions_min} next-level questions")
            
            # Try to find unanswered predefined next-level questions first
            found_predefined = False
            for q in next_level_repo_questions:
                qid = q.get('id')
                if qid and qid not in answered_ids:
                    # Check if this question applies to the profile (has dependencies)
                    if 'depends_on' in q:
                        # Skip checking this question if its dependency isn't met
                        continue
                    logging.info(f"Found unanswered next-level question: {qid}")
                    found_predefined = True
                    return q
                    
            # If we didn't find any predefined questions, try generating new ones using LLM
            if not found_predefined and next_level_answered_count < behavioral_questions_min:
                logging.info(f"No predefined next-level questions available, attempting LLM generation")
                
                # Initialize tracking dictionary if it doesn't exist
                if not hasattr(self, 'asked_questions'):
                    self.asked_questions = {}
                    
                # Initialize set for this profile if it doesn't exist
                if profile_id not in self.asked_questions:
                    self.asked_questions[profile_id] = set()
                
                # Get previously asked questions for this profile
                previously_asked = self.asked_questions.get(profile_id, set())
                
                # Generate new questions based on the profile's answers
                new_questions = self._generate_next_level_questions(profile)
                
                # Filter out questions that are similar to previously asked ones
                filtered_questions = self._filter_similar_questions(new_questions, previously_asked, answered_ids)
                
                # If we successfully generated new questions, return the first one
                if filtered_questions and len(filtered_questions) > 0:
                    logging.info(f"Successfully generated new LLM questions, returning first one")
                    first_question = filtered_questions[0]
                    # Mark this question as already asked to prevent it from being shown again
                    q_id = first_question.get('id')
                    if q_id:
                        self.asked_questions[profile_id].add(q_id)
                    # Cache the rest (excluding the one we're returning) for later
                    self.dynamic_questions_cache[profile_id] = filtered_questions[1:]
                    return first_question
                else:
                    logging.warning(f"LLM generation returned no questions, falling back to fallback questions")
                    
                    # As a last resort, if no LLM questions were generated, use fallback questions
                    if self._use_fallback_next_level_question(profile_id):
                        # Get the newly added fallback question from cache
                        cached_questions = self.dynamic_questions_cache.get(profile_id, [])
                        if cached_questions:
                            fallback_q = cached_questions[0]  # Get the one we just inserted at position 0
                            logging.info(f"Using fallback question: {fallback_q.get('id')}")
                            return fallback_q
            # If we get here and still need more next level questions but couldn't return one,
            # let's create and return an emergency fallback question directly
            if need_more_next_level:
                logging.warning(f"Still need more next-level questions for profile {profile_id}, creating emergency fallback")
                timestamp = int(time.time())
                emergency_q = {
                    "id": f"fallback_emergency_{timestamp}",
                    "question_id": f"fallback_emergency_{timestamp}",
                    "text": "What are your most important financial priorities right now?",
                    "category": "financial_basics",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 200,
                }
                logging.info(f"Created emergency fallback question with ID: {emergency_q['id']}")
                return emergency_q
        
        # Determine if it's time to regenerate questions
        # Count next-level questions already asked to determine if we should regenerate
        next_level_question_count = 0
        for q_id in answered_ids:
            if (q_id.startswith("next_level_") or 
                q_id.startswith("llm_next_level_") or 
                q_id.startswith("gen_question_") or 
                q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                next_level_question_count += 1
        
        # Force regeneration every 3 questions to ensure questions consider new answers
        regeneration_frequency = 3
        force_regeneration = next_level_question_count > 0 and next_level_question_count % regeneration_frequency == 0
        
        if force_regeneration:
            logging.info(f"Forcing LLM question regeneration after {next_level_question_count} next-level questions")
            # Generate new questions based on the updated profile with all latest answers
            new_questions = self._generate_next_level_questions(profile)
            
            # Initialize tracking dictionary if it doesn't exist
            if not hasattr(self, 'asked_questions'):
                self.asked_questions = {}
                
            # Initialize set for this profile if it doesn't exist
            if profile_id not in self.asked_questions:
                self.asked_questions[profile_id] = set()
            
            # Get previously asked questions for this profile
            previously_asked = self.asked_questions.get(profile_id, set())
            
            # Filter out questions that are similar to previously asked ones
            filtered_questions = self._filter_similar_questions(new_questions, previously_asked, answered_ids)
            
            # If we successfully generated new questions, return the first one
            if filtered_questions and len(filtered_questions) > 0:
                logging.info(f"Returning newly regenerated question after recent answers")
                first_question = filtered_questions[0]
                # Mark this question as already asked to prevent it from being shown again
                q_id = first_question.get('id')
                if q_id:
                    self.asked_questions[profile_id].add(q_id)
                # Store the rest (excluding the one we're returning) for later
                max_cache_size = 3  # Only keep 3 questions in cache to force regeneration soon
                self.dynamic_questions_cache[profile_id] = filtered_questions[1:max_cache_size+1]
                return first_question
        
        # If not regenerating, check if we already have cached questions for this profile
        cached_questions = self.dynamic_questions_cache.get(profile_id, [])
        
        # Initialize tracking dictionary if it doesn't exist
        if not hasattr(self, 'asked_questions'):
            self.asked_questions = {}
            
        # Initialize set for this profile if it doesn't exist
        if profile_id not in self.asked_questions:
            self.asked_questions[profile_id] = set()
        
        # Get previously asked questions for this profile
        previously_asked = self.asked_questions.get(profile_id, set())
        
        # Filter out already answered questions and previously asked questions
        unanswered_questions = []
        for q in cached_questions:
            q_id = q.get('id')
            q_question_id = q.get('question_id')
            
            # Check if neither id nor question_id is in answered_ids or previously_asked
            if ((q_id is not None and q_id not in answered_ids and q_id not in previously_asked) and 
                (q_question_id is not None and q_question_id not in answered_ids and q_question_id not in previously_asked)):
                unanswered_questions.append(q)
        
        # If we still have unanswered cached questions, return the first one
        if unanswered_questions and len(unanswered_questions) > 0:
            logging.info(f"Returning cached unanswered question for profile {profile_id}")
            first_question = unanswered_questions[0]
            # Mark this question as already asked to prevent it from being shown again
            q_id = first_question.get('id')
            if q_id:
                self.asked_questions[profile_id].add(q_id)
            # Update the cache to remove the question we're returning
            self.dynamic_questions_cache[profile_id] = [q for q in cached_questions if q.get('id') != q_id]
            return first_question
        
        # Check if we've reached a reasonable completion state (e.g., X questions answered)
        question_count = len(answered_ids)
        MAX_QUESTIONS = 30  # Define a reasonable upper limit for questions
        
        if question_count >= MAX_QUESTIONS:
            # Before marking as complete, check if we should transition to behavioral questions
            if self._is_ready_for_behavioral(profile):
                logging.info(f"Profile {profile_id} has completed {question_count} questions but is ready for behavioral questions")
                logging.info(f"Not marking as complete yet, will transition to behavioral questions")
                return None  # Return None to allow behavioral questions to be triggered
            
            logging.info(f"Profile {profile_id} has completed {question_count} questions, marking as complete")
            return self._get_completion_question(profile)
        
        # Otherwise, generate new questions based on the profile's answers
        new_questions = self._generate_next_level_questions(profile)
        
        # Filter out questions that are similar to previously asked ones
        filtered_questions = self._filter_similar_questions(new_questions, previously_asked, answered_ids)
        
        # Cache the new questions - we'll update this below when returning a question
        
        # Return the first new question if available
        if filtered_questions and len(filtered_questions) > 0:
            logging.info(f"Returning newly generated question for profile {profile_id}")
            first_question = filtered_questions[0]
            # Mark this question as already asked to prevent it from being shown again
            q_id = first_question.get('id')
            if q_id:
                self.asked_questions[profile_id].add(q_id)
            # Store the rest (excluding the one we're returning) for later
            # Only cache a few questions to ensure we regenerate frequently
            max_cache_size = 3  # Only keep 3 questions in cache to force regeneration soon
            self.dynamic_questions_cache[profile_id] = filtered_questions[1:max_cache_size+1]
            return first_question
        
        # If we need more next-level questions and can't generate them,
        # try to use fallback questions directly
        if need_more_next_level:
            # Find a category to use for fallback questions - prioritize financial_basics
            categories = ['financial_basics', 'assets_and_debts', 'demographics', 'special_cases']
            
            # Pick the first category with fallback questions that haven't been answered
            for category in categories:
                if category in self.FALLBACK_QUESTIONS:
                    for fallback_q in self.FALLBACK_QUESTIONS[category]:
                        fallback_id = fallback_q.get('id')
                        # Create a unique ID with timestamp to avoid conflicts
                        timestamp = int(time.time())
                        unique_id = f"{fallback_id}_{timestamp}"
                        
                        # Create a copy with the unique ID
                        question_copy = fallback_q.copy()
                        question_copy["id"] = unique_id
                        question_copy["question_id"] = unique_id
                        
                        # Mark this question as already asked to prevent it from being shown again
                        self.asked_questions[profile_id].add(unique_id)
                        
                        logging.info(f"Using fallback question for category {category}: {unique_id}")
                        return question_copy
                        
            logging.warning(f"No fallback questions available for profile {profile_id} that needs more next-level questions")
        
        # If still no questions available and we've answered enough, consider behavioral transition
        if question_count > 15:
            # Before marking as complete, check if we should transition to behavioral questions
            if self._is_ready_for_behavioral(profile):
                logging.info(f"QUESTION TRANSITION: No more next-level questions for profile {profile_id}")
                logging.info(f"QUESTION TRANSITION: Profile has answered {question_count} questions and is ready for behavioral questions")
                return None  # Return None to allow behavioral questions to be triggered
            
            logging.info(f"No more questions can be generated, but profile has answered {question_count} questions, marking as complete")
            return self._get_completion_question(profile)
        else:
            logging.warning(f"No questions available for profile {profile_id}")
            return None
    
    def _get_completion_question(self, profile):
        """
        Returns a special completion question when the profile is considered complete
        Since we're now using direct links instead of form submission for the completion screen,
        this redirects directly to the profile_complete page.
        
        Args:
            profile (dict): User profile
            
        Returns:
            None: To trigger a redirect to the profile_complete page
        """
        profile_id = profile.get('id')
        
        # Count next-level questions answered - same as in is_profile_complete method
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        next_level_answered_count = 0
        
        for q_id in answered_ids:
            if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                next_level_answered_count += 1
            # Count dynamic/LLM generated questions
            elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                next_level_answered_count += 1
                
        # Check if we've reached behavioral readiness (5+ next-level questions)
        # If not, see if there are unanswered next-level questions we could ask
        if next_level_answered_count < 5:
            # Check if there are any predefined next-level questions to ask
            next_level_repo_questions = self.question_repository.get_questions_by_type('next_level')
            has_unanswered_questions = False
            
            for q in next_level_repo_questions:
                qid = q.get('id')
                if qid and qid not in answered_ids:
                    # Check if this question applies to the profile (has dependencies)
                    if 'depends_on' in q:
                        # Skip checking this question if its dependency isn't met
                        continue
                    # Still have unanswered next-level questions
                    logging.info(f"Profile {profile_id} has unanswered next-level question: {qid}")
                    has_unanswered_questions = True
                    # Return None to continue showing questions
                    return None
                    
            # If we still need more next-level questions but don't have any predefined ones,
            # create an emergency fallback question directly
            if next_level_answered_count < 5:
                logging.warning(f"Profile {profile_id} needs {5-next_level_answered_count} more next-level questions before completion")
                timestamp = int(time.time())
                emergency_q = {
                    "id": f"fallback_emergency_{timestamp}",
                    "question_id": f"fallback_emergency_{timestamp}",
                    "text": "What are your most important financial priorities right now?",
                    "category": "financial_basics",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 200,
                }
                logging.info(f"Created emergency fallback question with ID: {emergency_q['id']}")
                return emergency_q
        
        # Count total number of answers
        num_answered = len(answered_ids)
        
        # Count goal questions (may need multiple rounds)
        goal_questions_count = len([a.get('question_id') for a in profile.get('answers', []) 
                                 if a.get('question_id', '').startswith('goals_') 
                                 and not a.get('question_id', '').endswith('_insights')])
                                 
        # Count behavioral questions
        behavioral_questions_answered = len([a.get('question_id') for a in profile.get('answers', []) 
                                         if a.get('question_id', '').startswith('behavioral_') 
                                         and not a.get('question_id', '').endswith('_insights')])
        
        # We need to ensure completion requires passing through all the required stages
        if goal_questions_count < 7:
            logging.info(f"Profile {profile_id} has only answered {goal_questions_count}/7 required goal questions - NOT marking as complete")
            return None
        elif next_level_answered_count < 5:
            logging.info(f"Profile {profile_id} has only answered {next_level_answered_count}/5 required next-level questions - NOT marking as complete")
            
            # Try to create a new emergency fallback question directly
            timestamp = int(time.time())
            emergency_q = {
                "id": f"fallback_emergency_{timestamp}",
                "question_id": f"fallback_emergency_{timestamp}",
                "text": "What are your most important financial priorities right now?",
                "category": "financial_basics",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 200,
            }
            logging.info(f"Created emergency fallback question with ID: {emergency_q['id']}")
            return emergency_q
        elif behavioral_questions_answered < 3:
            logging.info(f"Profile {profile_id} has only answered {behavioral_questions_answered}/3 required behavioral questions - NOT marking as complete")
            return None
            
        # Log that we're marking the profile as complete
        logging.info(f"Profile {profile_id} potentially complete, redirecting to completion page instead of showing question")
        
        # Return None to trigger a redirect in the app.py routes
        return None
        
    def _is_ready_for_behavioral(self, profile):
        """
        Check if the profile is ready for behavioral questions.
        This happens after completing sufficient core, goal, and next-level questions.
        
        Args:
            profile (dict): User profile
            
        Returns:
            bool: True if ready for behavioral questions
        """
        profile_id = profile.get('id')
        
        # Get answered question counts by type
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        
        # Calculate core completion percentage
        core_questions = self.question_repository.get_core_questions()
        required_core = [q for q in core_questions if q.get('required', False)]
        
        if not required_core:
            logging.warning(f"BEHAVIORAL CHECK: No required core questions found for profile {profile_id}")
            return False
            
        answered_required = [q for q in required_core if q.get('id') in answered_ids]
        core_completion = (len(answered_required) / len(required_core)) * 100
        
        # Count goal questions
        goal_questions_count = len([a for a in profile.get('answers', []) 
                                 if a.get('question_id', '').startswith('goals_')
                                 and not a.get('question_id', '').endswith('_insights')])
        
        # Check if we've answered enough next-level questions - using the exact same pattern as in get_profile_completion
        next_level_answered_count = 0
        
        # Count answered predefined next-level questions
        for q_id in answered_ids:
            if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                next_level_answered_count += 1
            # Count dynamic/LLM generated questions
            elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                next_level_answered_count += 1
        
        # Determine if we've shown behavioral questions already
        behavioral_count = len([q_id for q_id in answered_ids if q_id.startswith("behavioral_") and not q_id.endswith("_insights")])
        
        # Log the behavioral readiness check
        logging.info(f"BEHAVIORAL CHECK: Profile {profile_id} - Core completion: {core_completion:.1f}%, "
                    f"Goal questions: {goal_questions_count}, "
                    f"Next-level questions answered: {next_level_answered_count}, "
                    f"Behavioral questions answered: {behavioral_count}")
        
        # Get total number of behavioral questions available
        all_behavioral_questions = self.question_repository.get_questions_by_type("behavioral")
        max_behavioral = len(all_behavioral_questions)
        
        # Set maximum limits - from implementation plan
        MAX_NEXT_LEVEL = 15
        MAX_BEHAVIORAL = 7
        
        # Ready for behavioral when:
        # 1. Core is FULLY complete (100%)
        # 2. Completed at least 7 goal questions
        # 3. Completed at least 5 next-level questions 
        # 4. Not already reached maximum behavioral questions
        max_behavioral_to_show = min(MAX_BEHAVIORAL, max_behavioral)  # Cap at 7 or total available
        is_ready = (
            core_completion >= 100 and 
            goal_questions_count >= 7 and
            next_level_answered_count >= 5 and 
            behavioral_count < max_behavioral_to_show
        )
        
        if is_ready:
            logging.info(f"QUESTION TRANSITION: Profile {profile_id} is ready for behavioral questions")
        else:
            # Log why the profile is not ready for behavioral questions
            if core_completion < 80:
                logging.info(f"BEHAVIORAL CHECK: Profile {profile_id} core completion ({core_completion:.1f}%) below 80% threshold")
            if goal_questions_count < 7:
                logging.info(f"BEHAVIORAL CHECK: Profile {profile_id} goal question count ({goal_questions_count}) below threshold of 7")
            if next_level_answered_count < 5:
                logging.info(f"BEHAVIORAL CHECK: Profile {profile_id} next-level count ({next_level_answered_count}) below threshold of 5")
            if behavioral_count >= max_behavioral_to_show:
                logging.info(f"BEHAVIORAL CHECK: Profile {profile_id} already answered {behavioral_count} behavioral questions (max: {max_behavioral_to_show})")
        
        return is_ready
    
    def _get_behavioral_question(self, profile):
        """
        Get the next appropriate behavioral question.
        Selects questions that haven't been answered yet and limits to 3-4 per profile.
        
        Args:
            profile (dict): User profile
            
        Returns:
            dict: Next behavioral question or None
        """
        profile_id = profile.get('id')
        
        # Get all behavioral questions
        behavioral_questions = self.question_repository.get_questions_by_type("behavioral")
        if not behavioral_questions:
            logging.warning(f"BEHAVIORAL QUESTIONS: No behavioral questions found in repository for profile {profile_id}")
            return None
            
        logging.info(f"BEHAVIORAL QUESTIONS: Found {len(behavioral_questions)} behavioral questions in repository")
        
        # Get answered question IDs
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        
        # Filter for unanswered behavioral questions
        unanswered = [q for q in behavioral_questions if q.get('id') not in answered_ids]
        logging.info(f"BEHAVIORAL QUESTIONS: Found {len(unanswered)} unanswered behavioral questions for profile {profile_id}")
        
        try:
            # Count answered behavioral questions (excluding insights)
            behavioral_answered = [q_id for q_id in answered_ids if q_id.startswith("behavioral_") and not q_id.endswith("_insights")]
            
            # Limit to 7 behavioral questions maximum as per implementation plan
            if len(behavioral_answered) >= 7:
                logging.info(f"BEHAVIORAL QUESTIONS: Profile {profile_id} already answered {len(behavioral_answered)} behavioral questions, limiting to 7 max")
                return None
                
            # Also check if we've answered all available questions
            if len(behavioral_answered) >= len(behavioral_questions):
                logging.info(f"BEHAVIORAL QUESTIONS: Profile {profile_id} already answered all {len(behavioral_answered)} behavioral questions, no more will be shown")
                return None
        except Exception as e:
            logging.error(f"Error checking behavioral questions answered: {str(e)}")
            # If there's an error, default to showing a question
            
        # If we have unanswered behavioral questions, return the first one
        if unanswered:
            # Sort by order to ensure consistent presentation
            sorted_questions = sorted(unanswered, key=lambda q: q.get('order', 999))
            next_question = sorted_questions[0]
            
            logging.info(f"QUESTION TRANSITION: Returning behavioral question '{next_question.get('id')}' for profile {profile_id}")
            
            # Log the behavioral trait being assessed
            trait = next_question.get('behavioral_trait')
            if trait:
                logging.info(f"BEHAVIORAL QUESTIONS: Assessing '{trait}' trait with question {next_question.get('id')}")
            
            return next_question
            
        logging.info(f"BEHAVIORAL QUESTIONS: No suitable behavioral questions available for profile {profile_id}")
        return None
    
    def _filter_similar_questions(self, new_questions, previously_asked, answered_ids):
        """
        Filter out questions that are too similar to ones already asked
        
        Args:
            new_questions: List of newly generated questions
            previously_asked: Set of previously asked question IDs
            answered_ids: Set of answered question IDs
            
        Returns:
            list: Filtered list of questions
        """
        filtered_questions = []
        
        # Log how many questions we're starting with
        logging.info(f"Filtering {len(new_questions)} new questions")
        logging.info(f"Previously asked IDs count: {len(previously_asked)}")
        logging.info(f"Already answered IDs count: {len(answered_ids)}")
        
        # Ensure we have questions to filter
        if not new_questions:
            logging.warning("No questions provided to filter!")
            return []
            
        # Get all previously answered/asked question texts for content-based deduplication
        profile_id = None
        all_question_texts = set()
        
        # Collect questions from repository for textual comparison
        for q in self.question_repository.get_all_questions():
            if 'text' in q and q['text'] and q.get('id') in answered_ids:
                all_question_texts.add(q['text'].lower())
                
        # Also collect from the dynamic cache for all profiles
        for profile_questions in self.dynamic_questions_cache.values():
            for q in profile_questions:
                if 'text' in q and q['text'] and (q.get('id') in answered_ids or q.get('id') in previously_asked):
                    all_question_texts.add(q['text'].lower())
                    
        # Make sure each question has a unique ID to avoid filtering issues
        timestamp = int(time.time())
        filtered_count = 0
        duplicate_content_count = 0
        
        for idx, question in enumerate(new_questions):
            # Ensure question has both id and question_id fields
            original_id = question.get('id')
            original_question_id = question.get('question_id') 
            
            # Skip questions with IDs that have already been asked or answered
            if ((original_id is not None and (original_id in previously_asked or original_id in answered_ids)) or
                (original_question_id is not None and (original_question_id in previously_asked or original_question_id in answered_ids))):
                filtered_count += 1
                continue
                
            # Content-based deduplication - check if this question is semantically similar to previous ones
            if 'text' in question and question['text']:
                # Get normalized question text
                q_text = question['text'].lower().strip()
                
                # Check for exact duplicates
                if q_text in all_question_texts:
                    logging.info(f"⚠️ Filtering out duplicate question text: {q_text[:50]}...")
                    duplicate_content_count += 1
                    continue
                    
                # Special handling for investment allocation questions - these are particularly problematic
                investment_patterns = [
                    "savings are distributed", 
                    "savings are allocated", 
                    "how your savings are",
                    "investment allocation",
                    "portfolio allocation",
                    "asset allocation"
                ]
                
                # If this is an investment allocation question, check for other similar ones
                is_investment_q = any(pattern in q_text.lower() for pattern in investment_patterns)
                if is_investment_q:
                    for existing_text in all_question_texts:
                        # If the existing text is also an investment allocation question, filter this one
                        if any(pattern in existing_text.lower() for pattern in investment_patterns):
                            logging.info(f"⚠️ Filtering out investment allocation question: {q_text[:50]}...")
                            duplicate_content_count += 1
                            similar_found = True
                            break
                            
                    # If we found a similar investment question, skip to the next question
                    if similar_found:
                        continue
                    
                # Check for high similarity with existing questions
                similar_found = False
                for existing_text in all_question_texts:
                    # Calculate improved similarity for overlap detection
                    similarity = self._calculate_text_similarity(q_text, existing_text)
                    # Lower the threshold from 0.7 to 0.6 for broader matching
                    if similarity > 0.6:
                        logging.info(f"⚠️ Filtering out similar question (similarity: {similarity:.2f}): {q_text[:50]}...")
                        duplicate_content_count += 1
                        similar_found = True
                        break
                
                if similar_found:
                    continue
                
                # This is a unique question, add it to our tracking
                all_question_texts.add(q_text)
                
            # Generate unique ID based on timestamp and index if needed
            category = question.get('category', 'general')
            if not original_id and not original_question_id:
                unique_id = f"llm_next_level_{category}_{timestamp}_{idx}"
                question['id'] = unique_id
                question['question_id'] = unique_id
            elif not original_id:
                question['id'] = original_question_id
            elif not original_question_id:
                question['question_id'] = original_id
                
            # Add question to filtered list
            filtered_questions.append(question)
        
        # Log filtering results
        logging.info(f"Filtered out {filtered_count} ID-based dupes, {duplicate_content_count} content-based dupes, keeping {len(filtered_questions)} unique questions")
        
        return filtered_questions
        
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings using improved methods.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0-1
        """
        # First check for exact match after normalization
        normalized_text1 = ' '.join(text1.lower().split())
        normalized_text2 = ' '.join(text2.lower().split())
        
        if normalized_text1 == normalized_text2:
            return 1.0  # Exact match after normalization
            
        # Check if one text contains another significantly
        if len(normalized_text1) > 20 and len(normalized_text2) > 20:
            # If one text is almost fully contained in the other, consider it similar
            if normalized_text1 in normalized_text2 or normalized_text2 in normalized_text1:
                return 0.95  # Very high similarity for contained text
        
        # Use simple word-based Jaccard similarity as a fallback
        words1 = set(normalized_text1.split())
        words2 = set(normalized_text2.split())
        
        # Avoid division by zero
        if not words1 or not words2:
            return 0
            
        # Jaccard similarity = size of intersection / size of union
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        
        # Check for common keywords that indicate similar questions
        investment_keywords = [
            'savings', 'allocated', 'investment', 'portfolio', 'asset', 'distribution', 
            'accounts', 'equity', 'fixed deposit', 'mutual funds', 'stocks'
        ]
        
        # Count common financial keywords in both texts
        keyword_matches = sum(1 for kw in investment_keywords if kw in normalized_text1 and kw in normalized_text2)
        
        # If we have multiple financial keywords in common, boost the similarity score
        if keyword_matches >= 3:
            jaccard = max(jaccard, 0.8)  # Ensure at least 0.8 similarity for financial question matches
            
        return jaccard
        
    # Fallback questions for each category if LLM generation fails
    FALLBACK_QUESTIONS = {
        'demographics': [
            {
                "id": "fallback_demographics_1",
                "question_id": "fallback_demographics_1",
                "text": "How do your specific life circumstances affect your financial planning needs?",
                "category": "demographics",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 201,
            },
            {
                "id": "fallback_demographics_2", 
                "question_id": "fallback_demographics_2",
                "text": "Are there any significant life changes you anticipate in the next 3-5 years?",
                "category": "demographics",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 202,
            }
        ],
        'financial_basics': [
            {
                "id": "fallback_financial_basics_1",
                "question_id": "fallback_financial_basics_1",
                "text": "What are your top 3 financial priorities for the next year?",
                "category": "financial_basics",
                "type": "next_level",
                "input_type": "text", 
                "required": False,
                "order": 201,
            },
            {
                "id": "fallback_financial_basics_2",
                "question_id": "fallback_financial_basics_2",
                "text": "How do you plan to increase your savings rate over time?",
                "category": "financial_basics",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 202,
            }
        ],
        'assets_and_debts': [
            {
                "id": "fallback_assets_debts_1",
                "question_id": "fallback_assets_debts_1", 
                "text": "What is your strategy for managing your debt?",
                "category": "assets_and_debts",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 201,
            },
            {
                "id": "fallback_assets_debts_2",
                "question_id": "fallback_assets_debts_2",
                "text": "How diversified are your current assets?",
                "category": "assets_and_debts",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 202,
            }
        ],
        'special_cases': [
            {
                "id": "fallback_special_cases_1",
                "question_id": "fallback_special_cases_1",
                "text": "How does your business/real estate affect your overall financial position?",
                "category": "special_cases",
                "type": "next_level", 
                "input_type": "text",
                "required": False,
                "order": 201,
            },
            {
                "id": "fallback_special_cases_2",
                "question_id": "fallback_special_cases_2",
                "text": "What contingency plans do you have for your special assets?",
                "category": "special_cases",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 202,
            }
        ]
    }
    
    def _generate_next_level_questions(self, profile):
        """
        Generate new next-level questions based on profile data.
        
        Args:
            profile (dict): User profile
            
        Returns:
            list: List of next-level question definitions
        """
        # Get all categories and their completed core questions
        categories = ['demographics', 'financial_basics', 'assets_and_debts', 'special_cases']
        answered_questions = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        all_generated_questions = []
        
        # For each category, generate next-level questions based on core answers
        for category in categories:
            # Calculate completion percentage for this category's core questions
            category_questions = self.question_repository.get_questions_by_category(category)
            core_category_questions = [q for q in category_questions if q.get('type') == 'core']
            
            # Skip categories with no answered questions
            answered_core_in_category = {
                qid: answer for qid, answer in answered_questions.items()
                if any(q.get('id') == qid and q.get('type') == 'core' for q in category_questions)
            }
            
            if not answered_core_in_category:
                logging.info(f"Skipping {category}: no answered core questions")
                continue
                
            # Check if we have enough answers in this category to generate next-level questions
            completion = self.question_repository.get_category_completion(profile, category)
            if completion < 50:  # Need at least 50% completion in a category
                logging.info(f"Skipping {category}: completion {completion}% is below 50%")
                continue
                
            # Track if we have successful generation
            generation_successful = False
            questions = []
                
            # Generate questions for this category
            try:
                # Log the LLM service enabled status for debugging
                logging.info(f"LLM service enabled status: {self.llm_service.enabled}")
                
                if self.llm_service.enabled:
                    # Use LLM to generate questions with a shorter timeout
                    logging.info(f"Attempting to generate next-level questions for {category} using LLM")
                    questions = self.llm_service.generate_next_level_questions(
                        category, answered_core_in_category
                    )
                    
                    # Check if we actually got questions back
                    if questions and len(questions) > 0:
                        generation_successful = True
                        logging.info(f"Successfully generated {len(questions)} questions using LLM for {category}")
                    else:
                        logging.warning(f"LLM returned empty question list for {category}, will use fallbacks")
                
                # If LLM is disabled or returned no questions, use mock questions
                if not self.llm_service.enabled or not generation_successful:
                    logging.info(f"Using mock questions for {category} since LLM is disabled or returned no questions")
                    questions = self.llm_service.generate_mock_next_level_questions(category)
                    if questions and len(questions) > 0:
                        generation_successful = True
                        logging.info(f"Successfully generated {len(questions)} mock questions for {category}")
            except Exception as e:
                # Add full traceback for better debugging
                import traceback
                logging.error(f"Error generating next-level questions for {category}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Use fallback questions if LLM failed or returned no questions
            if not generation_successful and category in self.FALLBACK_QUESTIONS:
                logging.info(f"Using fallback questions for {category}")
                
                # Create copies of fallback questions with unique timestamps
                timestamp = int(time.time())
                fallback_questions = []
                
                for q in self.FALLBACK_QUESTIONS[category]:
                    # Create a copy to avoid modifying the original
                    question_copy = q.copy()
                    
                    # Add timestamp to ensure uniqueness
                    unique_id = f"{q['id']}_{timestamp}"
                    question_copy["id"] = unique_id
                    question_copy["question_id"] = unique_id
                    
                    fallback_questions.append(question_copy)
                
                questions = fallback_questions
                
            # Ensure all questions have both id and question_id fields with proper LLM prefixes
            timestamp = int(time.time())
            idx = 0
            for q in questions:
                # Check for generic IDs like 'next_level_question_1' or 'next_level_question_2'
                generic_id = False
                old_id = q.get('id', '')
                old_question_id = q.get('question_id', '')
                
                # Detect generic IDs from OpenAI that need to be replaced
                if old_id and (old_id.startswith('next_level_question_') or old_id == 'question_1' or old_id == 'question_2'):
                    generic_id = True
                    logging.info(f"⚠️ Replacing generic question ID: {old_id}")
                
                if old_question_id and (old_question_id.startswith('next_level_question_') or old_question_id == 'question_1' or old_question_id == 'question_2'):
                    generic_id = True
                    logging.info(f"⚠️ Replacing generic question ID: {old_question_id}")
                
                # Always generate a proper unique ID for LLM-generated questions
                # This ensures we never have generic IDs that could conflict
                if generic_id or "id" not in q or "question_id" not in q:
                    # Generate a new guaranteed unique ID
                    new_id = f"llm_next_level_{category}_{timestamp}_{idx}"
                    idx += 1
                    logging.info(f"🔄 Creating unique ID: {new_id} replacing {old_id}/{old_question_id}")
                    q["id"] = new_id
                    q["question_id"] = new_id
                elif "id" in q and "question_id" not in q:
                    q["question_id"] = q["id"]
                elif "question_id" in q and "id" not in q:
                    q["id"] = q["question_id"]
                    
                # Extra safety - log the final IDs
                logging.info(f"✓ Final question IDs: id={q['id']}, question_id={q['question_id']}")
                
                # Log each question generation
                profile_id = profile.get('id')
                self._log_question_generation(profile_id, q)
            
            all_generated_questions.extend(questions)
            logging.info(f"Added {len(questions)} next-level questions for category {category}")
            
        # Log the IDs of all generated questions for debugging
        question_ids = [q.get("id", "unknown") for q in all_generated_questions]
        logging.info(f"All generated question IDs: {question_ids}")
        
        return all_generated_questions
    
    def submit_answer(self, profile_id, question_id, answer_value):
        """
        Process and save an answer to a question.
        For next-level questions, also analyzes the answer using LLM.
        For goal questions, may create or update goals in the database.
        
        Args:
            profile_id (str): ID of the profile
            question_id (str): ID of the question
            answer_value: The answer value
            
        Returns:
            bool: Success status
            dict: Updated profile
        """
        try:
            logging.info(f"Submitting answer for question: {question_id}, profile: {profile_id}")
            
            # Load profile
            profile = self.profile_manager.get_profile(profile_id)
            if not profile:
                logging.error(f"Cannot submit answer: Profile {profile_id} not found")
                return False, {"error": f"Profile not found: {profile_id}"}
            
            # First check if it's a dynamic/LLM-generated question
            is_dynamic_question = False
            question = None
            
            # Detailed logging for LLM-generated questions
            is_llm_generated = (question_id and 
                               ('llm_next_level' in question_id or 
                                'gen_question' in question_id))
            
            if is_llm_generated:
                logging.info(f"💬 PROCESSING LLM-GENERATED QUESTION: {question_id}")
            
            # Check if it's a fallback question (they have unique timestamps)
            is_fallback = False
            if question_id and 'fallback_' in question_id and '_' in question_id.split('fallback_')[1]:
                is_fallback = True
                logging.info(f"🔄 DETECTED FALLBACK QUESTION: {question_id}")
                # Extract the base fallback question ID (without timestamp)
                parts = question_id.split('_')
                if len(parts) >= 4:  # Should be format like fallback_category_number_timestamp
                    base_id = f"fallback_{parts[1]}_{parts[2]}"
                    
                    # Try to find the matching fallback question template
                    for category in self.FALLBACK_QUESTIONS:
                        for template_q in self.FALLBACK_QUESTIONS[category]:
                            if template_q.get('id') == base_id:
                                # Create a copy with the full ID
                                question = template_q.copy()
                                question['id'] = question_id
                                question['question_id'] = question_id
                                is_dynamic_question = True
                                logging.info(f"✅ CREATED FALLBACK TEMPLATE: {question_id} based on {base_id}")
                                break
                        if question:
                            break
            
            # If not a fallback or fallback template not found, check dynamic questions cache
            if not question:
                cached_questions = self.dynamic_questions_cache.get(profile_id, [])
                logging.info(f"🔍 CHECKING DYNAMIC CACHE for {question_id}, cache has {len(cached_questions)} questions")
                
                # Log all the cached question IDs for debugging
                if cached_questions:
                    cache_ids = [q.get('id') for q in cached_questions]
                    logging.info(f"📋 CACHE IDS: {cache_ids}")
                
                for q in cached_questions:
                    q_id = q.get('id')
                    q_question_id = q.get('question_id')
                    
                    # Log the comparison for debugging
                    logging.info(f"🔄 COMPARING: question_id={question_id}, cached id={q_id}, cached question_id={q_question_id}")
                    
                    if q_id == question_id or q_question_id == question_id:
                        question = q
                        is_dynamic_question = True
                        logging.info(f"✅ FOUND IN CACHE: {question_id}")
                        break
                        
            # If not found in dynamic cache, check repository
            if not question:
                question = self.question_repository.get_question(question_id)
                if question:
                    logging.info(f"✅ FOUND IN REPOSITORY: {question_id}")
                
            # Special case handling for insights questions
            if not question and question_id.endswith('_insights'):
                # This is an insights-related question, create a placeholder
                logging.info(f"💡 CREATING INSIGHTS PLACEHOLDER: {question_id}")
                base_question_id = question_id.replace('_insights', '')
                related_question = self.question_repository.get_question(base_question_id)
                
                if related_question:
                    # Create a placeholder question for the insights
                    question = {
                        'id': question_id,
                        'question_id': question_id,
                        'type': 'insight',
                        'category': related_question.get('category', 'unknown'),
                        'text': f"Insights for {base_question_id}",
                        'input_type': 'json'
                    }
                    is_dynamic_question = True
                    logging.info(f"✅ CREATED INSIGHTS PLACEHOLDER: {question_id}")
                
            # Special handling for emergency fallback questions
            if not question and 'emergency' in question_id:
                logging.info(f"🚨 EMERGENCY FALLBACK: {question_id}")
                # Create an emergency question definition
                question = {
                    'id': question_id,
                    'question_id': question_id,
                    'type': 'next_level',
                    'category': 'financial_basics',
                    'text': "What are your most important financial priorities right now?",
                    'input_type': 'text',
                    'required': False
                }
                is_dynamic_question = True
                logging.info(f"✅ CREATED EMERGENCY FALLBACK: {question_id}")
            
            if not question:
                logging.error(f"❌ QUESTION NOT FOUND: {question_id}")
                logging.error(f"❌ Dynamic questions cache keys: {list(self.dynamic_questions_cache.keys())}")
                if profile_id in self.dynamic_questions_cache:
                    cache_ids = [q.get('id') for q in self.dynamic_questions_cache[profile_id]]
                    logging.error(f"❌ Questions in cache for profile {profile_id}: {cache_ids}")
                    
                # Emergency recovery: Create a placeholder question on demand to handle the answer
                # This ensures we don't lose user responses even if there's a cache/tracking issue
                if question_id and (question_id.startswith("llm_next_level_") or 
                                   question_id.startswith("gen_question_") or 
                                   question_id.startswith("fallback_")):
                    logging.info(f"🚨 EMERGENCY RECOVERY: Creating placeholder for missing question {question_id}")
                    
                    # Extract category from question ID if possible
                    category = "unknown"
                    for cat in ["demographics", "financial_basics", "assets_and_debts", "special_cases"]:
                        if cat in question_id:
                            category = cat
                            break
                            
                    # Create a placeholder question to accept the answer
                    question = {
                        'id': question_id,
                        'question_id': question_id,
                        'type': 'next_level',
                        'category': category,
                        'text': "Question not found in cache - placeholder created to save your answer",
                        'input_type': 'text',
                        'required': False,
                        'is_recovered': True  # Mark as recovered for tracking
                    }
                    is_dynamic_question = True
                else:
                    return False, {"error": f"Question not found: {question_id}"}
            
            # Log question details for debugging
            logging.info(f"Question details: type={question.get('type')}, category={question.get('category')}, input_type={question.get('input_type')}")
            
            # Validate answer based on question type
            valid, error_message = self._validate_answer(question, answer_value)
            if not valid:
                logging.error(f"Invalid answer for question {question_id}: {error_message}")
                return False, {"error": f"Invalid answer: {error_message}"}
                
            # For goal questions, process goal creation/update
            if question.get('type') == 'goal':
                try:
                    # Process goal-related answers
                    logging.info(f"Processing goal question {question_id} with answer: {answer_value}")
                    self._process_goal_answer(profile, question_id, answer_value)
                except Exception as e:
                    import traceback
                    logging.error(f"Error processing goal answer for question {question_id}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with normal answer saving even if goal processing fails
                    logging.info("Continuing with normal answer saving despite goal processing error")
                
            # For behavioral questions, extract behavioral insights
            elif question.get('type') == 'behavioral':
                try:
                    # Extract insights specific to the behavioral trait being assessed
                    behavioral_trait = question.get('behavioral_trait')
                    logging.info(f"Analyzing behavioral response for trait '{behavioral_trait}' with LLM...")
                    
                    insights = self.llm_service.extract_insights(
                        question.get('text', ''),
                        answer_value,
                        behavioral_trait=behavioral_trait
                    )
                    
                    # Store both the raw answer and the extracted insights
                    if self.llm_service.enabled and insights:
                        # Store the raw answer
                        updated_profile = self.profile_manager.add_answer(profile, question_id, answer_value)
                        
                        # Store the insights in a separate answer
                        insights_question_id = f"{question_id}_insights"
                        updated_profile = self.profile_manager.add_answer(
                            updated_profile, 
                            insights_question_id, 
                            insights
                        )
                        
                        logging.info(f"Saved behavioral answer with insights for {question_id} (trait: {behavioral_trait}) in profile {profile_id}")
                        return True, updated_profile
                except Exception as e:
                    import traceback
                    logging.error(f"Error analyzing behavioral response for question {question_id}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with normal answer saving if analysis fails
                    logging.info("Falling back to normal answer saving without behavioral LLM analysis")
            
            # For next-level or dynamically generated questions with text responses, analyze content
            elif (question.get('type') == 'next_level' or is_dynamic_question) and question.get('input_type') == 'text':
                try:
                    # Extract insights from text response
                    logging.info(f"Analyzing text response for question {question_id} with LLM...")
                    insights = self.llm_service.extract_insights(question.get('text', ''), answer_value)
                    
                    # Check for question patterns in the answer (without responding to them)
                    self._detect_user_questions_in_answer(answer_value, question_id)
                    
                    # Store both the raw answer and the extracted insights
                    if self.llm_service.enabled and insights:
                        # Store the raw answer
                        updated_profile = self.profile_manager.add_answer(profile, question_id, answer_value)
                        
                        # Store the insights in a separate answer
                        insights_question_id = f"{question_id}_insights"
                        updated_profile = self.profile_manager.add_answer(
                            updated_profile, 
                            insights_question_id, 
                            insights
                        )
                        
                        logging.info(f"Saved answer with insights for {question_id} in profile {profile_id}")
                        return True, updated_profile
                except Exception as e:
                    import traceback
                    logging.error(f"Error analyzing text response for question {question_id}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with normal answer saving if analysis fails
                    logging.info("Falling back to normal answer saving without LLM analysis")
            
            # Store the answer normally
            try:
                # Add specific logging for multiselect answers
                if question.get('input_type') == 'multiselect' and isinstance(answer_value, list):
                    logging.info(f"MULTISELECT ANSWER: User selected {len(answer_value)} options for question {question_id}: {answer_value}")
                
                updated_profile = self.profile_manager.add_answer(profile, question_id, answer_value)
                logging.info(f"Successfully saved answer for {question_id} in profile {profile_id}")
                
                # Add specific logging for business owner flow tracking
                if question_id == "demographics_employment_type" and answer_value == "Business owner":
                    logging.info(f"BUSINESS OWNER FLOW: User selected 'Business owner' employment type in profile {profile_id}")
                    logging.info(f"BUSINESS OWNER FLOW: Will trigger business value question next")
                elif question_id == "special_cases_business_value":
                    logging.info(f"BUSINESS OWNER FLOW: User provided business value ({answer_value}) in profile {profile_id}")
                
                # Add specific goals question tracking 
                if question_id == "goals_insurance_types" and isinstance(answer_value, list):
                    logging.info(f"GOALS FLOW: User selected insurance types: {answer_value}")
                elif question_id == "goals_other_categories" and isinstance(answer_value, list):
                    logging.info(f"GOALS FLOW: User selected financial goals: {answer_value}")
                
                # Add logging for question tier transitions
                if question_id.startswith("behavioral_") and not question_id.endswith("_insights"):
                    logging.info(f"QUESTION TIER: User answered BEHAVIORAL question '{question_id}' in profile {profile_id}")
                    # Check if we've answered enough behavioral questions
                    behavioral_answered = len([q_id for q_id in set(a.get('question_id') for a in profile.get('answers', [])) 
                                             if q_id.startswith("behavioral_") and not q_id.endswith("_insights")])
                    logging.info(f"QUESTION TIER: Profile {profile_id} has now answered {behavioral_answered}/4 behavioral questions")
                    
                # Track next-level questions using same patterns as in get_profile_completion and _is_ready_for_behavioral
                elif (
                    question_id.startswith("next_level_") or 
                    question_id.startswith("llm_next_level_") or 
                    question_id.startswith("gen_question_") or 
                    question_id.startswith("fallback_")
                ) and not question_id.endswith("_insights"):
                    logging.info(f"QUESTION TIER: User answered NEXT-LEVEL question '{question_id}' in profile {profile_id}")
                    # Log readiness for behavioral questions
                    self._is_ready_for_behavioral(updated_profile)
                    
                elif question_id.startswith("goals_"):
                    logging.info(f"QUESTION TIER: User answered GOAL question '{question_id}' in profile {profile_id}")
                
                # Check for unique questions after completion
                self._update_asked_questions_tracking(profile_id, question_id)
                
                return True, updated_profile
            except Exception as e:
                import traceback
                logging.error(f"Error saving answer: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                return False, {"error": f"Failed to save answer: {str(e)}"}
                
        except Exception as e:
            # Catch-all exception handler for robust error handling
            import traceback
            logging.error(f"Unexpected error in submit_answer: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False, {"error": f"An unexpected error occurred: {str(e)}"}
            
    def _process_goal_answer(self, profile, question_id, answer_value):
        """
        Process goal-related answers and create/update goals in the database.
        
        Args:
            profile (dict): User profile
            question_id (str): ID of the goal question
            answer_value: Answer value
        """
        from models.goal_models import GoalManager, Goal
        
        # Initialize goal manager with the same db_path as profile manager
        goal_manager = GoalManager()
        profile_id = profile.get('id')
        
        # Get all answers
        answers = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
        
        # Emergency Fund Goal
        if question_id == "goals_emergency_fund_target" and answer_value == "Yes":
            logging.info(f"User wants to create emergency fund goal for profile {profile_id}")
            
            # Check if we already have amount and timeframe
            amount = None
            timeframe = None
            
            if "goals_emergency_fund_amount" in answers:
                amount = float(answers["goals_emergency_fund_amount"])
            
            if "goals_emergency_fund_timeframe" in answers:
                timeframe = answers["goals_emergency_fund_timeframe"]
                
            # If we have both amount and timeframe, create the goal
            if amount and timeframe:
                emergency_fund_goal = Goal(
                    user_profile_id=profile_id,
                    category="emergency_fund",
                    title="Emergency Fund",
                    target_amount=amount,
                    timeframe=timeframe,
                    current_amount=0,
                    importance="high",
                    flexibility="somewhat_flexible",
                    notes="Building an emergency fund for unexpected expenses"
                )
                
                created_goal = goal_manager.create_goal(emergency_fund_goal)
                if created_goal:
                    logging.info(f"Created emergency fund goal for profile {profile_id} with target amount {amount}")
        
        # Insurance Goal
        elif question_id == "goals_insurance_target" and answer_value == "Yes":
            logging.info(f"User wants to create insurance goal for profile {profile_id}")
            
            # Check if we have insurance type and timeframe
            insurance_type = None
            timeframe = None
            
            if "goals_insurance_type_target" in answers:
                insurance_type = answers["goals_insurance_type_target"]
                
            if "goals_insurance_timeframe" in answers:
                timeframe = answers["goals_insurance_timeframe"]
                
            # If we have both type and timeframe, create the goal
            if insurance_type and timeframe:
                insurance_goal = Goal(
                    user_profile_id=profile_id,
                    category="insurance",
                    title=f"{insurance_type} Coverage",
                    timeframe=timeframe,
                    importance="high",
                    flexibility="somewhat_flexible",
                    notes=f"Obtaining adequate {insurance_type.lower()} coverage"
                )
                
                created_goal = goal_manager.create_goal(insurance_goal)
                if created_goal:
                    logging.info(f"Created insurance goal for profile {profile_id} for {insurance_type}")
        
        # Home Purchase Goal
        elif question_id == "goals_home_purchase_timeframe" and "goals_home_purchase_amount" in answers:
            logging.info(f"Processing home purchase goal for profile {profile_id}")
            
            amount = float(answers["goals_home_purchase_amount"])
            timeframe = answer_value
            
            if amount > 0:
                home_goal = Goal(
                    user_profile_id=profile_id,
                    category="home_purchase",
                    title="Home Purchase",
                    target_amount=amount,
                    timeframe=timeframe,
                    current_amount=0,
                    importance="high",
                    flexibility="somewhat_flexible",
                    notes="Saving for home purchase or down payment"
                )
                
                created_goal = goal_manager.create_goal(home_goal)
                if created_goal:
                    logging.info(f"Created home purchase goal for profile {profile_id} with target amount {amount}")
        
        # Education Funding Goal
        elif question_id == "goals_education_timeframe" and "goals_education_amount" in answers:
            logging.info(f"Processing education funding goal for profile {profile_id}")
            
            amount = float(answers["goals_education_amount"])
            timeframe = answer_value
            
            if amount > 0:
                education_goal = Goal(
                    user_profile_id=profile_id,
                    category="education",
                    title="Education Funding",
                    target_amount=amount,
                    timeframe=timeframe,
                    current_amount=0,
                    importance="medium",
                    flexibility="somewhat_flexible",
                    notes="Saving for education expenses"
                )
                
                created_goal = goal_manager.create_goal(education_goal)
                if created_goal:
                    logging.info(f"Created education funding goal for profile {profile_id} with target amount {amount}")
        
        # Debt Elimination Goal
        elif question_id == "goals_debt_timeframe" and "goals_debt_amount" in answers:
            logging.info(f"Processing debt elimination goal for profile {profile_id}")
            
            amount = float(answers["goals_debt_amount"])
            timeframe = answer_value
            
            if amount > 0:
                debt_goal = Goal(
                    user_profile_id=profile_id,
                    category="debt_elimination",
                    title="Debt Elimination",
                    target_amount=amount,
                    timeframe=timeframe,
                    current_amount=0,
                    importance="high",
                    flexibility="somewhat_flexible",
                    notes="Paying off existing debts"
                )
                
                created_goal = goal_manager.create_goal(debt_goal)
                if created_goal:
                    logging.info(f"Created debt elimination goal for profile {profile_id} with target amount {amount}")
        
        # Retirement Goal
        elif question_id == "goals_retirement_timeframe" and "goals_retirement_amount" in answers:
            logging.info(f"Processing retirement goal for profile {profile_id}")
            
            amount = float(answers["goals_retirement_amount"])
            retirement_age = int(answer_value)
            
            # Determine if it's early or traditional retirement
            category = "early_retirement" if retirement_age < 60 else "traditional_retirement"
            
            if amount > 0:
                retirement_goal = Goal(
                    user_profile_id=profile_id,
                    category=category,
                    title=f"Retirement at age {retirement_age}",
                    target_amount=amount,
                    timeframe=f"Age {retirement_age}",
                    current_amount=0,
                    importance="high",
                    flexibility="somewhat_flexible",
                    notes=f"Saving for retirement at age {retirement_age}"
                )
                
                created_goal = goal_manager.create_goal(retirement_goal)
                if created_goal:
                    logging.info(f"Created retirement goal for profile {profile_id} with target amount {amount}")
        
        # Custom Goal
        elif question_id == "goals_custom_timeframe" and "goals_custom_title" in answers and "goals_custom_amount" in answers:
            logging.info(f"Processing custom goal for profile {profile_id}")
            
            title = answers["goals_custom_title"]
            amount = float(answers["goals_custom_amount"])
            timeframe = answer_value
            
            if amount > 0 and title:
                custom_goal = Goal(
                    user_profile_id=profile_id,
                    category="custom",
                    title=title,
                    target_amount=amount,
                    timeframe=timeframe,
                    current_amount=0,
                    importance="medium",
                    flexibility="somewhat_flexible",
                    notes=f"Custom goal: {title}"
                )
                
                created_goal = goal_manager.create_goal(custom_goal)
                if created_goal:
                    logging.info(f"Created custom goal '{title}' for profile {profile_id} with target amount {amount}")
        
        # Goals importance and flexibility
        elif question_id == "goals_importance_flexibility":
            logging.info(f"Processing goal importance and flexibility for profile {profile_id}")
            
            # Get all goals for this profile
            profile_goals = goal_manager.get_profile_goals(profile_id)
            
            # Determine importance and flexibility based on answer
            importance = "high"
            flexibility = "somewhat_flexible"
            
            if answer_value == "Achieving the exact target amount is most important, timing is flexible":
                importance = "high"
                flexibility = "very_flexible"
            elif answer_value == "Achieving the goal by the target date is most important, amount is flexible":
                importance = "medium"
                flexibility = "fixed"
            elif answer_value == "Both target amount and timing are equally important":
                importance = "high"
                flexibility = "fixed"
            elif answer_value == "Both target amount and timing are somewhat flexible":
                importance = "medium"
                flexibility = "somewhat_flexible"
            
            # Update all goals
            for goal in profile_goals:
                goal.importance = importance
                goal.flexibility = flexibility
                goal_manager.update_goal(goal)
                
            logging.info(f"Updated importance and flexibility for {len(profile_goals)} goals in profile {profile_id}")
                
        # Other questions may not create goals directly but prepare for later goal creation
        else:
            logging.info(f"No immediate goal creation for question {question_id}")
            # Could add more processing for other question types as needed
        
    def _detect_user_questions_in_answer(self, answer_text, question_id):
        """
        Detect if the user is asking questions in their responses.
        This is for logging/analytics purposes only.
        
        Args:
            answer_text: The user's text response
            question_id: The ID of the question being answered
        """
        try:
            if not isinstance(answer_text, str):
                return
                
            # Simple pattern matching for questions
            question_indicators = [
                "?",
                "how can i",
                "how do i",
                "what is",
                "why is",
                "when should",
                "where can"
            ]
            
            for indicator in question_indicators:
                if indicator in answer_text.lower():
                    logging.info(f"DETECTED USER QUESTION: '{answer_text}' in response to question {question_id}")
                    break
        except Exception as e:
            # Don't let this affect the main flow
            logging.warning(f"Error in question detection: {str(e)}")
    
    def _update_asked_questions_tracking(self, profile_id, question_id):
        """
        Keep track of questions that have been asked to avoid duplicates
        
        Args:
            profile_id: The profile ID
            question_id: The question that was just answered
        """
        # Initialize tracking dictionary if it doesn't exist
        if not hasattr(self, 'asked_questions'):
            self.asked_questions = {}
            
        # Initialize set for this profile if it doesn't exist
        if profile_id not in self.asked_questions:
            self.asked_questions[profile_id] = set()
            
        # Add this question to the set of asked questions
        self.asked_questions[profile_id].add(question_id)
        
        # If this was a next-level question, log it clearly and check our count
        if (question_id.startswith("next_level_") or
            question_id.startswith("llm_next_level_") or
            question_id.startswith("gen_question_") or
            question_id.startswith("fallback_")) and not question_id.endswith("_insights"):
            
            # Get all answered questions to count next-level
            profile = self.profile_manager.get_profile(profile_id)
            answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
            next_level_count = 0
            
            for q_id in answered_ids:
                if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                    next_level_count += 1
                # Count dynamic/LLM generated questions
                elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                    next_level_count += 1
            
            logging.info(f"NEXT-LEVEL TRACKING: Just answered next-level question {question_id}, profile now has {next_level_count}/5 required next-level questions")
        
        # Log for debugging
        logging.info(f"Updated asked questions tracking for profile {profile_id}. Total questions asked: {len(self.asked_questions[profile_id])}")
        
        # Log available cached questions for this profile
        if profile_id in self.dynamic_questions_cache:
            cached_q_ids = [q.get('id') for q in self.dynamic_questions_cache[profile_id]]
            logging.info(f"Dynamic questions available in cache for profile {profile_id}: {cached_q_ids}")
        
    def _log_question_generation(self, profile_id, question):
        """
        Log when a question is generated.
        
        Args:
            profile_id: The profile ID
            question: The question definition
        """
        question_id = question.get('id')
        if not question_id:
            return
            
        # Log the question generation
        self.question_logger.log_question_generated(profile_id, question_id, question)
        logging.info(f"Logged generation of question {question_id} for profile {profile_id}")

    def _use_fallback_next_level_question(self, profile_id):
        """
        Generate and provide a fallback next-level question for profiles that need more
        next-level questions to reach the minimum threshold.
        
        Args:
            profile_id: The profile ID
            
        Returns:
            bool: True if a fallback question was successfully added to the cache
        """
        # Get profile
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            logging.error(f"Cannot provide fallback question: Profile {profile_id} not found")
            return False
            
        # Get answered question IDs
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        
        # Initialize tracking dictionary if it doesn't exist
        if not hasattr(self, 'asked_questions'):
            self.asked_questions = {}
            
        # Initialize set for this profile if it doesn't exist
        if profile_id not in self.asked_questions:
            self.asked_questions[profile_id] = set()
            
        # Get previously asked questions
        previously_asked = self.asked_questions.get(profile_id, set())
        
        # Find a category to use for fallback questions - prioritize financial_basics
        categories = ['financial_basics', 'assets_and_debts', 'demographics', 'special_cases']
        
        # Create timestamp for unique question IDs
        timestamp = int(time.time())
        
        # Track which questions have been used already
        used_base_ids = set()
        
        # Check previously asked questions
        for prev_id in previously_asked:
            if 'fallback_' in prev_id:
                parts = prev_id.split('_')
                if len(parts) >= 3:
                    base_id = f"fallback_{parts[1]}_{parts[2]}"
                    used_base_ids.add(base_id)
        
        # Also consider recently answered questions
        for ans_id in answered_ids:
            if 'fallback_' in ans_id:
                parts = ans_id.split('_')
                if len(parts) >= 3:
                    base_id = f"fallback_{parts[1]}_{parts[2]}"
                    used_base_ids.add(base_id)
                    
        # Also check the dynamic questions cache for fallback questions that are queued
        if profile_id in self.dynamic_questions_cache:
            for cached_q in self.dynamic_questions_cache[profile_id]:
                q_id = cached_q.get('id', '')
                if q_id and 'fallback_' in q_id:
                    parts = q_id.split('_')
                    if len(parts) >= 3:
                        base_id = f"fallback_{parts[1]}_{parts[2]}"
                        used_base_ids.add(base_id)
                        
        logging.info(f"Already used fallback questions: {used_base_ids}")
        
        # First collect all available unused fallback questions
        available_questions = []
        
        for category in categories:
            if category in self.FALLBACK_QUESTIONS:
                for fallback_q in self.FALLBACK_QUESTIONS[category]:
                    fallback_id = fallback_q.get('id')
                    
                    # Skip this question if it's already been used
                    if fallback_id in used_base_ids:
                        logging.info(f"Skipping already used fallback question: {fallback_id}")
                        continue
                    
                    # Add to available questions
                    available_questions.append((category, fallback_q))
        
        # If we have available questions, select one at random to ensure variety
        if available_questions:
            import random
            category, fallback_q = random.choice(available_questions)
            fallback_id = fallback_q.get('id')
            
            # Create a unique ID with timestamp to avoid conflicts
            unique_id = f"{fallback_id}_{timestamp}"
            
            # Create a copy with the unique ID
            question_copy = fallback_q.copy()
            question_copy["id"] = unique_id
            question_copy["question_id"] = unique_id
            
            # Add to cache for this profile
            if profile_id not in self.dynamic_questions_cache:
                self.dynamic_questions_cache[profile_id] = []
                
            # Add to front of cache so it will be selected next
            self.dynamic_questions_cache[profile_id].insert(0, question_copy)
            
            # Also add the base ID to our tracking to prevent duplicates in same session
            self.asked_questions[profile_id].add(unique_id)
            
            # Log the question generation
            self._log_question_generation(profile_id, question_copy)
            
            logging.info(f"Added fallback question for category {category} to cache: {unique_id}")
            return True
                    
        # If we've used all standard fallback questions, create additional variants
        if len(used_base_ids) >= 8:  # If we've used all 8 standard fallback questions
            # Create an extra fallback question with timestamp to ensure uniqueness
            extra_q = {
                "id": f"fallback_extra_{timestamp}",
                "question_id": f"fallback_extra_{timestamp}",
                "text": "What other financial goals or concerns would you like to discuss?",
                "category": "financial_basics",
                "type": "next_level",
                "input_type": "text",
                "required": False,
                "order": 201,
            }
            
            # Add to cache for this profile
            if profile_id not in self.dynamic_questions_cache:
                self.dynamic_questions_cache[profile_id] = []
                
            # Add to front of cache so it will be selected next
            self.dynamic_questions_cache[profile_id].insert(0, extra_q)
            
            # Also add to our tracking to prevent duplicates in same session
            self.asked_questions[profile_id].add(extra_q["id"])
            
            # Log the question generation
            self._log_question_generation(profile_id, extra_q)
            
            logging.info(f"Added extra fallback question to cache: {extra_q['id']}")
            return True
                    
        logging.warning(f"No unused fallback questions found for profile {profile_id}")
        return False
    
    def _validate_answer(self, question, answer_value):
        """
        Validate an answer against question constraints.
        
        Args:
            question (dict): Question definition
            answer_value: The answer value
            
        Returns:
            bool: Validity status
            str: Error message if invalid
        """
        # Special case for insights or json data
        if question.get('type') == 'insight' or question.get('input_type') == 'json':
            # Don't validate insights data structures
            return True, None
            
        # Handle missing input_type
        input_type = question.get('input_type', 'text')
        if not input_type:
            logging.warning(f"Question {question.get('id', 'unknown')} has no input_type, defaulting to 'text'")
            input_type = 'text'
        
        # Handle empty answers for required questions
        if question.get('required', False) and (
            answer_value is None or 
            (isinstance(answer_value, str) and answer_value.strip() == '')
        ):
            return False, f"Question {question.get('id', 'unknown')} requires an answer"
            
        # Validate based on input type
        if input_type == 'number':
            try:
                if isinstance(answer_value, str):
                    # Try to convert to float for validation
                    answer_value = float(answer_value)
                
                min_val = question.get('min')
                max_val = question.get('max')
                
                if min_val is not None and answer_value < min_val:
                    return False, f"Value must be at least {min_val}"
                    
                if max_val is not None and answer_value > max_val:
                    return False, f"Value must be at most {max_val}"
            except ValueError:
                return False, "Invalid number format"
            except TypeError:
                logging.warning(f"Type error validating number answer: {answer_value}")
                # Allow non-numeric answers for testing/development
                return True, None
                
        elif input_type in ['select', 'radio']:
            options = question.get('options', [])
            if options and answer_value not in options:
                return False, f"Answer must be one of: {', '.join(options)}"
        
        elif input_type == 'multiselect':
            # Validate that answer is a list for multiselect
            if not isinstance(answer_value, list):
                logging.warning(f"Expected list for multiselect question {question.get('id')}, got {type(answer_value)}: {answer_value}")
                # If it's a string, try to convert it to a list with one item
                if isinstance(answer_value, str):
                    logging.info(f"Converting string to list for multiselect: {answer_value}")
                    # Don't modify the original answer_value, just for validation
                    return True, None
                return False, "Multiselect answers must be provided as a list"
            
            # Validate that all selected options are valid
            options = question.get('options', [])
            if options:
                invalid_options = [opt for opt in answer_value if opt not in options]
                if invalid_options:
                    return False, f"Invalid options selected: {', '.join(invalid_options)}"
        
        # For text type or any other type, accept any non-empty answer
        return True, None
    
    def get_profile_completion(self, profile_id):
        """
        Calculate profile completion metrics for core, next-level, and behavioral tiers,
        and determine the profile understanding level.
        
        Args:
            profile_id (str): ID of the profile
            
        Returns:
            dict: Completion statistics by category, tier, overall, and understanding level
        """
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            logging.error(f"Profile {profile_id} not found")
            return None
        
        # Calculate completion for each category
        categories = ['demographics', 'financial_basics', 'assets_and_debts', 'special_cases']
        completion_by_category = {}
        
        # Get answered question IDs
        answered_questions = profile.get('answers', [])
        answered_ids = set(a.get('question_id') for a in answered_questions)
        
        # Count all types of next-level questions (both predefined and dynamically generated)
        dynamic_questions_count = 0
        dynamic_questions_answered = 0
        
        # Count predefined next-level questions from the repository
        next_level_repo_questions = self.question_repository.get_questions_by_type('next_level')
        for q in next_level_repo_questions:
            qid = q.get('id') or q.get('question_id')
            if qid:
                dynamic_questions_count += 1
                if qid in answered_ids:
                    dynamic_questions_answered += 1
        
        # Also count dynamic/LLM-generated questions from cache
        cached_questions = self.dynamic_questions_cache.get(profile.get('id', ''), [])
        for q in cached_questions:
            qid = q.get('id') or q.get('question_id')
            if qid:
                # Check if this is a next-level type question (to avoid double counting)
                if qid.startswith("llm_next_level_") or qid.startswith("gen_question_") or qid.startswith("fallback_"):
                    dynamic_questions_count += 1
                    if qid in answered_ids:
                        dynamic_questions_answered += 1
        
        # Calculate core completion by category
        core_completion_by_category = {}
        for category in categories:
            completion = self.question_repository.get_category_completion(profile, category)
            completion_by_category[category] = completion
            
            # Calculate core-specific completion
            category_questions = self.question_repository.get_questions_by_category(category)
            core_questions = [q for q in category_questions if q.get('type') == 'core']
            required_core = [q for q in core_questions if q.get('required', False)]
            
            if required_core:
                answered_required = [q for q in required_core if q.get('id') in answered_ids]
                core_completion = (len(answered_required) / len(required_core)) * 100
                core_completion_by_category[category] = round(core_completion, 1)
            else:
                core_completion_by_category[category] = 100.0
        
        # Calculate weights for overall completion
        weights = {
            'demographics': 0.4,
            'financial_basics': 0.3,
            'assets_and_debts': 0.2,
            'special_cases': 0.1
        }
        
        # Calculate core and overall completion
        core_overall = sum(core_completion_by_category[cat] * weights[cat] 
                          for cat in categories)
        
        overall_completion = sum(completion_by_category[cat] * weights[cat] 
                                for cat in categories)
        
        # Calculate next-level completion 
        next_level_completion = 0.0
        
        # Use the actual count of next-level questions for the progress bar
        # This includes both predefined and dynamic questions
        if dynamic_questions_count > 0:
            next_level_completion = (dynamic_questions_answered / dynamic_questions_count) * 100
            # Cap at 100%
            next_level_completion = min(100.0, next_level_completion)
            
        # Calculate behavioral questions completion
        behavioral_questions = self.question_repository.get_questions_by_type('behavioral')
        behavioral_questions_count = len(behavioral_questions)
        
        # Count only behavioral questions that don't end with _insights
        behavioral_questions_answered = len([q_id for q_id in answered_ids 
                                         if q_id.startswith("behavioral_") and not q_id.endswith("_insights")])
        
        # Target number of behavioral questions - use all available behavioral questions
        target_behavioral_count = behavioral_questions_count
        
        behavioral_completion = 0.0
        if behavioral_questions_answered > 0:
            behavioral_completion = min(100.0, (behavioral_questions_answered / target_behavioral_count) * 100)
        
        # Create the basic completion metrics structure
        completion_metrics = {
            'by_category': completion_by_category,
            'core': {
                'by_category': core_completion_by_category,
                'overall': round(core_overall, 1)
            },
            'next_level': {
                'questions_count': dynamic_questions_count,
                'questions_answered': dynamic_questions_answered,
                'completion': round(next_level_completion, 1)
            },
            'behavioral': {
                'questions_count': target_behavioral_count,
                'questions_answered': behavioral_questions_answered,
                'completion': round(behavioral_completion, 1)
            },
            'overall': round(overall_completion, 1)
        }
        
        # Calculate the understanding level
        try:
            understanding_level = self.understanding_calculator.calculate_level(profile, completion_metrics)
            logging.info(f"Profile {profile_id} understanding level: {understanding_level['id']}")
            
            # Add the understanding level to the completion metrics
            completion_metrics['understanding_level'] = understanding_level
        except Exception as e:
            # If there's an error calculating understanding level, log it but don't fail
            logging.error(f"Error calculating understanding level for profile {profile_id}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        
        return completion_metrics
    
    def get_answered_questions(self, profile_id):
        """
        Get all answered questions with their values.
        
        Args:
            profile_id (str): ID of the profile
            
        Returns:
            list: Answered questions with details
        """
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            logging.error(f"Profile {profile_id} not found")
            return []
        
        result = []
        for answer in profile.get('answers', []):
            question_id = answer.get('question_id')
            question = self.question_repository.get_question(question_id)
            
            if question:
                result.append({
                    'question': question,
                    'answer': answer.get('answer'),
                    'timestamp': answer.get('timestamp')
                })
        
        # Sort by category and order
        return sorted(result, key=lambda x: (
            x['question'].get('category', ''),
            x['question'].get('order', 9999)
        ))
    
    def is_profile_complete(self, profile_id):
        """
        Check if all required core questions are answered.
        
        Args:
            profile_id (str): ID of the profile
            
        Returns:
            bool: True if all required questions are complete
        """
        completion = self.get_profile_completion(profile_id)
        if not completion:
            return False
        
        # Get answered question IDs
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            return False
            
        answered_ids = set(a.get('question_id') for a in profile.get('answers', []))
        
        # Count next-level questions answered - using the exact same pattern as in get_profile_completion
        next_level_answered_count = 0
        
        for q_id in answered_ids:
            if q_id.startswith("next_level_") and not q_id.endswith("_insights"):
                next_level_answered_count += 1
            # Count dynamic/LLM generated questions
            elif (q_id.startswith("llm_next_level_") or q_id.startswith("gen_question_") or q_id.startswith("fallback_")) and not q_id.endswith("_insights"):
                next_level_answered_count += 1
                
        # Check if we've reached behavioral readiness (5+ next-level questions)
        behavioral_ready = next_level_answered_count >= 5
        
        # Count answered behavioral questions
        behavioral_questions_answered = len([q_id for q_id in answered_ids if q_id.startswith("behavioral_") and not q_id.endswith("_insights")])
        
        # Get total number of behavioral questions available
        behavioral_questions = self.question_repository.get_questions_by_type("behavioral")
        
        # Check if all behavioral questions are answered
        behavioral_completion = behavioral_questions_answered >= len(behavioral_questions)
        
        # Count total answers
        num_answered = len(answered_ids)
        
        # Require minimum thresholds for each question type to progress through zones
        # These align with profile understanding levels: RED → AMBER → YELLOW → GREEN → DARK_GREEN
        
        # Get core questions completion percentage
        core_completion_pct = completion.get('core', {}).get('overall', 0)
        
        # Count goal questions (may need multiple rounds)
        goal_questions_count = len([q_id for q_id in answered_ids if q_id.startswith("goals_") and not q_id.endswith("_insights")])
        
        # Determine if we meet the minimum requirements for each level
        
        # For AMBER level: Core 100% complete, 3+ goal questions
        amber_requirements_met = (core_completion_pct >= 100.0 and goal_questions_count >= 3)
        
        # For YELLOW level: Core 100% complete, 7+ goal questions, 5+ next-level questions
        yellow_requirements_met = (core_completion_pct >= 100.0 and 
                                 goal_questions_count >= 7 and 
                                 next_level_answered_count >= 5)
        
        # For GREEN level: Core 100% complete, 7+ goal questions, 5+ next-level questions, 3+ behavioral questions
        green_requirements_met = (core_completion_pct >= 100.0 and 
                                goal_questions_count >= 7 and 
                                next_level_answered_count >= 5 and 
                                behavioral_questions_answered >= 3)
        
        # For DARK_GREEN level: Core 100% complete, 7+ goal questions, 15+ next-level questions, 7+ behavioral questions
        dark_green_requirements_met = (core_completion_pct >= 100.0 and 
                                     goal_questions_count >= 7 and 
                                     next_level_answered_count >= 15 and 
                                     behavioral_questions_answered >= 7)
        
        # Log current level status
        logging.info(f"Profile {profile_id} level requirements: AMBER={amber_requirements_met}, YELLOW={yellow_requirements_met}, " +
                    f"GREEN={green_requirements_met}, DARK_GREEN={dark_green_requirements_met}")
        
        # For profile to be truly complete, we want at least GREEN level
        # This ensures they have core, goals, next-level, and behavioral questions
        minimum_complete_level = green_requirements_met
        
        # Extreme safety check - if user has answered a very large number of questions and met most requirements
        # This prevents endless loops if there's some edge case we haven't handled
        if num_answered >= 40 and yellow_requirements_met:
            logging.info(f"Safety threshold: Profile {profile_id} has completed {num_answered} questions (40+ safety threshold) " +
                        f"and has met YELLOW level requirements, marking as complete")
            return True
        
        # Return whether the profile meets the minimum requirements for completion
        if not minimum_complete_level:
            # Log which requirements are missing
            if not core_completion_pct >= 100.0:
                logging.info(f"Profile {profile_id} core completion ({core_completion_pct}%) is below required 100%, not marking as complete")
            if not goal_questions_count >= 7:
                logging.info(f"Profile {profile_id} has only answered {goal_questions_count}/7 required goal questions, not marking as complete")
            if not next_level_answered_count >= 5:
                logging.info(f"Profile {profile_id} has only answered {next_level_answered_count}/5 required next-level questions, not marking as complete")
            if not behavioral_questions_answered >= 3:
                logging.info(f"Profile {profile_id} has only answered {behavioral_questions_answered}/3 required behavioral questions, not marking as complete")
            
            return False
        else:
            logging.info(f"Profile {profile_id} meets minimum level requirements (GREEN), marking as complete")
            return True