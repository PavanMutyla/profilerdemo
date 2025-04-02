import os
import logging
import requests
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

class LLMService:
    """
    Service for integrating with Language Models (OpenAI/Claude) to generate
    dynamic next-level questions and analyze financial profile responses.
    """
    
    # Standardized schema for financial metrics extracted from responses
    INSIGHT_SCHEMA = {
        # Risk metrics
        "risk_scores": {
            "risk_tolerance": {"type": "float", "min": 0, "max": 10, "description": "Overall financial risk tolerance (0-10)"},
            "market_risk_comfort": {"type": "float", "min": 0, "max": 10, "description": "Comfort with market volatility (0-10)"},
            "income_stability": {"type": "float", "min": 0, "max": 10, "description": "Stability of income sources (0-10)"},
            "debt_comfort": {"type": "float", "min": 0, "max": 10, "description": "Comfort with debt (0-10)"},
        },
        
        # Knowledge metrics
        "knowledge_scores": {
            "financial_literacy": {"type": "float", "min": 0, "max": 10, "description": "Overall financial literacy (0-10)"},
            "investment_knowledge": {"type": "float", "min": 0, "max": 10, "description": "Knowledge of investment products (0-10)"},
            "tax_planning_knowledge": {"type": "float", "min": 0, "max": 10, "description": "Understanding of tax implications (0-10)"},
        },
        
        # Goal orientation
        "goal_scores": {
            "short_term_focus": {"type": "float", "min": 0, "max": 10, "description": "Focus on short-term goals (0-10)"},
            "long_term_focus": {"type": "float", "min": 0, "max": 10, "description": "Focus on long-term goals (0-10)"},
            "wealth_preservation": {"type": "float", "min": 0, "max": 10, "description": "Prioritizes preserving wealth (0-10)"},
            "wealth_growth": {"type": "float", "min": 0, "max": 10, "description": "Prioritizes growing wealth (0-10)"},
        },
        
        # Behavioral metrics
        "behavioral_indicators": {
            "loss_aversion": {"type": "float", "min": 0, "max": 10, "description": "Tendency to avoid losses (0-10)"},
            "recency_bias": {"type": "float", "min": 0, "max": 10, "description": "Influenced by recent market events (0-10)"},
            "herd_mentality": {"type": "float", "min": 0, "max": 10, "description": "Following others' investment decisions (0-10)"},
            "overconfidence": {"type": "float", "min": 0, "max": 10, "description": "Overconfidence in financial decisions (0-10)"},
            "fomo": {"type": "float", "min": 0, "max": 10, "description": "Fear of missing out on investment opportunities (0-10)"},
            "emotional_investing": {"type": "float", "min": 0, "max": 10, "description": "Emotional influence on financial decisions (0-10)"},
            "discipline": {"type": "float", "min": 0, "max": 10, "description": "Financial discipline and consistency (0-10)"},
            "information_processing": {"type": "float", "min": 0, "max": 10, "description": "Thoroughness in processing financial information (0-10)"},
        },
        
        # Financial priorities
        "financial_priorities": {
            "type": "ranked_list",
            "options": [
                "Emergency fund", "Debt reduction", "Retirement planning", 
                "Children's education", "Home ownership", "Wealth accumulation",
                "Tax optimization", "Legacy planning", "Lifestyle enhancement"
            ],
            "description": "Ranked financial priorities based on response"
        },
        
        # India-specific context
        "india_specific": {
            "tax_sensitivity": {"type": "float", "min": 0, "max": 10, "description": "Sensitivity to tax considerations (0-10)"},
            "family_financial_responsibility": {"type": "float", "min": 0, "max": 10, "description": "Level of financial responsibility for family (0-10)"},
            "gold_real_estate_preference": {"type": "float", "min": 0, "max": 10, "description": "Preference for traditional assets like gold and real estate (0-10)"},
            "equity_comfort": {"type": "float", "min": 0, "max": 10, "description": "Comfort with equity investments (0-10)"},
        },
        
        # Key facts extracted from response
        "extracted_facts": {
            "type": "list",
            "max_items": 5,
            "description": "Key factual information extracted from response"
        },
        
        # Concerns and opportunities
        "concerns": {
            "type": "list",
            "max_items": 3,
            "description": "Financial concerns identified in response"
        },
        
        "opportunities": {
            "type": "list",
            "max_items": 3,
            "description": "Financial opportunities identified in response"
        },
        
        # Time horizons mentioned
        "time_horizons": {
            "type": "dict",
            "description": "Time frames mentioned for different goals",
            "schema": {
                "short_term": {"type": "list", "description": "Short-term goals (0-2 years)"},
                "medium_term": {"type": "list", "description": "Medium-term goals (2-5 years)"},
                "long_term": {"type": "list", "description": "Long-term goals (5+ years)"}
            }
        },
        
        # Overall investment profile classification
        "investment_profile_type": {
            "type": "enum",
            "options": ["Conservative", "Moderately Conservative", "Balanced", "Moderately Aggressive", "Aggressive"],
            "description": "Overall investment profile classification"
        },
        
        # Confidence in extraction
        "confidence_score": {
            "type": "float",
            "min": 0,
            "max": 1,
            "description": "Confidence in the accuracy of extraction (0-1)"
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", cache_size: int = 100):
        """
        Initialize the LLM service with API credentials.
        
        Args:
            api_key: API key for the LLM service (OpenAI by default)
            model: Model to use for generation ("gpt-4o-mini" by default)
            cache_size: Maximum number of responses to cache (default 100)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logging.warning("No API key provided for LLM service. LLM features will be disabled.")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
        # Check if LLM_ENABLED is explicitly set in the environment
        llm_enabled_env = os.environ.get('LLM_ENABLED', '').lower()
        if llm_enabled_env in ('false', '0', 'f', 'no'):
            self.enabled = False
            logging.warning("LLM service explicitly disabled via LLM_ENABLED environment variable")
        else:
            self.enabled = bool(self.api_key)
            
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize response cache for performance improvement
        self.response_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.total_calls = 0
        
        logging.info(f"LLM Service initialized with model: {self.model}, caching enabled (max {cache_size} entries)")
        logging.info(f"LLM service enabled: {self.enabled}, API key length: {len(self.api_key or '')}")
        
        # Initialize schema for standardized data extraction
        self.insight_schema = self.INSIGHT_SCHEMA
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates for different question types and categories.
        """
        templates = {
            "next_level_demographics": """
            You are a financial advisor analyzing a user's financial profile.
            Based on the following demographic information, generate 1-2 relevant follow-up questions
            that would help deepen the financial profile.
            
            Current information:
            {{core_answers}}
            
            Focus on questions that would reveal:
            - Financial responsibilities and dependencies
            - Impact of health or employment status on financial planning
            - Geographic influence on financial situation
            
            Format your response as a JSON array of objects with the following structure:
            [
                {
                    "question_id": "next_level_question_1",
                    "text": "Your follow-up question here?",
                    "category": "demographics",
                    "type": "next_level",
                    "related_to": "demographics_dependents" // The core question this relates to
                }
            ]
            """,
            
            "next_level_financial_basics": """
            You are a financial advisor analyzing a user's financial profile.
            Based on the following financial information, generate 1-2 relevant follow-up questions
            that would help deepen the financial profile.
            
            Current information:
            {{core_answers}}
            
            Focus on questions that would reveal:
            - Savings allocation and strategy
            - Income stability and potential
            - Spending patterns and financial habits
            - Short-term financial management
            
            IMPORTANT: 
            - DO NOT generate questions about essential vs. discretionary expense percentages as this is already covered elsewhere.
            - DO NOT generate questions about financial goals, long-term plans, or savings targets as these are covered in the goals section.
            
            Format your response as a JSON array of objects with the following structure:
            [
                {
                    "question_id": "next_level_question_1",
                    "text": "Your follow-up question here?",
                    "category": "financial_basics",
                    "type": "next_level",
                    "related_to": "financial_basics_current_savings" // The core question this relates to
                }
            ]
            """,
            
            "next_level_assets_and_debts": """
            You are a financial advisor analyzing a user's financial profile.
            Based on the following assets and debts information, generate 1-2 relevant follow-up questions
            that would help deepen the financial profile.
            
            Current information:
            {{core_answers}}
            
            Focus on questions that would reveal:
            - Debt strategy and prioritization
            - Asset liquidity and accessibility
            - Risk exposure in asset portfolio
            - Current management of existing assets
            
            IMPORTANT: DO NOT generate questions about plans to acquire major assets like real estate or vehicles, as these are already covered in the goals section.
            
            Format your response as a JSON array of objects with the following structure:
            [
                {
                    "question_id": "next_level_question_1",
                    "text": "Your follow-up question here?",
                    "category": "assets_and_debts",
                    "type": "next_level",
                    "related_to": "assets_debts_total_debt" // The core question this relates to
                }
            ]
            """,
            
            "next_level_special_cases": """
            You are a financial advisor analyzing a user's financial profile.
            Based on the following special case information, generate 1-2 relevant follow-up questions
            that would help deepen the financial profile.
            
            Current information:
            {{core_answers}}
            
            Focus on questions that would reveal:
            - Business strategy and succession planning
            - Real estate investment strategy
            - Exposure to special financial circumstances
            - Long-term planning for unusual assets
            
            Format your response as a JSON array of objects with the following structure:
            [
                {
                    "question_id": "next_level_question_1",
                    "text": "Your follow-up question here?",
                    "category": "special_cases",
                    "type": "next_level",
                    "related_to": "special_cases_business_value" // The core question this relates to
                }
            ]
            """,
            
            "extract_insights": """
            You are a financial advisor in India analyzing a user's response to a financial question to extract standardized metrics for financial planning.
            Focus on extracting insights relevant to financial planning and investment advisory in the Indian market context.
            
            Question: {{question}}
            
            User's response: {{answer}}
            
            I need you to extract the following metrics in a standardized format according to this exact schema:
            
            ```
            {
                "risk_scores": {
                    "risk_tolerance": 0-10,  // Overall financial risk tolerance
                    "market_risk_comfort": 0-10,  // Comfort with market volatility
                    "income_stability": 0-10,  // Stability of income sources
                    "debt_comfort": 0-10  // Comfort with debt
                },
                "knowledge_scores": {
                    "financial_literacy": 0-10,  // Overall financial literacy
                    "investment_knowledge": 0-10,  // Knowledge of investment products
                    "tax_planning_knowledge": 0-10  // Understanding of tax implications
                },
                "goal_scores": {
                    "short_term_focus": 0-10,  // Focus on short-term goals
                    "long_term_focus": 0-10,  // Focus on long-term goals
                    "wealth_preservation": 0-10,  // Prioritizes preserving wealth
                    "wealth_growth": 0-10  // Prioritizes growing wealth
                },
                "behavioral_indicators": {
                    "loss_aversion": 0-10,  // Tendency to avoid losses
                    "recency_bias": 0-10,  // Influenced by recent market events
                    "herd_mentality": 0-10,  // Following others' investment decisions
                    "overconfidence": 0-10,  // Overconfidence in financial decisions
                    "fomo": 0-10,  // Fear of missing out on investment opportunities
                    "emotional_investing": 0-10,  // Emotional influence on financial decisions
                    "discipline": 0-10,  // Financial discipline and consistency
                    "information_processing": 0-10  // Thoroughness in processing financial information
                },
                "financial_priorities": [
                    // Ranked list of priorities, include only those that apply
                    "Emergency fund", 
                    "Debt reduction",
                    "Retirement planning", 
                    "Children's education", 
                    "Home ownership", 
                    "Wealth accumulation",
                    "Tax optimization", 
                    "Legacy planning", 
                    "Lifestyle enhancement"
                ],
                "india_specific": {
                    "tax_sensitivity": 0-10,  // Sensitivity to tax considerations in India
                    "family_financial_responsibility": 0-10,  // Level of financial responsibility for family
                    "gold_real_estate_preference": 0-10,  // Preference for traditional assets like gold and real estate
                    "equity_comfort": 0-10  // Comfort with equity investments
                },
                "extracted_facts": [
                    // List key factual information (maximum 5 items)
                ],
                "concerns": [
                    // Financial concerns identified (maximum 3 items)
                ],
                "opportunities": [
                    // Financial opportunities identified (maximum 3 items)
                ],
                "time_horizons": {
                    "short_term": [],  // Short-term goals (0-2 years)
                    "medium_term": [],  // Medium-term goals (2-5 years)
                    "long_term": []  // Long-term goals (5+ years)
                },
                "investment_profile_type": "Conservative"|"Moderately Conservative"|"Balanced"|"Moderately Aggressive"|"Aggressive",
                "confidence_score": 0.0-1.0  // Confidence in extraction accuracy
            }
            ```
            
            IMPORTANT GUIDELINES:
            1. For all numeric scores (0-10), provide a value only if you have reasonable evidence to infer it from the response
            2. For undetectable metrics, omit them from the JSON rather than assigning default values
            3. Focus on explicit and implicit cues in the user's language
            4. Consider cultural context relevant to financial planning in India
            5. Be conservative in your assessment - prefer confidence over comprehensiveness
            6. Provide a confidence_score that reflects your certainty in the overall extraction
            
            Format your response ONLY as a valid JSON object matching this schema exactly. Make sure to remove any trailing commas.
            """,
            
            "extract_behavioral_insights": """
            You are a financial behavioral analyst evaluating a user's response to a psychological question about their financial behavior.
            Focus on extracting insights related to financial psychology and behavioral biases.
            
            Behavioral trait being assessed: {{behavioral_trait}}
            Question: {{question}}
            User's response: {{answer}}
            
            I need you to extract behavioral insights based on the user's response:
            
            ```
            {
                "behavioral_indicators": {
                    "loss_aversion": 0-10,  // Tendency to avoid losses over seeking gains
                    "recency_bias": 0-10,  // Influenced by recent market events
                    "herd_mentality": 0-10,  // Following others' investment decisions
                    "overconfidence": 0-10,  // Overconfidence in financial decisions
                    "fomo": 0-10,  // Fear of missing out on investment opportunities
                    "emotional_investing": 0-10,  // Emotional influence on financial decisions
                    "discipline": 0-10,  // Financial discipline and consistency
                    "information_processing": 0-10  // Thoroughness in processing financial information
                },
                "risk_profile": {
                    "actual_risk_tolerance": 0-10,  // Actual risk tolerance based on behaviors
                    "stated_vs_actual_gap": -5 to 5  // Gap between stated and actual risk tolerance (negative means actual < stated)
                },
                "primary_bias": "loss_aversion|recency_bias|herd_mentality|overconfidence|fomo|emotional_investing",  // The most dominant bias
                "behavioral_summary": "",  // 1-2 sentence summary of behavioral profile
                "behavioral_strengths": [
                    // List behavioral strengths (maximum 2 items)
                ],
                "behavioral_challenges": [
                    // List behavioral challenges to address (maximum 2 items)
                ],
                "confidence_score": 0.0-1.0  // Confidence in this assessment
            }
            ```
            
            IMPORTANT GUIDELINES:
            1. Focus only on the specific behavioral trait being assessed
            2. For behavioral indicators, provide values only for relevant traits that can be assessed from this response
            3. Be honest but compassionate in your assessment
            4. Maintain financial profiling perspective (not clinical diagnosis)
            5. Focus on descriptive insights, not prescriptive advice
            
            Format your response ONLY as a valid JSON object matching this schema exactly. Make sure to remove any trailing commas.
            """
        }
        
        return templates
    
    def generate_next_level_questions(
        self, 
        category: str, 
        core_answers: Dict[str, Any],
        max_questions: int = 2,
        existing_question_ids: List[str] = None,
        profile: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate next-level questions based on core answers for a specific category.
        
        Args:
            category: Category to generate questions for (demographics, financial_basics, etc.)
            core_answers: Dictionary of core answers with question_id as keys
            max_questions: Maximum number of questions to generate
            existing_question_ids: List of question IDs that have already been asked or are in the repository
            profile: User profile data to check for completed sections (like goals)
            
        Returns:
            List of question definitions in dictionary format
        """
        # More detailed logging about the request and state
        logging.info(f"generate_next_level_questions called for category: {category}")
        logging.info(f"LLM service enabled: {self.enabled}, API key length: {len(self.api_key) if self.api_key else 0}")
        logging.info(f"Core answers available: {len(core_answers)}")
        
        if not self.enabled:
            logging.warning("LLM service is disabled. Returning empty question list.")
            return []
        
        # Initialize existing questions if None
        if existing_question_ids is None:
            existing_question_ids = []
        
        # Format core answers for the prompt
        formatted_answers = []
        for question_id, answer in core_answers.items():
            formatted_answers.append(f"{question_id}: {answer}")
        
        answers_text = "\n".join(formatted_answers)
        
        # Get the appropriate template
        template_key = f"next_level_{category}"
        if template_key not in self.prompt_templates:
            logging.error(f"No prompt template found for category: {category}")
            return []
        
        # Create context about existing questions
        existing_questions_context = ""
        if existing_question_ids:
            existing_questions_context = f"\nIMPORTANT: The following questions have already been asked or are in the repository, so avoid generating similar questions:\n- " + "\n- ".join(existing_question_ids)
            logging.info(f"Added context about {len(existing_question_ids)} existing questions")
        
        # Check if goals section has been completed
        goals_completed = False
        goals_context = ""
        if profile is not None:
            answers = profile.get('answers', [])
            goal_answers = [a for a in answers if a.get('question_id', '').startswith('goals_')]
            if any(a.get('question_id') == 'goals_other_categories' for a in goal_answers):
                goals_completed = True
                # Extract the actual goals the user selected
                for a in goal_answers:
                    if a.get('question_id') == 'goals_other_categories':
                        selected_goals = a.get('answer', [])
                        if selected_goals and isinstance(selected_goals, list):
                            goals_context = f"\nIMPORTANT: The user has already completed the goals section and selected the following goals: {', '.join(selected_goals)}.\nDO NOT generate questions about financial goals, savings goals, or future financial plans that would overlap with these existing goals."
                        else:
                            goals_context = "\nIMPORTANT: The user has already completed the goals section. DO NOT generate questions about financial goals, savings goals, or future financial plans."
                        logging.info("Added context about completed goals section")
                        break
        
        # Fill in the template
        prompt = self.prompt_templates[template_key].replace("{{core_answers}}", answers_text)
        
        # Add context to the prompt
        additional_context = ""
        if existing_questions_context:
            additional_context += existing_questions_context
        if goals_context:
            additional_context += goals_context
            
        if additional_context:
            # Find the position to insert the context - before the Format section
            format_section = "Format your response as a JSON array"
            if format_section in prompt:
                parts = prompt.split(format_section, 1)
                prompt = parts[0] + additional_context + "\n\n" + format_section + parts[1]
            else:
                # If we can't find the format section, append to the end
                prompt += additional_context
        
        try:
            # Call the LLM API
            logging.info(f"Calling LLM API for category: {category} with prompt length: {len(prompt)}")
            response = self._call_llm_api(prompt)
            logging.info(f"Received LLM API response with length: {len(response)}")
            
            # Log response preview for debugging
            preview = response[:200] + "..." if len(response) > 200 else response
            logging.info(f"LLM API response preview: {preview}")
            
            try:
                # Parse the generated questions
                parsed_json = json.loads(response)
                logging.info(f"Successfully parsed JSON response of type: {type(parsed_json)}")
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse LLM response as JSON: {str(json_err)}")
                logging.error(f"Raw response: {response}")
                
                # Try to extract JSON using regex as a fallback
                import re
                json_matches = re.findall(r'(\{.*\}|\[.*\])', response, re.DOTALL)
                
                if json_matches:
                    # Try each potential JSON match
                    for match in json_matches:
                        try:
                            parsed_json = json.loads(match)
                            logging.info(f"Successfully extracted JSON using regex: {type(parsed_json)}")
                            break
                        except:
                            continue
                    else:  # No successful parsing
                        logging.error("Could not extract valid JSON with regex")
                        return []
                else:
                    logging.error("No JSON-like patterns found in response")
                    return []
            
            # Add detailed logging for debugging
            logging.info(f"Parsed LLM response: {type(parsed_json)}")
            
            # Handle different response formats
            if isinstance(parsed_json, list):
                # Direct list of questions (expected format)
                questions = parsed_json
                logging.info("LLM response is a direct list of questions")
            elif isinstance(parsed_json, dict):
                # Check if response has a 'questions' field containing the array
                if 'questions' in parsed_json and isinstance(parsed_json['questions'], list):
                    questions = parsed_json['questions']
                    logging.info("Extracted questions array from response object")
                # Check if it's a single question object
                elif any(key in parsed_json for key in ['question_id', 'text', 'category']):
                    questions = [parsed_json]
                    logging.info("Converted single question object to list")
                else:
                    # Try to find any array in the response
                    for key, value in parsed_json.items():
                        if isinstance(value, list) and len(value) > 0:
                            if all(isinstance(item, dict) for item in value):
                                questions = value
                                logging.info(f"Extracted questions array from '{key}' field")
                                break
                    else:
                        logging.error(f"Could not extract questions from dict response: {parsed_json}")
                        return []
            else:
                # Cannot process this response
                logging.error(f"Unexpected response type: {type(parsed_json)}")
                return []
            
            # Safely limit the number of questions without using slices
            question_subset = []
            for i, q in enumerate(questions):
                if i >= max_questions:
                    break
                question_subset.append(q)
            
            # Validate questions and filter out any too similar to existing ones
            valid_questions = []
            for i, q in enumerate(question_subset):
                # Generate a consistent question_id if one wasn't provided
                if not q.get("question_id") or q.get("question_id") == "next_level_question_1":
                    q["question_id"] = f"llm_next_level_{category}_{int(time.time())}_{i}"
                
                # Add required fields
                q["input_type"] = q.get("input_type", "text")
                q["required"] = False
                q["order"] = 100 + i  # Place after core questions
                
                # Add an ID field if missing (needed in some places where id vs question_id is used)
                if "id" not in q:
                    q["id"] = q["question_id"]
                
                # Check if this question is too similar to existing questions
                question_text = q.get("text", "").lower()
                skip_question = False
                
                # Keywords that indicate potential redundancy
                redundancy_keywords = {
                    "essential": ["discretionary", "percentage", "expenses"],
                    "plans to": ["acquire", "major assets", "real estate", "vehicle", "property"],
                    "vehicle": ["purchase", "buy", "car", "automobile"],
                    "discretionary": ["essential", "percentage", "expenses"],
                    "property": ["purchase", "buy", "real estate", "home"],
                    "financial goals": ["primary", "next 5", "future", "priorities"],
                    "goals": ["financial", "saving", "investment", "planning", "future"],
                    "saving for": ["future", "goals", "plans", "aim"],
                    "future plans": ["financial", "money", "invest", "save"]
                }
                
                # Check for combinations of redundancy keywords
                for primary_keyword, related_keywords in redundancy_keywords.items():
                    if primary_keyword in question_text:
                        if any(related in question_text for related in related_keywords):
                            logging.warning(f"Skipping potentially redundant question based on keywords: {question_text}")
                            skip_question = True
                            break
                
                # Additional check for goal-related questions if goals have been completed
                if not skip_question and goals_completed:
                    goal_related_terms = [
                        "financial goal", "saving goal", "investment goal", "future plan", 
                        "retirement", "education fund", "home purchase", "financial priority", 
                        "what are your goals", "primary objective", "financial objective"
                    ]
                    if any(term in question_text for term in goal_related_terms):
                        logging.warning(f"Skipping goal-related question since goals section is complete: {question_text}")
                        skip_question = True
                
                if not skip_question and existing_question_ids:
                    # Basic similarity check with existing questions
                    for q_id in existing_question_ids:
                        if q_id in question_text or question_text in q_id:
                            logging.warning(f"Skipping question similar to existing ID {q_id}: {question_text}")
                            skip_question = True
                            break
                
                if not skip_question:
                    valid_questions.append(q)
            
            filtered_count = len(question_subset) - len(valid_questions)
            logging.info(f"Generated {len(valid_questions)} next-level questions for category {category}" + 
                         (f" (filtered out {filtered_count} redundant ones)" if filtered_count > 0 else ""))
            return valid_questions
            
        except Exception as e:
            # Log the full stack trace
            import traceback
            logging.error(f"Error generating next-level questions: {str(e)}")
            logging.error(f"Stack trace: {traceback.format_exc()}")
            
            # Return mock questions as a fallback
            logging.warning(f"Using mock questions for {category} due to LLM error")
            return self.generate_mock_next_level_questions(category)
    
    def extract_insights(self, question: str, answer: str, behavioral_trait: str = None) -> Dict[str, Any]:
        """
        Extract standardized insights from a user's text response.
        Uses a consistent schema to ensure comparable metrics across all responses.
        
        Args:
            question: The question that was asked
            answer: The user's text response
            behavioral_trait: Optional behavioral trait being assessed (for behavioral questions)
            
        Returns:
            Dictionary of extracted insights with standardized metrics
        """
        if not self.enabled:
            logging.warning("LLM service is disabled. Returning comprehensive fallback insights structure.")
            
            # If this is a behavioral question, return a behavioral-specific fallback structure
            if behavioral_trait:
                logging.info(f"Creating default behavioral insights structure for trait: {behavioral_trait}")
                return {
                    "raw_answer": answer,
                    "question": question,
                    "behavioral_indicators": {
                        behavioral_trait: 6.0  # Default middle value 
                    },
                    "behavioral_strengths": [f"Awareness of {behavioral_trait}"],
                    "behavioral_challenges": [f"Managing {behavioral_trait} consistently"],
                    "behavioral_summary": f"You show awareness of how {behavioral_trait} affects your financial decisions.",
                    "confidence_score": 0.5,
                    "primary_bias": behavioral_trait,
                    "insight_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Otherwise return a more comprehensive fallback structure with all expected fields
            return {
                "raw_answer": answer,
                "question": question,
                "extracted_facts": ["LLM service disabled - no extraction performed"],
                "concerns": ["Unable to extract concerns - LLM service is disabled"],
                "opportunities": ["Consider enabling the LLM service for detailed insights"],
                "risk_scores": {
                    "risk_tolerance": 5.0,  # Neutral default
                    "market_risk_comfort": 5.0,
                    "income_stability": 5.0,
                    "debt_comfort": 5.0
                },
                "knowledge_scores": {
                    "financial_literacy": 5.0,
                    "investment_knowledge": 5.0,
                    "tax_planning_knowledge": 5.0
                },
                "goal_scores": {
                    "short_term_focus": 5.0,
                    "long_term_focus": 5.0
                },
                "financial_priorities": ["Emergency fund", "Debt reduction", "Retirement planning"],
                "investment_profile_type": "Balanced",  # Neutral default
                "confidence_score": 0.0,
                "insight_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Determine which template to use based on whether this is a behavioral question
            if behavioral_trait:
                template_name = "extract_behavioral_insights"
                prompt = self.prompt_templates[template_name]
                prompt = prompt.replace("{{behavioral_trait}}", behavioral_trait)
                prompt = prompt.replace("{{question}}", question)
                prompt = prompt.replace("{{answer}}", answer)
                logging.info(f"Using behavioral insights extraction for trait: {behavioral_trait}")
            else:
                template_name = "extract_insights"
                prompt = self.prompt_templates[template_name]
                prompt = prompt.replace("{{question}}", question)
                prompt = prompt.replace("{{answer}}", answer)
                
            # Call the LLM API
            response = self._call_llm_api(prompt)
            
            # Parse the insights with improved error handling
            try:
                insights = json.loads(response)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {str(e)}, Response content: {response[:100]}...")
                # Try to extract JSON from the response if it contains other text
                import re
                
                # Try different regex patterns to extract JSON
                json_patterns = [
                    r'({[\s\S]*})',     # Standard JSON object
                    r'(\[[\s\S]*\])',   # JSON array
                    r'```json([\s\S]*?)```',  # JSON in markdown code block
                    r'```([\s\S]*?)```'  # Any code block
                ]
                
                extracted_json = None
                for pattern in json_patterns:
                    json_match = re.search(pattern, response)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        try:
                            extracted_json = json.loads(json_str)
                            logging.info(f"Successfully extracted JSON using pattern: {pattern}")
                            insights = extracted_json
                            break
                        except json.JSONDecodeError:
                            continue
                
                # If we still couldn't parse JSON, return a fallback structure
                if extracted_json is None:
                    logging.error("Could not extract valid JSON from response, using fallback structure")
                    # Return a fallback structure
                    return {
                        "raw_answer": answer,
                        "question": question,
                        "extracted_facts": ["Could not parse LLM response as valid JSON"],
                        "error": f"JSON parsing error: {str(e)}",
                        "confidence_score": 0.0,
                        "insight_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Validate the insights against our schema
            validated_insights = self._validate_insights(insights)
            
            # Always include the original answer and question for reference
            validated_insights["raw_answer"] = answer
            validated_insights["question"] = question
            
            # Add a unique ID for this insight extraction
            validated_insights["insight_id"] = str(uuid.uuid4())
            
            # Add timestamp
            validated_insights["timestamp"] = datetime.now().isoformat()
            
            return validated_insights
            
        except Exception as e:
            logging.error(f"Error extracting insights: {str(e)}")
            return {
                "raw_answer": answer,
                "question": question,
                "error": str(e),
                "confidence_score": 0.0,
                "insight_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted insights against the standard schema.
        Ensures all metrics are in the correct format and range.
        
        Args:
            insights: Raw extracted insights from LLM
            
        Returns:
            Validated and normalized insights
        """
        validated = {}
        
        # Validate numeric scores in nested structures
        for category in ["risk_scores", "knowledge_scores", "goal_scores", "behavioral_indicators", "india_specific"]:
            if category in insights and isinstance(insights[category], dict):
                validated[category] = {}
                for metric, value in insights[category].items():
                    # Validate numeric scores are in range 0-10
                    if isinstance(value, (int, float)) and 0 <= value <= 10:
                        validated[category][metric] = value
        
        # Validate ranked list of financial priorities
        if "financial_priorities" in insights and isinstance(insights["financial_priorities"], list):
            valid_priorities = self.insight_schema["financial_priorities"]["options"]
            validated["financial_priorities"] = [p for p in insights["financial_priorities"] if p in valid_priorities]
        
        # Validate lists with maximum items
        for list_field in ["extracted_facts", "concerns", "opportunities"]:
            if list_field in insights and isinstance(insights[list_field], list):
                max_items = self.insight_schema[list_field].get("max_items", 5)
                validated[list_field] = insights[list_field][:max_items]
        
        # Validate time horizons
        if "time_horizons" in insights and isinstance(insights["time_horizons"], dict):
            validated["time_horizons"] = {}
            for horizon in ["short_term", "medium_term", "long_term"]:
                if horizon in insights["time_horizons"] and isinstance(insights["time_horizons"][horizon], list):
                    validated["time_horizons"][horizon] = insights["time_horizons"][horizon]
        
        # Validate investment profile type
        if "investment_profile_type" in insights:
            valid_types = self.insight_schema["investment_profile_type"]["options"]
            if insights["investment_profile_type"] in valid_types:
                validated["investment_profile_type"] = insights["investment_profile_type"]
        
        # Validate behavioral strengths and challenges
        if "behavioral_strengths" in insights and isinstance(insights["behavioral_strengths"], list):
            validated["behavioral_strengths"] = insights["behavioral_strengths"][:3]  # Limit to top 3
        
        if "behavioral_challenges" in insights and isinstance(insights["behavioral_challenges"], list):
            validated["behavioral_challenges"] = insights["behavioral_challenges"][:3]  # Limit to top 3
            
        # Validate behavioral summary
        if "behavioral_summary" in insights and isinstance(insights["behavioral_summary"], str):
            validated["behavioral_summary"] = insights["behavioral_summary"]
            
        # Validate primary bias
        if "primary_bias" in insights:
            validated["primary_bias"] = insights["primary_bias"]
        
        # Validate confidence score
        if "confidence_score" in insights and isinstance(insights["confidence_score"], (int, float)):
            if 0 <= insights["confidence_score"] <= 1:
                validated["confidence_score"] = insights["confidence_score"]
            else:
                validated["confidence_score"] = max(0, min(1, insights["confidence_score"]))  # Clamp to [0,1]
        else:
            # Default confidence if missing
            validated["confidence_score"] = 0.5
            
        return validated
    
    def analyze_answer_sentiment(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Analyze the sentiment and implications of a user's answer.
        
        Args:
            question: The question that was asked
            answer: The user's response
            
        Returns:
            Dictionary with sentiment analysis
        """
        if not self.enabled:
            return {
                "sentiment": "neutral", 
                "confidence": 0.5, 
                "risk_indication": "moderate", 
                "key_concern": "LLM service disabled - unable to analyze sentiment"
            }
        
        prompt = f"""
        Analyze the sentiment and financial implications of this response to a financial question.
        
        Question: {question}
        Response: {answer}
        
        Return your analysis as a JSON object with these fields:
        - sentiment: (positive, negative, neutral, or mixed)
        - confidence: (number from 0.0 to 1.0)
        - risk_indication: (conservative, moderate, aggressive, or unclear)
        - key_concern: (main financial concern implied, if any)
        
        IMPORTANT: The response MUST be a valid JSON object with no additional text.
        """
        
        try:
            response = self._call_llm_api(prompt)
            
            # Parse with improved error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse sentiment analysis as JSON: {str(e)}")
                
                # Try to extract JSON if wrapped in text
                import re
                json_match = re.search(r'({[\s\S]*})', response)
                
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        logging.error("Failed to extract valid JSON from sentiment analysis response")
                        result = {"sentiment": "neutral", "confidence": 0, "error": "JSON parsing failed", "raw_response": response[:100]}
                else:
                    result = {"sentiment": "neutral", "confidence": 0, "error": "No JSON found in response", "raw_response": response[:100]}
            
            # Ensure required fields are present
            for field in ["sentiment", "confidence", "risk_indication"]:
                if field not in result:
                    result[field] = "neutral" if field == "sentiment" else ("moderate" if field == "risk_indication" else 0)
            
            return result
        except Exception as e:
            logging.error(f"Error analyzing answer sentiment: {str(e)}")
            return {
                "sentiment": "neutral", 
                "confidence": 0, 
                "risk_indication": "moderate", 
                "error": str(e)
            }
    
    def personalize_question_text(self, question_text: str, profile: Dict[str, Any]) -> str:
        """
        Personalize question text by filling in placeholders with profile data.
        
        Args:
            question_text: Question text template with placeholders
            profile: User profile with answers
            
        Returns:
            Personalized question text
        """
        personalized_text = question_text
        
        # Look for placeholders in the format {{answer:question_id}}
        import re
        placeholders = re.findall(r"{{answer:(.*?)}}", question_text)
        
        if placeholders:
            # Get the answers from the profile
            answers_dict = {a.get('question_id'): a.get('answer') for a in profile.get('answers', [])}
            
            # Replace each placeholder
            for placeholder in placeholders:
                if placeholder in answers_dict:
                    personalized_text = personalized_text.replace(
                        f"{{{{answer:{placeholder}}}}}",
                        str(answers_dict[placeholder])
                    )
        
        return personalized_text
    
    def _call_llm_api(self, prompt: str, use_cache: bool = True, timeout: int = 30) -> str:
        """
        Call the LLM API with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM API
            use_cache: Whether to use the cache (default True)
            timeout: Timeout in seconds for the API call (default 30)
            
        Returns:
            The response text from the API
        """
        if not self.api_key:
            logging.error("API key is not set. Unable to call LLM API.")
            raise ValueError("LLM API key is not configured. Please set OPENAI_API_KEY environment variable.")
        
        self.total_calls += 1
        
        # Generate a cache key from the prompt and model
        cache_key = f"{self.model}:{hash(prompt)}"
        
        # Check if we have a cached response
        if use_cache and cache_key in self.response_cache:
            self.cache_hits += 1
            hit_rate = (self.cache_hits / self.total_calls) * 100
            logging.info(f"Cache hit for prompt. Hit rate: {hit_rate:.1f}%")
            return self.response_cache[cache_key]
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial advisor AI that generates follow-up questions and analyzes financial responses. ALWAYS return responses in valid JSON format. For generating questions, return a list of question objects."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Low temperature for more consistent outputs
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # Explicitly request JSON format
        }
        
        max_retries = 2
        retry_count = 0
        retry_delay = 2  # seconds
        
        while retry_count <= max_retries:
            try:
                logging.info(f"Making OpenAI API call to {self.base_url}/chat/completions with model {self.model}")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                logging.info(f"OpenAI API response status code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        content = response.json()
                        logging.info(f"Successfully parsed API response to JSON, response length: {len(str(content))}")
                        
                        if "choices" in content and len(content["choices"]) > 0:
                            logging.info(f"API response contains {len(content['choices'])} choices")
                            
                            if "message" in content["choices"][0] and "content" in content["choices"][0]["message"]:
                                result = content["choices"][0]["message"]["content"]
                                logging.info(f"Extracted message content from API response, length: {len(result)}")
                                
                                # Log a preview of the result for debugging
                                preview = result[:100] + "..." if len(result) > 100 else result
                                logging.info(f"API response preview: {preview}")
                                
                                # Cache the result for future use
                                if use_cache:
                                    # Manage cache size - remove oldest entry if at capacity
                                    if len(self.response_cache) >= self.cache_size:
                                        oldest_key = next(iter(self.response_cache))
                                        del self.response_cache[oldest_key]
                                    
                                    self.response_cache[cache_key] = result
                                    logging.info(f"Cached new response. Cache size: {len(self.response_cache)}/{self.cache_size}")
                                
                                return result
                            else:
                                # Log more details about the structure when expected fields are missing
                                logging.error(f"API response missing message or content fields. Response structure: {json.dumps(content['choices'][0]) if content.get('choices') else 'No choices'}")
                                raise Exception("Invalid API response structure: missing message or content")
                        else:
                            logging.error(f"No choices in API response: {content}")
                            raise Exception("No choices in API response")
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse API response as JSON: {str(e)}")
                        raise Exception(f"Failed to parse API response: {str(e)}")
                elif response.status_code == 429:  # Rate limit error
                    if retry_count < max_retries:
                        retry_count += 1
                        logging.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logging.error(f"Rate limit exceeded after {max_retries} retries")
                        raise Exception(f"Rate limit exceeded after {max_retries} retries")
                else:
                    error_msg = f"API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data and "message" in error_data["error"]:
                            error_msg += f", {error_data['error']['message']}"
                    except:
                        error_msg += f", {response.text}"
                    
                    logging.error(error_msg)
                    raise Exception(error_msg)
                    
            except requests.exceptions.RequestException as e:
                if retry_count < max_retries:
                    retry_count += 1
                    logging.warning(f"Request error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    logging.error(f"Request error after {max_retries} retries: {str(e)}")
                    raise Exception(f"Request error after {max_retries} retries: {str(e)}")
            
            # If we get here, we've either succeeded or failed with no retry needed
            break
    
    # Mock implementation for testing without API
    def generate_mock_next_level_questions(self, category: str) -> List[Dict[str, Any]]:
        """Generate mock next-level questions for testing without an API key"""
        mock_questions = {
            "demographics": [
                {
                    "question_id": "mock_nl_demographics_dependents",
                    "id": "mock_nl_demographics_dependents",  # Ensure both id and question_id exist
                    "text": "You mentioned having dependents. Could you share more about how their financial needs impact your planning?",
                    "category": "demographics",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 101,
                    "related_to": "demographics_dependents"
                }
            ],
            "financial_basics": [
                {
                    "question_id": "mock_nl_financial_savings",
                    "id": "mock_nl_financial_savings",  # Ensure both id and question_id exist
                    "text": "Could you share how your current savings are distributed across different asset classes?",
                    "category": "financial_basics",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 101,
                    "related_to": "financial_basics_current_savings"
                }
            ],
            "assets_and_debts": [
                {
                    "question_id": "mock_nl_assets_debts",
                    "id": "mock_nl_assets_debts",  # Ensure both id and question_id exist
                    "text": "What types of debt do you currently have, and what are their interest rates?",
                    "category": "assets_and_debts",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 101,
                    "related_to": "assets_debts_total_debt"
                }
            ],
            "special_cases": [
                {
                    "question_id": "mock_nl_special_cases",
                    "id": "mock_nl_special_cases",  # Ensure both id and question_id exist
                    "text": "How integrated is your business with your personal finances?",
                    "category": "special_cases",
                    "type": "next_level",
                    "input_type": "text",
                    "required": False,
                    "order": 101,
                    "related_to": "special_cases_business_value"
                }
            ]
        }
        
        # Add timestamp to question IDs to ensure uniqueness
        # This helps avoid potential key conflicts in question caches
        timestamp = int(time.time())
        questions = mock_questions.get(category, [])
        
        for q in questions:
            # Update IDs with timestamp to ensure uniqueness
            unique_id = f"{q['question_id']}_{timestamp}"
            q['id'] = unique_id
            q['question_id'] = unique_id
            
        return questions