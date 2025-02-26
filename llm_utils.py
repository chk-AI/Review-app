import json
import re
import time
from openai import OpenAI


def get_model_params(model_name, base_params=None, **kwargs):
    """
    Returns appropriate parameters for mini LLM models.
    
    Args:
        model_name (str): The name of the model ('o1-mini' or 'o3-mini')
        base_params (dict, optional): Base parameters to extend
        **kwargs: Additional parameters with their values
    
    Returns:
        dict: Complete parameters dictionary for the API call
    """
    # Start with empty dict if no base_params provided
    params = base_params.copy() if base_params else {}
    
    # Add model name
    params["model"] = model_name
    
    # Default values
    default_max_tokens = 500
    
    # Get values from kwargs or use defaults
    max_tokens_value = kwargs.get('max_tokens', default_max_tokens)
    
    # Model-specific parameter handling
    if model_name == "o1-mini" or model_name == "o3-mini":
        # o1-mini and o3-mini use max_completion_tokens and don't use temperature
        params["max_completion_tokens"] = max_tokens_value
        
        # Remove temperature if present in base_params
        if "temperature" in params:
            del params["temperature"]
            
        # Remove max_tokens if present in base_params
        if "max_tokens" in params:
            del params["max_tokens"]
        
        # Add reasoning_effort for o3-mini
        if model_name == "o3-mini":
            params["reasoning_effort"] = kwargs.get("reasoning_effort", "low")
    
    # Remove response_format for o1-mini and o3-mini which don't support it
    if "response_format" in params:
        del params["response_format"]
    
    # Handle message format for o1-mini which doesn't support system role
    if model_name == "o1-mini" and "messages" in params:
        fixed_messages = []
        for i, msg in enumerate(params["messages"]):
            if msg["role"] == "system":
                # Convert system message to user message with a prefix
                if i < len(params["messages"]) - 1 and params["messages"][i+1]["role"] == "user":
                    # If the next message is a user message, we'll combine them
                    continue
                else:
                    fixed_messages.append({
                        "role": "user", 
                        "content": f"You are acting as: {msg['content']}\n\nPlease respond to the following:"
                    })
            elif msg["role"] == "user" and i > 0 and params["messages"][i-1]["role"] == "system":
                # This is a user message that follows a system message
                system_content = params["messages"][i-1]["content"]
                fixed_messages.append({
                    "role": "user",
                    "content": f"You are acting as: {system_content}\n\n{msg['content']}"
                })
            else:
                fixed_messages.append(msg)
        params["messages"] = fixed_messages
            
    # Add any other kwargs not already set
    for key, value in kwargs.items():
        if key not in ["max_tokens", "reasoning_effort"] and key not in params:
            params[key] = value
            
    return params

def sanitize_column_name(name):
    """Convert a feature question into a valid (snake_case) column name."""
    clean_name = re.sub(r'[^a-zA-Z0-9_\s]', '', name)
    clean_name = clean_name.lower().strip()
    clean_name = re.sub(r'\s+', '_', clean_name)
    return clean_name

def create_llm_prompt(text, features, possible_answers):
    """
    Creates a structured prompt with possible answers for each feature.
    Also returns a mapping of feature -> sanitized_column_name.
    """
    prompt = (
        "You are analyzing a scientific paper. For each question below, "
        "answer ONLY using one of the provided possible answers. "
        "If none of the answers fit or if you're unsure, use 'unclear'.\n\n"
        f"Title and Abstract to analyze:\n{text}\n\n"
        "Questions:\n"
    )
    
    # Create a dictionary mapping each feature to its normalized key
    feature_keys = {feature: sanitize_column_name(feature) for feature in features}
    
    for i, (feature, answers) in enumerate(zip(features, possible_answers), 1):
        answer_list = "', '".join(answers)
        prompt += f"{i}. {feature}\n   Possible answers: ['{answer_list}']\n\n"
    
    prompt += "\nRespond with a JSON object where each key matches the *exact* question text. Example:\n"
    prompt += "{\n"
    for feature in features:
        prompt += f'    "{feature}": "your_answer",\n'
    prompt += "}\n"
    
    return prompt, feature_keys

def analyze_with_llm(text, features, possible_answers, model_name, llm_api_key):
    """
    Sends a structured prompt to the chosen LLM model and extracts structured answers.
    """
    client = OpenAI(api_key=llm_api_key)
    prompt, feature_keys = create_llm_prompt(text, features, possible_answers)
    
    try:
        # Create base parameters
        if model_name == "o1-mini":
            # o1-mini doesn't support system messages or response_format
            base_params = {
                "messages": [
                    {"role": "user", "content": "You are a scientific paper analyzer that provides concise, structured answers.\n\n" + prompt}
                ]
            }
        elif model_name == "o3-mini":
            # o3-mini supports system message but not response_format
            base_params = {
                "messages": [
                    {"role": "system", "content": "You are a scientific paper analyzer that provides concise, structured answers."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning_effort": "low"
            }
        
        # Get model parameters for mini models
        params = get_model_params(model_name, base_params)
            
        response = client.chat.completions.create(**params)
        
        # Parse the JSON response
        result = {}
        raw_content = response.choices[0].message.content
        try:
            # We need to parse the JSON from the text
            if raw_content:
                result = json.loads(raw_content)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            # Try to extract a JSON object if the response contains one
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, raw_content)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    # If still fails, create a basic result with error info
                    result = {f: "extraction_error" for f in features}
            else:
                # No JSON found, create error result
                result = {f: "extraction_error" for f in features}
        
        # Create a normalized result using the feature keys
        normalized_result = {}
        for feature in features:
            value = result.get(feature, "unclear")
            normalized_result[feature_keys[feature]] = value
                
        return normalized_result

    except Exception as e:
        print(f"Error in analyze_with_llm: {str(e)}")
        return {sanitize_column_name(f): f"Error: {str(e)}" for f in features}

def suggest_pubmed_search_strategy(research_question, search_type, model_name, llm_api_key):
    """
    Given a research question and search type, ask the LLM to suggest a PubMed search string.
    """
    client = OpenAI(api_key=llm_api_key)
    
    # Customize prompt based on search type
    if search_type == 'broad':
        strategy_guidance = """
        Create a broad PubMed search strategy that:
        - Includes synonyms and related terms
        - Uses OR operators liberally to capture related concepts
        - Minimizes restrictions on study types or other limiters
        - Aims for high sensitivity (recall) over precision
        """
    else:  # narrow
        strategy_guidance = """
        Create a focused PubMed search strategy that:
        - Uses specific, precise terms
        - Includes appropriate methodological filters
        - Uses AND operators to combine key concepts
        - Aims for high precision over sensitivity
        """
    
    # Include JSON instructions directly in the prompt
    json_instructions = "\n\nFormat your response as a valid JSON object exactly like this example (with no additional text before or after the JSON):\n{\"search_string\": \"your search string here\", \"explanation\": \"your explanation here\", \"estimated_results\": \"low/medium/high\"}"
    
    prompt = f"""Given this research question: "{research_question}"

{strategy_guidance}

Return the following information:
1. A complete PubMed search string
2. A brief explanation of the search strategy and key terms used
3. Rough estimate of expected number of results (low/medium/high){json_instructions}"""

    try:
        # Base system message
        system_message = "You are a medical librarian expert in creating PubMed search strategies."
        
        # Create messages and parameters based on model
        if model_name == "o1-mini":
            # o1-mini doesn't support system messages
            base_params = {
                "messages": [
                    {"role": "user", "content": f"{system_message}\n\n{prompt}"}
                ]
            }
        elif model_name == "o3-mini":
            # o3-mini supports system message
            base_params = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "reasoning_effort": "low"
            }
        
        # Get parameters for mini models
        params = get_model_params(model_name, base_params, max_tokens=1000)
            
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Try to extract the JSON from the response
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, content)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return result
            except json.JSONDecodeError as e:
                return {
                    "search_string": "",
                    "explanation": f"Error parsing response as JSON: {str(e)}\nOriginal response: {content}",
                    "estimated_results": "unknown"
                }
        else:
            # No JSON found, create a structured result from the text if possible
            # Try to extract search string, explanation, and estimated results using patterns
            search_string = ""
            explanation = ""
            estimated_results = "unknown"
            
            # Very basic parsing - in a real app, you'd want more robust extraction
            if "search string" in content.lower() or "search strategy" in content.lower():
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "search string" in line.lower() or "search strategy" in line.lower():
                        if i+1 < len(lines) and lines[i+1].strip():
                            search_string = lines[i+1].strip()
                    if "explanation" in line.lower():
                        explanation_lines = []
                        j = i+1
                        while j < len(lines) and not any(keyword in lines[j].lower() for keyword in ["estimate", "results"]):
                            explanation_lines.append(lines[j])
                            j += 1
                        explanation = "\n".join(explanation_lines).strip()
                    if "estimate" in line.lower() or "results" in line.lower():
                        if "low" in line.lower():
                            estimated_results = "low"
                        elif "medium" in line.lower():
                            estimated_results = "medium"
                        elif "high" in line.lower():
                            estimated_results = "high"
            
            return {
                "search_string": search_string,
                "explanation": explanation or content,  # Default to full content if no explanation found
                "estimated_results": estimated_results
            }
    except Exception as e:
        return {
            "search_string": "",
            "explanation": f"Error generating search strategy: {str(e)}",
            "estimated_results": "unknown"
        }

def suggest_pico_features(research_question, model_name, llm_api_key):
    """
    Ask the LLM to identify PICO elements from a research question for a systematic review.
    
    Args:
        research_question (str): The research question to analyze
        model_name (str): The LLM model to use
        llm_api_key (str): API key
    
    Returns:
        dict: Dictionary with PICO components (population, intervention, comparison, outcome)
    """
    client = OpenAI(api_key=llm_api_key)
    
    # Create a prompt that asks for PICO identification
    prompt = f"""Analyze this research question for a systematic review: "{research_question}"

Extract the PICO elements (Population, Intervention, Comparison, Outcome) from this question.
Focus on identifying:
1. Population/Problem: What patient group or population is being studied?
2. Intervention/Exposure: What main intervention or exposure is being evaluated?
3. Comparison: What is the intervention being compared to (if applicable)?
4. Outcome: What outcomes or effects are being measured?

Respond with a valid JSON object using this exact format (no additional text before or after):
{{
  "population": "description of the population/patient group",
  "intervention": "description of the intervention/exposure",
  "comparison": "description of the comparison (or 'not specified' if none)",
  "outcome": "description of the outcome measures"
}}

Provide clear, specific descriptions that would be suitable for screening studies (about 5-15 words per element)."""

    try:
        # Set up parameters based on model
        if model_name == "o1-mini":
            # o1-mini doesn't support system messages
            base_params = {
                "messages": [
                    {"role": "user", "content": "You are a systematic review methodology expert who extracts PICO elements from research questions.\n\n" + prompt}
                ]
            }
        elif model_name == "o3-mini":
            # o3-mini supports system message
            base_params = {
                "messages": [
                    {"role": "system", "content": "You are a systematic review methodology expert who extracts PICO elements from research questions."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning_effort": "low"
            }
        
        # Get parameters for mini models
        params = get_model_params(model_name, base_params, max_tokens=500)
            
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Try to extract the JSON from the response
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, content)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                # Ensure all PICO elements are present
                for element in ['population', 'intervention', 'comparison', 'outcome']:
                    if element not in result:
                        result[element] = "not specified"
                return result
            except json.JSONDecodeError:
                # Return default values if JSON parsing fails
                return {
                    "population": "relevant patient population",
                    "intervention": "primary intervention or exposure",
                    "comparison": "comparison or control",
                    "outcome": "relevant outcome measures"
                }
        else:
            # If no JSON found, create a default response
            return {
                "population": "relevant patient population",
                "intervention": "primary intervention or exposure",
                "comparison": "comparison or control",
                "outcome": "relevant outcome measures"
            }
    
    except Exception as e:
        print(f"Error in suggest_pico_features: {str(e)}")
        # Return default values in case of any error
        return {
            "population": "relevant patient population",
            "intervention": "primary intervention or exposure",
            "comparison": "comparison or control",
            "outcome": "relevant outcome measures"
        }
   
def suggest_features_for_extraction(research_question, model_name, llm_api_key):
    """
    Ask the LLM to suggest relevant features (data items) to extract for a scoping review 
    based on the given research question.
    """
    client = OpenAI(api_key=llm_api_key)
    
    # Define the user prompt
    user_prompt = (
        f"Given the research question:\n'{research_question}'\n"
        "Suggest a list of important data items (features) we should extract from each study. "
        "Provide them in a concise numbered list, with each item as a short phrase."
    )
    
    # Create messages and parameters based on model
    if model_name == "o1-mini":
        # o1-mini doesn't support system messages
        base_params = {
            "messages": [
                {"role": "user", "content": "You are acting as a scoping review methodology expert.\n\n" + user_prompt}
            ]
        }
    elif model_name == "o3-mini":
        # o3-mini supports system message but add reasoning_effort
        base_params = {
            "messages": [
                {"role": "system", "content": "You are a scoping review methodology expert."},
                {"role": "user", "content": user_prompt}
            ],
            "reasoning_effort": "low"
        }
    
    # Get parameters for mini models with shorter max tokens
    params = get_model_params(model_name, base_params, max_tokens=200)
        
    response = client.chat.completions.create(**params)

    return response.choices[0].message.content.strip()