import streamlit as st
from openai import OpenAI
import json
import re

def suggest_research_questions(subject, model_name, api_key):
    """
    Generate research question suggestions based on a subject area.
    
    Args:
        subject (str): The subject area of interest
        model_name (str): The LLM model to use
        api_key (str): OpenAI API key
    
    Returns:
        list: List of suggested research questions
    """
    client = OpenAI(api_key=api_key)
    
    # Import here to avoid circular imports
    from llm_utils import get_model_params
    
    # Adjust the prompt based on model to help with JSON output
    json_format_instructions = "\n\nProvide your response as valid JSON with this exact format (no additional text before or after the JSON):\n{\"suggestions\": [\"question1\", \"question2\", \"question3\"]}"
    
    prompt = f"""Given the subject area: {subject}
    
    Suggest 3 specific, well-formed research questions suitable for a literature review.
    Each question should:
    1. Be specific and answerable
    2. Include a clear population, intervention/exposure, and outcome
    3. Be suitable for systematic review methodology{json_format_instructions}"""
    
    try:
        # Set up parameters based on model
        if model_name == "o1-mini":
            # o1-mini doesn't support system messages
            base_params = {
                "messages": [
                    {"role": "user", "content": "You are acting as a research methodology expert.\n\n" + prompt}
                ]
            }
        elif model_name == "o3-mini":
            # o3-mini supports system message but needs reasoning_effort
            base_params = {
                "messages": [
                    {"role": "system", "content": "You are a research methodology expert."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning_effort": "low"
            }
        
        # Get parameters for mini models
        params = get_model_params(model_name, base_params, max_tokens=800)
        
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Try to extract a JSON object from the response
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, content)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return result.get("suggestions", [])
            except json.JSONDecodeError:
                # If we can't parse the JSON, try to extract the questions manually
                questions = []
                question_pattern = r'\d+\.\s*(.*?)\s*(?=\d+\.|$)'
                questions_match = re.findall(question_pattern, content)
                if questions_match:
                    return [q.strip() for q in questions_match]
                return [f"Error parsing questions from: {content}"]
        else:
            # No JSON found, try to extract questions manually
            questions = []
            lines = content.strip().split('\n')
            for line in lines:
                if re.match(r'^\d+\.', line) or '?' in line:
                    questions.append(line.strip())
            
            if questions:
                return questions
            return [f"Error extracting questions from: {content}"]
            
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []