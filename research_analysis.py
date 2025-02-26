import tiktoken
from openai import OpenAI
import pandas as pd

def format_citation(authors, year):
    """Format citation in 'Author et al., Year' style."""
    if not authors or not year:
        return "Unknown"
    
    # Split authors and get first author's lastname
    first_author = authors.split(',')[0].strip().split()[-1]
    return f"{first_author} et al., {year}"

def format_vancouver_reference(row, ref_number):
    """Format a reference in Vancouver style."""
    # Extract all authors
    authors = row['Authors'].split(', ')
    if len(authors) > 6:
        # If more than 6 authors, list first 6 followed by "et al."
        author_str = ', '.join(authors[:6]) + ', et al.'
    else:
        author_str = ', '.join(authors)
    
    # Format the reference
    reference = (
        f"{ref_number}. {author_str}. "
        f"{row['Title']}. "
        f"{row['Journal']} {row['Year']}"
    )
    
    # Add DOI if available
    if row['DOI'] and row['DOI'] != 'Not available':
        reference += f". DOI: {row['DOI']}"
    
    # Add PMID
    if row['PMID']:
        reference += f". PMID: {row['PMID']}"
    
    return reference

def create_analysis_prompt(research_question, df):
    """Create a structured prompt for analyzing the filtered papers."""
    # Prepare paper summaries with citations and reference list
    paper_summaries = []
    references = []
    
    for idx, row in df.iterrows():
        citation = format_citation(row['Authors'], row['Year'])
        ref_num = idx + 1
        
        summary = (
            f"Title: {row['Title']}\n"
            f"Abstract: {row['Abstract']}\n"
            f"Citation: {citation} [ref {ref_num}]\n"
            f"PMID: {row['PMID']}\n"
            "---"
        )
        paper_summaries.append(summary)
        
        # Create Vancouver style reference
        references.append(format_vancouver_reference(row, ref_num))

    prompt = f"""Given the research question: "{research_question}"

You are analyzing a specific set of {len(df)} papers listed below. It is CRUCIAL that you:
1. ONLY use information from these papers
2. ONLY cite these papers
3. Do NOT introduce or reference any external papers or knowledge

Papers to analyze:
{'\n\n'.join(paper_summaries)}

Provide a structured analysis with the following sections:

# Key Evidence Found
- Present the main findings relevant to the research question
- Use in-text citations in the format (Author et al., Year [ref X])
- Every claim must be supported by specific citations from the provided papers

# Gaps in Evidence
- Identify what aspects of the research question aren't well addressed by these papers
- Focus on gaps within the scope of the provided papers

# Conclusion
- Summarize only what can be concluded from these specific papers
- Be explicit about limitations

# References
Use this numbered reference list in your citations:
{'\n'.join(references)}

Important rules:
- Every claim must cite specific papers from the provided list
- Use ONLY the papers provided - do not reference any external sources
- Use citation format: (Author et al., Year [ref X])
- If insufficient evidence exists to answer any aspect of the question, explicitly state this
- Format your response in markdown
- Maintain academic rigor and precision"""

    return prompt

def analyze_filtered_results(research_question, filtered_df, model_name, api_key, max_tokens=4000):
    """
    Analyze filtered papers in relation to the research question.
    Returns the analysis and a boolean indicating if it was truncated.
    Max tokens increased to 4000 to accommodate full analysis with references.
    """
    client = OpenAI(api_key=api_key)
    
    # Import here to avoid circular imports
    from llm_utils import get_model_params
    
    # Create the analysis prompt
    prompt = create_analysis_prompt(research_question, filtered_df)
    
    try:
        # Set up parameters based on model
        if model_name == "o1-mini":
            # o1-mini doesn't support system messages
            base_params = {
                "messages": [
                    {"role": "user", "content": "You are a precise scientific researcher who only makes claims with direct evidence from the provided papers.\n\n" + prompt}
                ]
            }
        elif model_name == "o3-mini":
            # o3-mini supports system message but needs reasoning_effort
            base_params = {
                "messages": [
                    {"role": "system", "content": "You are a precise scientific researcher who only makes claims with direct evidence from the provided papers."},
                    {"role": "user", "content": prompt}
                ],
                "reasoning_effort": "medium"  # Use medium for more thorough analysis
            }
        
        # Get parameters for mini models with low temperature for factual responses
        params = get_model_params(model_name, base_params, max_tokens=max_tokens)
        
        response = client.chat.completions.create(**params)
        
        analysis = response.choices[0].message.content
        
        # Check if response was truncated
        was_truncated = False
        
        try:
            # Use cl100k_base encoding for the new models
            encoder = tiktoken.get_encoding("cl100k_base")
            tokens_used = len(encoder.encode(analysis))
            
            # Check if we reached max tokens based on model
            if model_name == "o1-mini" or model_name == "o3-mini":
                was_truncated = tokens_used >= params.get("max_completion_tokens", max_tokens)
                
        except Exception as token_error:
            print(f"Warning: Could not check token count: {token_error}")
            was_truncated = False  # Default to false if we can't check
        
        return analysis, was_truncated
        
    except Exception as e:
        error_msg = (
            "# Error in Analysis\n\n"
            f"An error occurred while analyzing the results: {str(e)}\n\n"
            "Please try again or contact support if the problem persists."
        )
        return error_msg, False