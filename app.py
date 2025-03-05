import streamlit as st
import pandas as pd
import json
import time
import concurrent.futures
from datetime import datetime

# Import custom modules
from pubmed_utils import search_pubmed, fetch_details, papers_to_df
from llm_utils import (
    sanitize_column_name,
    suggest_pubmed_search_strategy,
    suggest_features_for_extraction,
    suggest_pico_features,
    analyze_with_llm
)
from filter_utils import filter_dataframe
from research_analysis import analyze_filtered_results
from visualization_utils import display_visualizations
from search_utils import SearchManager, render_advanced_search
from progress_utils import render_progress_bar, get_step_description

def initialize_session_state():
    """Initialize all session state variables if they don't exist."""
    # Initialize review type if not set
    if 'review_type' not in st.session_state:
        st.session_state.review_type = None
        
    # Initialize features with an empty list if not set
    if 'features' not in st.session_state:
        st.session_state.features = []
    
    # Remove credentials from session state if page is refreshed
    st.session_state.llm_api_key = None
    st.session_state.pubmed_email = None
    
    # Add hypothesis initialization
    if 'hypothesis' not in st.session_state:
        st.session_state.hypothesis = ""
        
    if 'full_df' not in st.session_state:
        st.session_state.full_df = None
        
    if 'unfiltered_df' not in st.session_state:
        st.session_state.unfiltered_df = None
        
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
        
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = None
        
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
        
    if 'analysis_truncated' not in st.session_state:
        st.session_state.analysis_truncated = False
    
    # Initialize current_step for progress tracking
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
        
    # Initialize flag to track if PICO features have been generated
    if 'pico_features_generated' not in st.session_state:
        st.session_state.pico_features_generated = False

def handle_pubmed_search(query, email):
    """Handle PubMed search and store results in session state."""
    with st.spinner('Searching PubMed...'):
        try:
            pmids, count = search_pubmed(query, email)
            papers = fetch_details(pmids, email)
            df = papers_to_df(papers)
            st.session_state.full_df = df
            st.session_state.search_results = {'count': count, 'df': df}
            
            # Update progress step if search is successful
            if st.session_state.current_step < 2:
                st.session_state.current_step = 2
                
            return count
        except Exception as e:
            st.error(f"Error during PubMed search: {str(e)}")
            return None

def handle_batch_analysis(df_batch, features, possible_answers, model_name, llm_api_key, run_in_parallel):
    """Handle batch analysis of papers with progress tracking."""
    sanitized_names = [sanitize_column_name(q) for q in features]
    
    # Initialize columns
    for col in sanitized_names:
        df_batch[col] = None
    
    def analyze_paper(row):
        text = f"Title: {row['Title']}\nAbstract: {row['Abstract']}"
        return analyze_with_llm(text, features, possible_answers, model_name, llm_api_key)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if run_in_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(analyze_paper, row): idx 
                for idx, row in df_batch.iterrows()
            }
            for i, future in enumerate(concurrent.futures.as_completed(future_to_index)):
                idx = future_to_index[future]
                analysis_res = future.result()
                for col_name, val in analysis_res.items():
                    df_batch.at[idx, col_name] = val
                progress = (i+1)/len(future_to_index)
                progress_bar.progress(progress)
                status_text.text(f"Analyzed {i+1} of {len(df_batch)} papers...")
    else:
        for i, row in df_batch.iterrows():
            analysis_res = analyze_paper(row)
            for col_name, val in analysis_res.items():
                df_batch.at[i, col_name] = val
            progress = (i - df_batch.index[0] + 1)/len(df_batch)
            progress_bar.progress(progress)
            status_text.text(f"Analyzed {i - df_batch.index[0] + 1} of {len(df_batch)} papers...")
            time.sleep(0.1)
    
    status_text.text("Analysis complete!")
    
    # Update progress step
    if st.session_state.current_step < 4:
        st.session_state.current_step = 4
        
    return df_batch

def select_review_type():
    """Display UI for selecting the review type."""
    st.subheader("Select Review Type")
    
    # Add descriptive information to help users choose
    st.write("""
    Please select the type of review you're conducting:
    
    **Scoping Review**: Best for mapping evidence on a topic and identifying main concepts, 
    theories, sources, and knowledge gaps. Use this when you want to explore the full scope of 
    literature on a topic.
    
    **Systematic Review**: Best for answering a specific clinical question using precisely defined
    PICO criteria (Population, Intervention, Comparison, Outcome). Use this when you need a 
    comprehensive, unbiased synthesis of evidence on a focused question.
    """)
    
    # Create selection buttons in columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Scoping Review", use_container_width=True):
            st.session_state.review_type = "scoping"
            st.session_state.features = [{
                'question': 'Does this study appear to examine ____?',
                'answers': 'yes, no, unclear'
            }]
            st.rerun()
            
    with col2:
        if st.button("Systematic Review", use_container_width=True):
            st.session_state.review_type = "systematic"
            st.session_state.features = [
                {'question': 'Does this study examine ___?', 'answers': 'yes, no, unclear'},
                {'question': 'Does this study use ___?', 'answers': 'yes, no, unclear'},
                {'question': 'Does this study compare ___?', 'answers': 'yes, no, unclear'},
                {'question': 'Does this study measure ___?', 'answers': 'yes, no, unclear'}
            ]
            st.rerun()

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    st.title("Literature Review Assistant")
    
    # Add a warning about credential handling
    st.info("ℹ️ Your API key and email are only stored temporarily in your browser session.")
    
    # If review type is not selected yet, show the selection UI
    if st.session_state.review_type is None:
        select_review_type()
        return
    
    # Update title based on review type
    if st.session_state.review_type == "scoping":
        st.title("Scoping Review Assistant")
    else:
        st.title("Systematic Review Assistant")
    
    # Display progress tracking
    st.markdown("### Progress Steps")
    render_progress_bar(st.session_state.current_step)
    
    # Step 1: Research Question input
    st.subheader("1) Research question")
    
    # API Configuration first
    llm_api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        help="Your API key is only stored temporarily in your browser session"
    )
    if llm_api_key:
        st.session_state.llm_api_key = llm_api_key
    
    model_name = st.selectbox(
        "Choose an LLM model:",
        ["o1-mini", "o3-mini"],
        index=0,
        help="Choose the model for API calls"
    )
    st.session_state.model_name = model_name
    
    # Manual research question input
    research_question = st.text_area(
        "Research Question:",
        value=st.session_state.get('research_question', '')
    )
    if research_question:
        st.session_state.research_question = research_question
        # Move to next step if we have a research question
        if st.session_state.current_step < 2:
            st.session_state.current_step = 2
    
    hypothesis = st.text_area(
        "Hypothesis (optional):",
        value=st.session_state.get('hypothesis', ''),
        help="If you have a specific hypothesis to test, enter it here. Example: 'The short term mortality is lower than 1%'"
    )
    if hypothesis:
        st.session_state.hypothesis = hypothesis

    # Step 2: PubMed Search
    st.subheader("2) PubMed search")
    
    # Search strategy generation
    st.write("### Generate Search Strategy")
    search_type = st.radio(
        "Search Type",
        ["broad", "narrow"],
        help="Broad search prioritizes recall, narrow search prioritizes precision"
    )
    
    if st.button("Generate Search Strategy"):
        if not st.session_state.get('research_question'):
            st.warning("Please define a research question first.")
        elif not st.session_state.get('llm_api_key'):
            st.warning("Please provide an API key first.")
        else:
            with st.spinner('Generating search strategy...'):
                strategy = suggest_pubmed_search_strategy(
                    st.session_state.research_question,
                    search_type,
                    st.session_state.model_name,
                    st.session_state.llm_api_key
                )
                st.write("**Suggested Search Strategy:**")
                st.code(strategy['search_string'])
                st.write("**Explanation:**")
                st.write(strategy['explanation'])
                st.write(f"**Estimated Results:** {strategy['estimated_results']}")
                
                # Add a button to use this search string
                if st.button("Use this search string"):
                    st.session_state.search_params['terms'] = strategy['search_string']
                    st.rerun()
    
    st.write("### Manual Search Configuration")
    # Advanced search interface
    query = render_advanced_search()
    
    email = st.text_input(
        "Enter email (required by NCBI):",
        help="Your email is only used for PubMed queries and is not stored"
    )
    
    if st.button("Search PubMed"):
        if not st.session_state.search_params['terms'].strip():
            st.warning("Please provide search terms.")
        elif not email.strip():
            st.warning("Please provide an email address.")
        else:
            count = handle_pubmed_search(query, email)
            if count is not None:
                st.write(f"Number of results found: {count}")
                if st.session_state.search_results is not None:
                    st.write(st.session_state.search_results['df'].head())
                    display_visualizations(st.session_state.search_results['df'])

    # Step 3: Feature Definition with automatic generation based on review type
    st.subheader("3) Define features to extract")
    
    # Generate features based on review type and research question
    if (st.session_state.research_question and st.session_state.llm_api_key and 
        (not st.session_state.pico_features_generated) and 
        st.button("Generate features based on research question")):
        
        with st.spinner('Generating feature suggestions...'):
            if st.session_state.review_type == "scoping":
                # For scoping review, just update the first feature
                main_question = f"Does this study appear to examine {st.session_state.research_question}?"
                if st.session_state.features and len(st.session_state.features) > 0:
                    st.session_state.features[0]['question'] = main_question
                else:
                    st.session_state.features = [{
                        'question': main_question,
                        'answers': 'yes, no, unclear'
                    }]
            else:
                # For systematic review, generate and update PICO features
                pico_features = suggest_pico_features(
                    st.session_state.research_question,
                    st.session_state.model_name,
                    st.session_state.llm_api_key
                )
                
                # Make sure we have 4 features
                while len(st.session_state.features) < 4:
                    st.session_state.features.append({
                        'question': '',
                        'answers': 'yes, no, unclear'
                    })
                    
                # Update the features with PICO-specific questions
                for i, component in enumerate(['population', 'intervention', 'comparison', 'outcome']):
                    if i < len(pico_features) and component in pico_features:
                        st.session_state.features[i]['question'] = (
                            f"Does this study appear to examine {pico_features[component]}?"
                        )
            
            st.session_state.pico_features_generated = True
            st.success("Features generated based on research question!")
    
    # Feature management
    col_add, col_remove = st.columns([1,1])
    with col_add:
        if st.button("Add Feature"):
            st.session_state.features.append({'question': '', 'answers': ''})
            # Update progress step
            if st.session_state.current_step < 3:
                st.session_state.current_step = 3
                
    with col_remove:
        if st.button("Remove Last Feature") and len(st.session_state.features) > 0:
            st.session_state.features.pop()
    
    # Feature input forms
    features = []
    possible_answers = []
    for i, feat in enumerate(st.session_state.features):
        col1, col2 = st.columns([1,1])
        with col1:
            feat['question'] = st.text_input(
                f"Feature Question {i+1}",
                value=feat['question'],
                key=f"feat_q_{i}"
            )
        with col2:
            feat['answers'] = st.text_input(
                f"Possible Answers {i+1} (comma separated)",
                value=feat['answers'],
                key=f"feat_a_{i}"
            )
        
        if feat['question'] and feat['answers']:
            features.append(feat['question'])
            possible_answers.append([x.strip() for x in feat['answers'].split(',')])
    
    # Step 4: Batch Processing
    st.subheader("4) Batch processing")
    
    if st.session_state.full_df is not None:
        df = st.session_state.full_df
        total_papers = len(df)
        
        batch_size = st.number_input(
            "Batch size (# of papers to analyze)",
            min_value=1,
            max_value=total_papers,
            value=min(10, total_papers)
        )
        
        start_index = st.number_input(
            "Start from paper number:",
            min_value=1,
            max_value=total_papers-batch_size+1,
            value=1
        )
        
        # Batch analysis
        run_in_parallel = st.checkbox("Run in parallel?", value=False)
        
        if st.button("Analyze batch"):
            if not llm_api_key:
                st.warning("Please provide an LLM API Key.")
            elif not features:
                st.warning("Please define features and possible answers.")
            else:
                with st.spinner('Analyzing batch...'):
                    df_batch = df.iloc[start_index-1:start_index-1+batch_size].copy()
                    analyzed_batch = handle_batch_analysis(
                        df_batch,
                        features,
                        possible_answers,
                        model_name,
                        llm_api_key,
                        run_in_parallel
                    )
                    st.session_state.unfiltered_df = analyzed_batch
                    st.success("Batch analysis completed!")
                    
                    # Provide CSV download
                    csv_data = analyzed_batch.to_csv(index=False)
                    st.download_button(
                        "Download unfiltered CSV",
                        data=csv_data,
                        file_name=f"literature_review_{start_index}_to_{start_index+batch_size-1}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Please complete the PubMed search first to enable batch processing.")
    
    # Step 5: Filtering
    st.subheader("5) Filter results")
    
    if st.session_state.unfiltered_df is not None:
        # Get sanitized column names
        sanitized_names = [sanitize_column_name(q) for q in features]
        
        # Create filter controls
        filter_dict = {}
        for col_name, possible_ans in zip(sanitized_names, possible_answers):
            selected_answer = st.selectbox(
                f"Filter by {col_name}:",
                [""] + possible_ans,
                key=f"filter_{col_name}"
            )
            if selected_answer:
                filter_dict[col_name] = selected_answer
        
        # Apply filters and display results
        filtered_df = filter_dataframe(st.session_state.unfiltered_df, filter_dict)
        
        if not filtered_df.empty:
            st.write(filtered_df)
            
            # Provide filtered CSV download
            filtered_csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "Download filtered CSV",
                data=filtered_csv_data,
                file_name="filtered_literature_review.csv",
                mime="text/csv"
            )
            
            # Store filtered results in session state
            st.session_state.filtered_df = filtered_df
            
            # Update progress step
            if st.session_state.current_step < 5:
                st.session_state.current_step = 5
        else:
            st.warning("No results match the selected filters.")
    else:
        st.info("Please complete batch processing first to enable filtering.")
    
    # Step 6: Research Question Analysis
    st.subheader("6) Analyze research question")
    
    if hasattr(st.session_state, 'filtered_df') and not st.session_state.filtered_df.empty:
        if st.button("Analyze filtered results"):
            if not research_question:
                st.warning("Please provide a research question for analysis.")
            elif not llm_api_key:
                st.warning("Please provide an LLM API key.")
            else:
                with st.spinner('Analyzing filtered papers...'):
                    analysis, was_truncated = analyze_filtered_results(
                        research_question,
                        st.session_state.filtered_df,
                        model_name,
                        llm_api_key,
                        st.session_state.get('hypothesis', None)  # Pass the hypothesis here
                    )
                    st.session_state.analysis_result = analysis
                    st.session_state.analysis_truncated = was_truncated
                    
                    # Update progress step to the final step
                    st.session_state.current_step = 6
        
        # Display analysis results if they exist
        if st.session_state.analysis_result:
            st.markdown(st.session_state.analysis_result)
            
            if st.session_state.analysis_truncated:
                st.warning(
                    "Note: The analysis was truncated to stay within "
                    "the token limit. Some information might "
                    "have been omitted."
                )
            
            # Export options
            st.subheader("Export Options")
            
            # Markdown export
            st.download_button(
                "Download Analysis (Markdown)",
                data=st.session_state.analysis_result,
                file_name="research_analysis.md",
                mime="text/markdown"
            )
    else:
        st.info("Please complete filtering first to enable analysis.")

if __name__ == "__main__":
    main()