import streamlit as st

def render_progress_bar(current_step):
    """
    Render a progress bar showing all steps in the review process.
    Always displays all steps, with the current step highlighted.
    
    Args:
        current_step (int): Current step number (1-6)
    """
    steps = [
        "Research Question",
        "PubMed Search",
        "Feature Definition",
        "Batch Processing",
        "Filtering",
        "Analysis"
    ]
    
    # Create a container for the progress bar
    progress_container = st.container()
    
    with progress_container:
        # Calculate progress based on current_step
        progress = (current_step - 1) / (len(steps) - 1)
        
        # Display progress bar
        st.progress(progress)
        
        # Create columns for step labels
        cols = st.columns(len(steps))
        
        # Display step labels with appropriate styling
        for i, (col, step) in enumerate(zip(cols, steps), 1):
            with col:
                if i < current_step:
                    st.markdown(f"✅ {step}")
                elif i == current_step:
                    st.markdown(f"**🔵 {step}**")
                else:
                    st.markdown(f"⚪ {step}")

def get_step_description(step_number, review_type="scoping"):
    """
    Get the description for each step based on review type.
    
    Args:
        step_number (int): Step number (1-6)
        review_type (str): Type of review - "scoping" or "systematic"
    
    Returns:
        str: Description of the step
    """
    if review_type == "scoping":
        descriptions = {
            1: "Define your research question for a scoping review",
            2: "Search PubMed using advanced filters",
            3: "Define features to extract from papers",
            4: "Process papers in batches",
            5: "Filter and refine results",
            6: "Analyze findings and generate report"
        }
    else:  # systematic review
        descriptions = {
            1: "Define your PICO-based research question",
            2: "Search PubMed using advanced filters",
            3: "Define PICO screening features",
            4: "Process papers in batches",
            5: "Filter papers based on PICO criteria",
            6: "Analyze findings and generate systematic review"
        }
    
    return descriptions.get(step_number, "")