import streamlit as st
import pandas as pd
from datetime import datetime, date
import json
from pathlib import Path

class SearchManager:
    def __init__(self):
        self.saved_searches_file = Path("data/saved_searches.json")
        self.ensure_saved_searches_file()
    
    def ensure_saved_searches_file(self):
        """Create saved searches file if it doesn't exist."""
        self.saved_searches_file.parent.mkdir(exist_ok=True)
        if not self.saved_searches_file.exists():
            with open(self.saved_searches_file, 'w') as f:
                json.dump({}, f)
    
    def save_search(self, username, search_name, search_params):
        """Save a search configuration."""
        with open(self.saved_searches_file, 'r') as f:
            saved_searches = json.load(f)
        
        if username not in saved_searches:
            saved_searches[username] = {}
        
        saved_searches[username][search_name] = {
            'params': search_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.saved_searches_file, 'w') as f:
            json.dump(saved_searches, f, indent=2)
    
    def get_saved_searches(self, username):
        """Get all saved searches for a user."""
        with open(self.saved_searches_file, 'r') as f:
            saved_searches = json.load(f)
        return saved_searches.get(username, {})
    
    def delete_saved_search(self, username, search_name):
        """Delete a saved search."""
        with open(self.saved_searches_file, 'r') as f:
            saved_searches = json.load(f)
        
        if username in saved_searches and search_name in saved_searches[username]:
            del saved_searches[username][search_name]
            
            with open(self.saved_searches_file, 'w') as f:
                json.dump(saved_searches, f, indent=2)
            return True
        return False

def build_advanced_search_query(params):
    """Build PubMed query string from advanced search parameters."""
    query_parts = []
    
    # Add basic search terms
    if params.get('terms'):
        query_parts.append(f"({params['terms']})")
    
    # Add date range
    if params.get('date_from') and params.get('date_to'):
        # Convert dates to string format expected by PubMed
        date_from = params['date_from'].strftime("%Y/%m/%d")
        date_to = params['date_to'].strftime("%Y/%m/%d")
        query_parts.append(
            f"(\"{date_from}\"[Date - Publication] : "
            f"\"{date_to}\"[Date - Publication])"
        )
    
    # Add publication types
    pub_types = params.get('publication_types', [])
    if pub_types:
        pub_type_query = " OR ".join(f"\"{pt}\"[Publication Type]" for pt in pub_types)
        query_parts.append(f"({pub_type_query})")
    
    # Add languages
    languages = params.get('languages', [])
    if languages:
        lang_query = " OR ".join(f"\"{lang}\"[Language]" for lang in languages)
        query_parts.append(f"({lang_query})")
    
    # Combine all parts with AND
    return " AND ".join(query_parts)

def render_advanced_search():
    """Render the advanced search interface."""
    st.write("### Advanced Search Options")
    
    # Initialize search parameters
    if 'search_params' not in st.session_state:
        st.session_state.search_params = {
            'terms': '',
            'date_from': None,
            'date_to': None,
            'publication_types': [],
            'languages': []
        }
    
    # Search terms
    st.session_state.search_params['terms'] = st.text_area(
        "Search Terms",
        value=st.session_state.search_params['terms']
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.search_params['date_from'] = st.date_input(
            "Date From",
            value=st.session_state.search_params['date_from']
        )
    with col2:
        st.session_state.search_params['date_to'] = st.date_input(
            "Date To",
            value=st.session_state.search_params['date_to']
        )
    
    # Publication types
    publication_types = [
        "Journal Article",
        "Clinical Trial",
        "Meta-Analysis",
        "Randomized Controlled Trial",
        "Review",
        "Systematic Review"
    ]
    st.session_state.search_params['publication_types'] = st.multiselect(
        "Publication Types",
        options=publication_types,
        default=st.session_state.search_params['publication_types']
    )
    
    # Languages
    languages = ["English", "French", "German", "Spanish", "Italian"]
    st.session_state.search_params['languages'] = st.multiselect(
        "Languages",
        options=languages,
        default=st.session_state.search_params['languages']
    )
    
    # Always generate the query based on current parameters
    query = build_advanced_search_query(st.session_state.search_params)
    
    # Show the current query
    if query:
        st.write("### Current Query")
        st.code(query)
    
    return query  # Always return the current query

def init_search_state():
    """Initialize search-related session state variables."""
    if 'search_manager' not in st.session_state:
        st.session_state.search_manager = SearchManager()