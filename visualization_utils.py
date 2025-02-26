import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pycountry
import re
from collections import Counter

def create_geographical_visualization(df):
    """Create a choropleth map of study locations."""
    # Extract countries from affiliations (assuming they're in the abstract)
    countries = []
    for abstract in df['Abstract']:
        # Simple pattern matching for country names
        found_countries = []
        for country in pycountry.countries:
            if country.name in abstract:
                found_countries.append(country.name)
        countries.extend(found_countries)
    
    # Count country frequencies
    country_counts = pd.DataFrame(Counter(countries).items(), 
                                columns=['country', 'count'])
    
    # Create choropleth map
    fig = px.choropleth(country_counts,
                        locations='country',
                        locationmode='country names',
                        color='count',
                        hover_name='country',
                        color_continuous_scale='Viridis')
    
    return fig

def create_word_cloud(df):
    """Create a word cloud from abstracts."""
    # Combine all abstracts
    text = ' '.join(df['Abstract'].fillna(''))
    
    # Clean text
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    
    # Create and configure word cloud
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         max_words=100).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def create_timeline_visualization(df):
    """Create a timeline visualization of publications."""
    year_counts = df['Year'].value_counts().sort_index()
    fig = px.line(x=year_counts.index, y=year_counts.values,
                  labels={'x': 'Year', 'y': 'Number of Publications'},
                  title='Publication Timeline')
    return fig

def display_visualizations(df):
    """Display all visualizations in the Streamlit app."""
    st.subheader("Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Geographical Distribution",
        "Word Cloud",
        "Publication Timeline"
    ])
    
    with tab1:
        st.plotly_chart(create_geographical_visualization(df))
        
    with tab2:
        st.pyplot(create_word_cloud(df))
        
    with tab3:
        st.plotly_chart(create_timeline_visualization(df))