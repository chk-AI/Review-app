# Literature Review Assistant

# Overview
The Literature Review Assistant is a Streamlit application designed to assist users in conducting systematic and scoping literature reviews. It semi-automates several time-consuming aspects of the review process, from searching PubMed to analyzing and synthesizing findings.

# Access the Application
The Literature Review Assistant is available online at: 
https://review-assistant.streamlit.app/

# Features
Dual Review Types: Support for both systematic reviews (with PICO framework) and scoping reviews
PubMed Integration: Direct search of medical literature with advanced filtering options
LLM-Powered Analysis: Automated feature extraction and paper analysis using AI
Interactive Filtering: Dynamic filtering of results based on custom criteria
Data Visualization: Publication timelines, word clouds, and geographical distribution of research
Progress Tracking: Step-by-step workflow with visual progress indicators

# Prerequisites
OpenAI API key (for LLM integration)

Email address for PubMed queries

# Usage Guide
1. Select Review Type
Choose between a systematic review (focused on a specific clinical question using PICO criteria) or a scoping review (broader evidence mapping on a topic).
2. Define Research Question
Enter your research question and select an appropriate LLM model for analysis.
3. Search PubMed
Use the advanced search options to query PubMed. The application can suggest search strategies based on your research question.
4. Define Extraction Features
Specify what information should be extracted from each paper. For systematic reviews, PICO elements are automatically suggested.
5. Process Papers
Analyze papers in batches. The application will extract your defined features from titles and abstracts.
6. Filter Results
Apply filters to refine your dataset based on the extracted features.
7. Analyze Findings
Generate a comprehensive analysis of your filtered papers concerning your research question.

# Limitations
Note, that while the LLMs here have been strictly prompted to only comment on the filtered results, hallucinations can occur. For this reason, always double-check the results with the referenced literature. Please also note that the app currently does not support full-text analyses, which is required for comprehensive systematic and scoping reviews.

# Deployment Notes
The application uses Streamlit's session state for temporary data storage

# License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

# Acknowledgments
Built with Streamlit, OpenAI, Claude and Biopython
Uses the Entrez API for PubMed integration
