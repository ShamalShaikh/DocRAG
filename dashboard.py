"""
Streamlit Dashboard for RAG System.

This module provides a user interface for interacting with the RAG system,
including search functionality, visualization of results, and system management.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "http://localhost:8000/api"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class RAGDashboard:
    """
    Main class for the RAG system dashboard.
    
    This class handles:
    1. UI layout and components
    2. API communication
    3. Data visualization
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        # Set page config
        st.set_page_config(
            page_title="RAG System Dashboard",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize session state
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
        
        # Initialize embedding model for visualization
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
    def render_header(self):
        """Render the dashboard header."""
        st.title("RAG System Dashboard")
        st.markdown("""
        This dashboard provides access to the RAG system's functionality:
        - Search and query the knowledge base
        - Manage web scraping targets
        - Visualize document embeddings
        - Add annotations to documents
        """)
        
    def render_search_section(self):
        """Render the search and query section."""
        st.header("Search & Query")
        
        # Search input
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are vector databases?"
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            filter_criteria = st.text_input(
                "Filter criteria (JSON):",
                placeholder='{"tags": ["databases"]}',
                help="Optional JSON filter criteria"
            )
        with col2:
            max_results = st.number_input(
                "Max results:",
                min_value=1,
                max_value=10,
                value=3
            )
            
        # Search button
        if st.button("Search", type="primary"):
            if query:
                self.process_query(
                    query,
                    filter_criteria,
                    max_results
                )
            else:
                st.warning("Please enter a query")
                
    def process_query(
        self,
        query: str,
        filter_criteria: str,
        max_results: int
    ):
        """
        Process a search query and display results.
        
        Args:
            query: User's query string
            filter_criteria: JSON string of filter criteria
            max_results: Maximum number of results to return
        """
        try:
            # Parse filter criteria
            filters = (
                json.loads(filter_criteria)
                if filter_criteria else None
            )
            
            # Make API request
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "query": query,
                    "filter_criteria": filters,
                    "max_results": max_results
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Display results
            st.subheader("Answer")
            st.write(data["answer"])
            
            st.subheader("Sources")
            for i, source in enumerate(data["sources"], 1):
                with st.expander(
                    f"Source {i} - Score: {source['similarity_score']:.3f}"
                ):
                    st.markdown(f"**Title:** {source['metadata'].get('title', 'Untitled')}")
                    st.markdown(f"**Content:**\n{source['content']}")
                    
                    # Annotation interface
                    if st.button(f"Add Annotation #{i}"):
                        annotation = st.text_area(
                            "Enter your annotation:",
                            key=f"annotation_{i}"
                        )
                        if st.button("Save Annotation", key=f"save_{i}"):
                            self.save_annotation(
                                document_id=source["metadata"].get("id", ""),
                                annotation=annotation,
                                highlight_text=""  # TODO: Implement text highlighting
                            )
                            
            st.caption(f"Total tokens used: {data['total_tokens']}")
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            
    def render_scraping_section(self):
        """Render the web scraping management section."""
        st.header("Web Scraping Management")
        
        # Add new target
        with st.expander("Add Scraping Target"):
            url = st.text_input(
                "URL:",
                placeholder="https://example.com"
            )
            schedule = st.text_input(
                "Schedule (cron expression):",
                placeholder="0 * * * *",
                help="Optional cron schedule"
            )
            metadata = st.text_area(
                "Metadata (JSON):",
                placeholder='{"category": "tech"}',
                help="Optional metadata"
            )
            
            if st.button("Add Target"):
                if url:
                    self.add_scraping_target(url, schedule, metadata)
                else:
                    st.warning("Please enter a URL")
                    
        # Display existing targets
        try:
            # TODO: Implement API endpoint for listing targets
            st.info("Scraping targets will be displayed here")
        except Exception as e:
            st.error(f"Error loading scraping targets: {str(e)}")
            
    def render_visualization_section(self):
        """Render the data visualization section."""
        st.header("Data Visualization")
        
        try:
            # Get system stats
            stats = requests.get(f"{API_BASE_URL}/stats").json()
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats["total_documents"])
            with col2:
                st.metric("Total Embeddings", stats["total_embeddings"])
            with col3:
                st.metric("Last Update", stats["last_update"])
                
            # Embedding visualization
            st.subheader("Document Embeddings")
            st.info(
                "This visualization shows document clusters based on their embeddings. "
                "Similar documents appear closer together."
            )
            
            # TODO: Implement actual embedding visualization
            # For now, show dummy plot
            df = pd.DataFrame({
                'x': np.random.randn(50),
                'y': np.random.randn(50),
                'cluster': np.random.choice(['A', 'B', 'C'], 50)
            })
            
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='cluster',
                title='Document Embedding Clusters'
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error loading visualizations: {str(e)}")
            
    def add_scraping_target(
        self,
        url: str,
        schedule: Optional[str],
        metadata: Optional[str]
    ):
        """
        Add a new scraping target.
        
        Args:
            url: Target URL
            schedule: Optional cron schedule
            metadata: Optional JSON metadata
        """
        try:
            # Parse metadata
            meta_dict = json.loads(metadata) if metadata else None
            
            # Make API request
            response = requests.post(
                f"{API_BASE_URL}/scrape",
                json={
                    "url": url,
                    "schedule": schedule,
                    "metadata": meta_dict
                }
            )
            response.raise_for_status()
            
            st.success(f"Added scraping target: {url}")
            
        except Exception as e:
            st.error(f"Error adding scraping target: {str(e)}")
            
    def save_annotation(
        self,
        document_id: str,
        annotation: str,
        highlight_text: str
    ):
        """
        Save a document annotation.
        
        Args:
            document_id: ID of the document
            annotation: Annotation text
            highlight_text: Text to highlight
        """
        try:
            # Make API request
            response = requests.post(
                f"{API_BASE_URL}/annotate",
                json={
                    "document_id": document_id,
                    "annotation": annotation,
                    "highlight_text": highlight_text
                }
            )
            response.raise_for_status()
            
            # Update session state
            if document_id not in st.session_state.annotations:
                st.session_state.annotations[document_id] = []
            st.session_state.annotations[document_id].append({
                "text": annotation,
                "timestamp": datetime.now().isoformat()
            })
            
            st.success("Annotation saved successfully")
            
        except Exception as e:
            st.error(f"Error saving annotation: {str(e)}")
            
    def render(self):
        """Render the complete dashboard."""
        self.render_header()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Search & Query",
            "Scraping Management",
            "Visualization"
        ])
        
        with tab1:
            self.render_search_section()
            
        with tab2:
            self.render_scraping_section()
            
        with tab3:
            self.render_visualization_section()

def main():
    """Main entry point for the dashboard."""
    dashboard = RAGDashboard()
    dashboard.render()

if __name__ == "__main__":
    main() 