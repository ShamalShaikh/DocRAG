"""
Streamlit Dashboard for RAG System.

This module provides a user interface for interacting with the RAG system,
including search functionality, visualization of results, and system management.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sentence_transformers import SentenceTransformer
import streamlit as st
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from plotly.graph_objects import Figure
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "http://127.0.0.1:8000/api"  # Base URL already includes /api
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using a standard embedding model
DIMENSION_REDUCTION_METHODS = {
    "PCA": "Principal Component Analysis",
    "t-SNE": "t-Distributed Stochastic Neighbor Embedding",
    "UMAP": "Uniform Manifold Approximation and Projection"
}

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
        if "embeddings_cache" not in st.session_state:
            st.session_state.embeddings_cache = None
        if "last_update" not in st.session_state:
            st.session_state.last_update = None
            
        # Check for API key
        if not os.getenv("GROQ_API_KEY"):
            st.error(
                "Groq API key not found. Please set the GROQ_API_KEY "
                "environment variable."
            )
            st.stop()
        
        # Initialize embedding model for visualization
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            st.error(
                "Failed to initialize embedding model. Please check your "
                "internet connection and try again."
            )
            st.stop()
        
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
        
        # Display system info
        with st.expander("System Information"):
            st.markdown(f"""
            - **Embedding Model**: {EMBEDDING_MODEL}
            - **LLM Model**: gemma2-9b-it (via Groq)
            - **API Status**: {'🟢 Online' if self._check_api_health() else '🔴 Offline'}
            """)
        
    def _check_api_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            return response.status_code == 200
        except:
            return False
        
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
        """Render the document embeddings visualization section."""
        st.header("Data Visualization")
        
        try:
            # Fetch embeddings data
            response = requests.get(f"{API_BASE_URL}/embeddings")  # API_BASE_URL already includes /api
            response.raise_for_status()
            data = response.json()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(data["embeddings"]))
            with col2:
                st.metric("Total Embeddings", len(data["embeddings"]))
            with col3:
                st.metric("Last Updated", data["last_updated"])
                
            st.subheader("Document Embeddings")
            st.info(
                "This visualization shows the relationships between documents in the vector space. "
                "Similar documents appear closer together in the plot."
            )
            
            # Visualization controls
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                reduction_method = st.selectbox(
                    "Dimension Reduction Method",
                    options=list(DIMENSION_REDUCTION_METHODS.keys()),
                    format_func=lambda x: DIMENSION_REDUCTION_METHODS[x]
                )
            with col2:
                use_3d = st.checkbox("3D Plot", value=False)
            with col3:
                perplexity = st.slider("Perplexity", 5, 50, 30) if reduction_method == "t-SNE" else None
            with col4:
                if st.button("Refresh Data"):
                    st.session_state.embeddings_cache = None
                    
            if data["embeddings"]:
                # Convert embeddings to numpy array
                embeddings_array = np.array(data["embeddings"])
                
                # Perform dimension reduction
                if reduction_method == "PCA":
                    reducer = PCA(n_components=3 if use_3d else 2)
                elif reduction_method == "t-SNE":
                    reducer = TSNE(
                        n_components=3 if use_3d else 2,
                        perplexity=perplexity,
                        random_state=42
                    )
                else:  # UMAP
                    reducer = umap.UMAP(
                        n_components=3 if use_3d else 2,
                        random_state=42
                    )
                
                reduced_embeddings = reducer.fit_transform(embeddings_array)
                
                # Create interactive plot
                if use_3d:
                    # Determine coloring based on metadata
                    colors = []
                    color_labels = []
                    for meta in data["metadata"]:
                        # Try to use category/tag if available
                        if "category" in meta:
                            colors.append(meta["category"])
                            color_labels.append(f"Category: {meta['category']}")
                        elif "tags" in meta and meta["tags"]:
                            colors.append(meta["tags"][0])  # Use first tag
                            color_labels.append(f"Tag: {meta['tags'][0]}")
                        elif "processed_at" in meta:
                            # Convert timestamp to age in days
                            try:
                                processed_time = datetime.fromisoformat(meta["processed_at"])
                                age_days = (datetime.now() - processed_time).days
                                colors.append(age_days)
                                color_labels.append(f"Age: {age_days} days")
                            except:
                                colors.append(0)
                                color_labels.append("Age: Unknown")
                        else:
                            colors.append(0)
                            color_labels.append("No category")

                    fig = go.Figure(data=[
                        go.Scatter3d(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            z=reduced_embeddings[:, 2],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title="Document Category" if isinstance(colors[0], str) else "Document Age (days)"
                                )
                            ),
                            text=[
                                f"Title: {m.get('title', 'Untitled')}<br>"
                                f"Preview: {m.get('content_preview', '')}<br>"
                                f"{label}"
                                for m, label in zip(data["metadata"], color_labels)
                            ],
                            hoverinfo='text'
                        )
                    ])
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="Component 1",
                            yaxis_title="Component 2",
                            zaxis_title="Component 3"
                        ),
                        title=f"Document Embeddings ({reduction_method})"
                    )
                else:
                    # Determine coloring based on metadata (same as 3D case)
                    colors = []
                    color_labels = []
                    for meta in data["metadata"]:
                        # Try to use category/tag if available
                        if "category" in meta:
                            colors.append(meta["category"])
                            color_labels.append(f"Category: {meta['category']}")
                        elif "tags" in meta and meta["tags"]:
                            colors.append(meta["tags"][0])  # Use first tag
                            color_labels.append(f"Tag: {meta['tags'][0]}")
                        elif "processed_at" in meta:
                            # Convert timestamp to age in days
                            try:
                                processed_time = datetime.fromisoformat(meta["processed_at"])
                                age_days = (datetime.now() - processed_time).days
                                colors.append(age_days)
                                color_labels.append(f"Age: {age_days} days")
                            except:
                                colors.append(0)
                                color_labels.append("Age: Unknown")
                        else:
                            colors.append(0)
                            color_labels.append("No category")

                    fig = go.Figure(data=[
                        go.Scatter(
                            x=reduced_embeddings[:, 0],
                            y=reduced_embeddings[:, 1],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=colors,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title="Document Category" if isinstance(colors[0], str) else "Document Age (days)"
                                )
                            ),
                            text=[
                                f"Title: {m.get('title', 'Untitled')}<br>"
                                f"Preview: {m.get('content_preview', '')}<br>"
                                f"{label}"
                                for m, label in zip(data["metadata"], color_labels)
                            ],
                            hoverinfo='text'
                        )
                    ])
                    fig.update_layout(
                        xaxis_title="Component 1",
                        yaxis_title="Component 2",
                        title=f"Document Embeddings ({reduction_method})"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster statistics
                if st.checkbox("Show Cluster Statistics"):
                    # Calculate document counts by category/tag
                    categories = {}
                    for meta in data["metadata"]:
                        for tag in meta.get("tags", []):
                            categories[tag] = categories.get(tag, 0) + 1
                    
                    # Create bar chart
                    if categories:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(categories.keys()),
                                y=list(categories.values())
                            )
                        ])
                        fig.update_layout(
                            title="Documents per Category",
                            xaxis_title="Category",
                            yaxis_title="Number of Documents"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No category information available in metadata")
            else:
                st.warning("No embeddings available for visualization")
            
        except Exception as e:
            st.error(f"Error loading visualization: {str(e)}")
            logger.error(f"Visualization error: {str(e)}", exc_info=True)
            
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
            
    def _fetch_embeddings(self) -> Tuple[np.ndarray, List[Dict], datetime]:
        """
        Fetch document embeddings from the vector database.
        
        Returns:
            Tuple containing:
            - numpy.ndarray: Matrix of embeddings
            - List[Dict]: Document metadata
            - datetime: Last update timestamp
        """
        try:
            # Get embeddings from API
            response = requests.get(f"{API_BASE_URL}/embeddings")
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings and metadata
            embeddings = np.array([doc["embedding"] for doc in data["documents"]])
            metadata = [
                {
                    "id": doc["id"],
                    "title": doc["metadata"].get("title", "Untitled"),
                    "url": doc["metadata"].get("url", ""),
                    "date": doc["metadata"].get("publication_date", ""),
                    "cluster": doc["metadata"].get("cluster", "Unknown")
                }
                for doc in data["documents"]
            ]
            
            last_update = datetime.fromisoformat(data["last_update"])
            return embeddings, metadata, last_update
            
        except Exception as e:
            logger.error(f"Error fetching embeddings: {str(e)}")
            raise
            
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "UMAP",
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Dimension reduction method ("PCA", "t-SNE", or "UMAP")
            n_components: Number of dimensions to reduce to (2 or 3)
            
        Returns:
            numpy.ndarray: Reduced-dimension embeddings
        """
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        elif method == "t-SNE":
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(embeddings) - 1)
            )
        else:  # UMAP
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                min_dist=0.1,
                n_neighbors=min(15, len(embeddings) - 1)
            )
            
        return reducer.fit_transform(embeddings)
        
    def _create_embedding_plot(
        self,
        reduced_embeddings: np.ndarray,
        metadata: List[Dict],
        plot_3d: bool = False
    ) -> Figure:
        """
        Create an interactive embedding visualization.
        
        Args:
            reduced_embeddings: Dimension-reduced embeddings
            metadata: Document metadata for each embedding
            plot_3d: Whether to create a 3D plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        # Create DataFrame for plotting
        df = pd.DataFrame(
            reduced_embeddings,
            columns=[f"Dimension {i+1}" for i in range(reduced_embeddings.shape[1])]
        )
        df["Title"] = [m["title"] for m in metadata]
        df["URL"] = [m["url"] for m in metadata]
        df["Date"] = [m["date"] for m in metadata]
        df["Cluster"] = [m["cluster"] for m in metadata]
        
        # Create hover text
        df["Hover Text"] = df.apply(
            lambda row: f"Title: {row['Title']}<br>"
                       f"Date: {row['Date']}<br>"
                       f"Cluster: {row['Cluster']}",
            axis=1
        )
        
        if plot_3d:
            fig = px.scatter_3d(
                df,
                x="Dimension 1",
                y="Dimension 2",
                z="Dimension 3",
                color="Cluster",
                hover_data=["Title", "Date"],
                title="Document Embedding Clusters (3D)",
                template="plotly_dark"
            )
        else:
            fig = px.scatter(
                df,
                x="Dimension 1",
                y="Dimension 2",
                color="Cluster",
                hover_data=["Title", "Date"],
                title="Document Embedding Clusters",
                template="plotly_dark"
            )
            
        # Update layout for better interactivity
        fig.update_traces(
            marker=dict(size=10),
            hovertemplate="%{customdata[0]}<br>"
                         "Date: %{customdata[1]}<br>"
                         "Cluster: %{marker.color}<br>"
                         "<extra></extra>"
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            legend_title_text="Clusters",
            hovermode="closest"
        )
        
        return fig
            
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