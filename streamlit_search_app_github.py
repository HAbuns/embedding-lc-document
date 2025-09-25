"""
Streamlit App for Document Search - GITHUB-ONLY VERSION
Uses custom embedder instead of sentence-transformers
Perfect for environments that can only pull code from     def __init__(self):
        self.embeddings_dir = "/app/embeddings" if os.path.exists("/app/embeddings") else "./embeddings"
        self.local_model_path = "/app/local_models/sentence-transformer" if os.path.exists("/app/local_models") else "./local_models/sentence-transformer"
        self.model = None
        self.documents = {}
        self.is_initialized = False
        self.similarity_calculator = MultiSimilarityCalculator()
"""

import streamlit as st
import numpy as np
import json
import time
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
import pandas as pd

# Add current directory to path for custom imports
sys.path.append('.')

# Import custom embedder and similarity methods
from custom_embedder import CustomSentenceEmbedder  # Custom embedder
from similarity_methods import MultiSimilarityCalculator

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="VPBank Document Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    /* Hide Streamlit header and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .search-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem 0;
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-title {
        color: #1f77b4;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    
    .result-preview {
        color: #333;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .similarity-score {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .response-time {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin: 1rem 0;
        font-style: italic;
    }
    
    .github-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .method-comparison {
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .method-header {
        background: #f8f9fa;
        padding: 1rem;
        border-bottom: 1px solid #e0e0e0;
        font-weight: 600;
        color: #333;
    }
    
    .method-results {
        padding: 1rem;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .comparison-table th, .comparison-table td {
        padding: 0.8rem;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .comparison-table th {
        background: #f8f9fa;
        font-weight: 600;
        color: #333;
    }
    
    .method-score {
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }
    
    .score-high { background: #28a745; }
    .score-medium { background: #ffc107; color: #333; }
    .score-low { background: #dc3545; }
</style>
""", unsafe_allow_html=True)

class GitHubOnlyVectorSearchEngine:
    """Search engine using custom embedder (no sentence-transformers)"""
    
    def __init__(self):
        self.embeddings_dir = "/app/embeddings" if os.path.exists("/app/embeddings") else "./embeddings"
        self.local_model_path = "/app/local_models/sentence-transformer" if os.path.exists("/app/local_models") else "./local_models/sentence-transformer"
        self.model = None
        self.documents = {}
        self.is_initialized = False
        self.similarity_calculator = MultiSimilarityCalculator()
        
    def _load_model_and_data(self):
        """Load custom model and document data"""
        if self.is_initialized:
            return self.model, self.documents
            
        try:
            # Load custom model
            if os.path.exists(self.local_model_path):
                st.info("🚀 Loading custom embedder from local path...")
                model = CustomSentenceEmbedder(self.local_model_path)
            else:
                st.info("🚀 Loading custom embedder from Hugging Face...")
                model = CustomSentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            
            # Load document data
            documents = {}
            doc_names = ["isbp-745", "UCP600-1"]
            
            for doc_name in doc_names:
                try:
                    embeddings_path = f"{self.embeddings_dir}/{doc_name}_embeddings.npy"
                    chunks_path = f"{self.embeddings_dir}/{doc_name}_chunks.json"
                    metadata_path = f"{self.embeddings_dir}/{doc_name}_metadata.json"
                    
                    if not all(os.path.exists(path) for path in [embeddings_path, chunks_path, metadata_path]):
                        st.warning(f"⚠️ Some files missing for {doc_name}")
                        continue
                    
                    # Load embeddings
                    embeddings = np.load(embeddings_path)
                    
                    # Load chunks
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    documents[doc_name] = {
                        'embeddings': embeddings,
                        'chunks': chunks,
                        'metadata': metadata
                    }
                    
                except Exception as e:
                    st.error(f"Error loading {doc_name}: {str(e)}")
            
            if documents:
                self.model = model
                self.documents = documents
                self.is_initialized = True
                st.success("✅ Custom embedder and data loaded successfully!")
            
            return model, documents
            
        except Exception as e:
            st.error(f"Error initializing GitHub-only search engine: {str(e)}")
            return None, {}
    
    def search(self, query, top_k=5):
        """Search using custom embedder"""
        start_time = time.time()
        
        if not self.is_initialized:
            st.info("🔧 Initializing search engine for first time...")
            model, documents = self._load_model_and_data()
            if not model or not documents:
                return [], 0.0
        
        try:
            # Embed query using custom embedder
            st.info(f"🧮 Encoding query: '{query[:50]}...'")
            query_embedding = self.model.encode([query])
            
            results = []
            
            # Search through all documents
            st.info("🔍 Searching through documents...")
            for doc_name, doc_data in self.documents.items():
                st.info(f"📄 Searching in {doc_name}...")
                embeddings = doc_data['embeddings']
                chunks = doc_data['chunks']
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                
                # Get results from this document
                for idx, similarity in enumerate(similarities):
                    results.append({
                        'document': doc_name,
                        'chunk_index': idx,
                        'similarity': float(similarity),
                        'content': chunks[idx],
                        'preview': chunks[idx][:300] + "..." if len(chunks[idx]) > 300 else chunks[idx]
                    })
            
            # Sort by similarity and get top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = results[:top_k]
            
            search_time = time.time() - start_time
            
            return top_results, search_time
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return [], 0.0
    
    def search_with_multiple_methods(self, query, top_k=5):
        """Search using multiple similarity methods for comparison"""
        start_time = time.time()
        
        if not self.is_initialized:
            st.info("🔧 Initializing search engine for comparison...")
            model, documents = self._load_model_and_data()
            if not model or not documents:
                return {}, 0.0
        
        try:
            # Embed query using custom embedder
            st.info(f"🧮 Encoding query for multi-method comparison: '{query[:50]}...'")
            query_embedding = self.model.encode([query])[0]  # Get single vector
            
            all_results = {}
            
            # Search through all documents
            for doc_name, doc_data in self.documents.items():
                st.info(f"📄 Multi-method search in {doc_name}...")
                embeddings = doc_data['embeddings']
                chunks = doc_data['chunks']
                
                # Use multi-similarity calculator
                comparison_results = self.similarity_calculator.compare_methods(
                    query_embedding, embeddings, chunks, top_k
                )
                
                # Add document info to results
                comparison_results['document'] = doc_name
                all_results[doc_name] = comparison_results
            
            search_time = time.time() - start_time
            return all_results, search_time
            
        except Exception as e:
            st.error(f"Multi-method search error: {str(e)}")
            return {}, 0.0
    
    def get_stats(self):
        """Get database statistics"""
        if not self.is_initialized:
            self._load_model_and_data()
        
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': sum(len(doc['chunks']) for doc in self.documents.values()) if self.documents else 0,
            'embedding_dimension': 384,
            'model_name': 'Custom Embedder (GitHub-only)',
            'github_only': True
        }
        
        for doc_name, doc_data in self.documents.items():
            stats[f'{doc_name}_chunks'] = len(doc_data['chunks'])
        
        return stats

def main():
    """Main Streamlit app for GitHub-only deployment"""
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = GitHubOnlyVectorSearchEngine()
    
    search_engine = st.session_state.search_engine
    
    # Header with GitHub badge
    st.markdown('<h1 class="main-header">📚 VPBank Document Search Engine</h1>', unsafe_allow_html=True)
    st.markdown('<div class="github-badge">📦 GITHUB-ONLY MODE - No sentence-transformers</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Custom AI-powered semantic search using transformers + torch only</p>', unsafe_allow_html=True)
    
    # Search container
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "",
        placeholder="Enter your search query... (e.g., 'letter of credit requirements', 'payment terms')",
        key="search_query",
        help="Search through banking documents using custom AI embedder"
    )
    
    # Search options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        comparison_mode = st.checkbox("🔬 Multi-Method Comparison", 
                                    help="Compare results from different similarity methods")
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Perform search
    if search_clicked and query.strip():
        # Show search info
        search_info = st.empty()
        search_info.info(f"🔍 Starting {'multi-method comparison' if comparison_mode else 'standard'} search for: **{query}**")
        
        with st.spinner('🤖 Processing with custom embedder... Please wait...'):
            # Add some feedback during search
            progress = st.progress(0)
            progress.progress(25, text="Loading embedder...")
            
            if comparison_mode:
                results, search_time = search_engine.search_with_multiple_methods(query, top_k=5)
                progress.progress(100, text="Multi-method comparison completed!")
            else:
                results, search_time = search_engine.search(query, top_k=5)
                progress.progress(100, text="Search completed!")
        
        # Clear progress bar
        progress.empty()
        search_info.empty()
        
        if comparison_mode and results:
            # Display multi-method comparison results
            st.markdown(f'<div class="response-time">⚡ Multi-method comparison completed in {search_time:.3f} seconds</div>', 
                       unsafe_allow_html=True)
            
            # Create tabs for each document
            doc_tabs = st.tabs([f"📄 {doc_name.replace('-', ' ').title()}" for doc_name in results.keys()])
            
            for idx, (doc_name, doc_results) in enumerate(results.items()):
                with doc_tabs[idx]:
                    st.markdown(f"## 🔬 Method Comparison - {doc_name.replace('-', ' ').title()}")
                    
                    # Show timing information
                    timing_info = doc_results.get('timing', {})
                    if timing_info:
                        st.markdown("### ⏱️ Performance by Method:")
                        timing_cols = st.columns(len(timing_info))
                        for i, (method, time_val) in enumerate(timing_info.items()):
                            timing_cols[i].metric(method.title(), f"{time_val:.4f}s")
                    
                    # Show descriptions
                    descriptions = doc_results.get('descriptions', {})
                    if descriptions:
                        with st.expander("ℹ️ Method Descriptions"):
                            for method, desc in descriptions.items():
                                st.write(f"**{method.title()}**: {desc}")
                    
                    # Display comparison results
                    methods = doc_results.get('methods', {})
                    if methods:
                        # Create tabs for each method
                        method_tabs = st.tabs([f"{method.title()}" for method in methods.keys()])
                        
                        for method_idx, (method_name, method_results) in enumerate(methods.items()):
                            with method_tabs[method_idx]:
                                st.markdown(f"#### {method_name.title()} Results")
                                
                                for result in method_results[:3]:  # Show top 3
                                    score_class = "score-high" if result['score'] > 0.7 else "score-medium" if result['score'] > 0.4 else "score-low"
                                    
                                    result_html = f'''
                                    <div class="result-card">
                                        <div class="result-title">#{result['rank']} - {doc_name.replace('-', ' ').title()}</div>
                                        <div class="result-meta">
                                            📍 Index: {result['index']} | 
                                            <span class="method-score {score_class}">Score: {result['score']:.4f}</span>
                                        </div>
                                        <div class="result-preview">{result['preview']}</div>
                                    </div>
                                    '''
                                    st.markdown(result_html, unsafe_allow_html=True)
                        
                        # Summary comparison table
                        st.markdown("### 📊 Summary Comparison")
                        summary_data = []
                        for method_name, method_results in methods.items():
                            if method_results:
                                avg_score = sum(r['score'] for r in method_results) / len(method_results)
                                max_score = max(r['score'] for r in method_results)
                                summary_data.append({
                                    'Method': method_name.title(),
                                    'Avg Score': f"{avg_score:.4f}",
                                    'Max Score': f"{max_score:.4f}",
                                    'Results': len(method_results)
                                })
                        
                        if summary_data:
                            df = pd.DataFrame(summary_data)
                            st.dataframe(df, use_container_width=True)
        
        elif not comparison_mode and results:
            # Display response time
            st.markdown(f'<div class="response-time">⚡ Found {len(results)} results in {search_time:.3f} seconds (Custom Embedder)</div>', 
                       unsafe_allow_html=True)
            
            # Display results
            st.markdown("## 🎯 Search Results")
            
            for i, result in enumerate(results, 1):
                doc_display_name = result['document'].replace('-', ' ').title()
                
                result_html = f'''
                <div class="result-card">
                    <div class="result-title">#{i} - {doc_display_name}</div>
                    <div class="result-meta">
                        📄 Document: {result['document']} | 📍 Chunk: {result['chunk_index']} | 
                        <span class="similarity-score">Match: {result['similarity']:.1%}</span>
                    </div>
                    <div class="result-preview">{result['preview']}</div>
                </div>
                '''
                st.markdown(result_html, unsafe_allow_html=True)
                
                # Add expander for full content
                with st.expander(f"📖 View full content - {doc_display_name} (Chunk #{result['chunk_index']})"):
                    st.text_area(
                        "Full Content:",
                        result['content'],
                        height=200,
                        key=f"content_{i}",
                        disabled=True
                    )
        
        else:
            st.warning("⚠️ No results found. Try a different search query.")
    
    elif search_clicked and not query.strip():
        st.warning("⚠️ Please enter a search query.")
    
    # Sidebar with statistics
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        stats = search_engine.get_stats()
        
        # GitHub-only status
        if stats.get('github_only'):
            st.markdown('📦 **GITHUB-ONLY MODE**')
            st.success("✅ Using custom embedder")
        
        st.markdown("## 📈 Database Stats")
        st.markdown(f"""
        - **Total Documents**: {stats['total_documents']}
        - **Total Chunks**: {stats['total_chunks']}
        - **Model**: {stats['model_name']}
        - **Vector Dimension**: {stats['embedding_dimension']}
        """)
        
        if 'isbp-745_chunks' in stats:
            st.markdown(f"- **ISBP-745 Chunks**: {stats['isbp-745_chunks']}")
        if 'UCP600-1_chunks' in stats:
            st.markdown(f"- **UCP600-1 Chunks**: {stats['UCP600-1_chunks']}")
        
        st.markdown("---")
        st.markdown("## 💡 Search Tips")
        st.markdown("""
        - Use specific banking terms
        - Ask natural language questions
        - Try: "letter of credit", "compliance", "payment terms"
        - Results ranked by semantic similarity
        """)
        
        st.markdown("---")
        st.markdown("## ℹ️ Technical Info")
        st.markdown("""
        **📦 GitHub-Only Setup:**
        - Custom embedder implementation
        - No sentence-transformers dependency
        - Uses transformers + torch only
        - Drop-in replacement functionality
        
        **📚 Documents:**
        - ISBP-745 (Banking Practice)
        - UCP600-1 (Uniform Customs)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">📦 GitHub-Only Deployment | 🔧 Custom Embedder | Built for VPBank</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()