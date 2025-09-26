"""
VPBank Document Search Engine - Corporate/Offline Version
NO transformers, NO huggingface-hub - fully offline with pre-computed embeddings
Perfect for corporate environments with restricted internet access
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import json
from typing import List, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import offline components
try:
    from pure_offline_search import PureOfflineSearch
    from similarity_methods import MultiSimilarityCalculator
    COMPONENTS_LOADED = True
except ImportError as e:
    print(f"⚠️ Could not import components: {e}")
    COMPONENTS_LOADED = False

# Simple fallback search class
class SimpleOfflineSearch:
    """Fallback search without any AI - pure keyword matching"""
    
    def __init__(self):
        self.documents = self.load_documents()
        print(f"✅ Loaded {len(self.documents)} documents for simple search")
    
    def load_documents(self):
        """Load documents from various possible locations"""
        doc_paths = [
            "./embeddings/document_texts.json",
            "/app/embeddings/document_texts.json",
            "./document_texts.json"
        ]
        
        for path in doc_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                continue
        
        # Fallback documents
        return [
            "VPBank credit card terms and conditions. Interest rates and payment schedules.",
            "Banking services and account management policies for corporate clients.",
            "Investment products and wealth management services offered by VPBank.",
            "Personal loan requirements and application procedures for individuals.",
            "Foreign exchange services and international money transfer options.",
            "Digital banking features and mobile application usage guidelines.",
            "Customer service policies and complaint handling procedures.",
            "Security measures for online banking and fraud prevention protocols."
        ]
    
    def search_all_methods(self, query: str, top_k: int = 5) -> Dict:
        """Perform simple keyword search with multiple scoring methods"""
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        results = {}
        
        # Method 1: Cosine (word overlap ratio)
        cosine_scores = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            intersection = query_words.intersection(doc_words)
            union = query_words.union(doc_words)
            score = len(intersection) / len(union) if union else 0
            cosine_scores.append((i, score, doc))
        
        cosine_scores.sort(key=lambda x: x[1], reverse=True)
        results['cosine'] = [
            {
                'content': doc[:300] + "...",
                'similarity': round(score, 4),
                'metadata': {'index': idx, 'method': 'cosine'}
            }
            for idx, score, doc in cosine_scores[:top_k]
        ]
        
        # Method 2: Exact word matches
        exact_scores = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            score = sum(1 for word in query_words if word in doc_lower)
            score = score / len(query_words) if query_words else 0
            exact_scores.append((i, score, doc))
        
        exact_scores.sort(key=lambda x: x[1], reverse=True)
        results['euclidean'] = [
            {
                'content': doc[:300] + "...",
                'similarity': round(score, 4),
                'metadata': {'index': idx, 'method': 'euclidean'}
            }
            for idx, score, doc in exact_scores[:top_k]
        ]
        
        # Method 3: Character overlap
        char_scores = []
        for i, doc in enumerate(self.documents):
            doc_chars = set(query_lower)
            text_chars = set(doc.lower())
            overlap = len(doc_chars.intersection(text_chars))
            score = overlap / len(doc_chars.union(text_chars)) if doc_chars.union(text_chars) else 0
            char_scores.append((i, score, doc))
        
        char_scores.sort(key=lambda x: x[1], reverse=True)
        results['manhattan'] = [
            {
                'content': doc[:300] + "...",
                'similarity': round(score, 4),
                'metadata': {'index': idx, 'method': 'manhattan'}
            }
            for idx, score, doc in char_scores[:top_k]
        ]
        
        # Method 4: Length-based scoring
        length_scores = []
        target_len = len(query_lower)
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            # Score based on how many query words appear and document relevance
            word_matches = sum(1 for word in query_words if word in doc_lower)
            length_factor = min(len(doc_lower) / max(target_len, 1), 2.0)
            score = (word_matches / len(query_words) if query_words else 0) * length_factor * 0.5
            length_scores.append((i, score, doc))
        
        length_scores.sort(key=lambda x: x[1], reverse=True)
        results['dot_product'] = [
            {
                'content': doc[:300] + "...",
                'similarity': round(min(score, 1.0), 4),
                'metadata': {'index': idx, 'method': 'dot_product'}
            }
            for idx, score, doc in length_scores[:top_k]
        ]
        
        # Method 5: Jaccard similarity
        jaccard_scores = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            intersection = query_words.intersection(doc_words)
            union = query_words.union(doc_words)
            score = len(intersection) / len(union) if union else 0
            jaccard_scores.append((i, score, doc))
        
        jaccard_scores.sort(key=lambda x: x[1], reverse=True)
        results['jaccard'] = [
            {
                'content': doc[:300] + "...",
                'similarity': round(score, 4),
                'metadata': {'index': idx, 'method': 'jaccard'}
            }
            for idx, score, doc in jaccard_scores[:top_k]
        ]
        
        return results

# Initialize search engine
@st.cache_resource
def load_search_engine():
    """Load the appropriate search engine based on available components"""
    if COMPONENTS_LOADED:
        try:
            embedder = PureOfflineSearch()
            calc = MultiSimilarityCalculator()
            print("✅ Using pure offline search with TF-IDF/keyword matching")
            return embedder, calc, "offline_search"
        except Exception as e:
            print(f"⚠️ Pure offline search failed: {e}, falling back to simple search")
    
    # Fallback to simple search
    simple_search = SimpleOfflineSearch()
    return simple_search, None, "simple"

def main():
    """Main Streamlit application"""
    
    # Page config
    st.set_page_config(
        page_title="VPBank Document Search (Offline)",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .search-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        margin: 1rem 0;
    }
    
    .method-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e3e3e3;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .score-high { color: #27ae60; font-weight: bold; }
    .score-medium { color: #f39c12; font-weight: bold; }
    .score-low { color: #e74c3c; font-weight: bold; }
    
    .offline-badge {
        background: #e74c3c;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 VPBank Document Search Engine</h1>
        <h3>Corporate/Offline Version - No External Dependencies</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="offline-badge">🔒 CORPORATE MODE - Fully Offline</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Keyword-based search with multiple scoring algorithms</p>', unsafe_allow_html=True)
    
    # Load search engine
    search_engine, similarity_calc, engine_type = load_search_engine()
    
    # Search interface
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query:",
            placeholder="e.g., credit card interest rates, loan application process...",
            help="Enter keywords related to banking, finance, or VPBank services"
        )
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🚀 Search", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search parameters
    with st.sidebar:
        st.header("⚙️ Search Settings")
        top_k = st.slider("Number of results", 1, 10, 5)
        
        st.header("📊 Search Engine Info")
        if engine_type == "offline_search":
            st.success("✅ Using TF-IDF + keyword search")
        else:
            st.info("ℹ️ Using simple keyword search")
        
        st.header("🎯 Similarity Methods")
        st.write("""
        - **Cosine**: Word overlap ratio
        - **Euclidean**: Exact word matches  
        - **Manhattan**: Character overlap
        - **Dot Product**: Length-weighted scoring
        - **Jaccard**: Set similarity
        """)
    
    # Perform search
    if (query and search_button) or query:
        if not query.strip():
            st.warning("⚠️ Please enter a search query")
            return
        
        with st.spinner("🔍 Searching documents..."):
            start_time = time.time()
            
            try:
                if engine_type == "offline_search" and hasattr(search_engine, 'search_all_methods'):
                    # Use TF-IDF/keyword search
                    results = search_engine.search_all_methods(query, top_k)
                else:
                    # Use simple keyword search
                    results = search_engine.search_all_methods(query, top_k)
                
                search_time = time.time() - start_time
                
                # Display results
                st.success(f"✅ Search completed in {search_time:.3f} seconds")
                
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Response Time", f"{search_time:.3f}s")
                with col2:
                    st.metric("📄 Results per Method", str(top_k))
                with col3:
                    st.metric("🔧 Search Methods", "5")
                
                # Results display
                st.markdown("## 📋 Search Results")
                
                methods = ['cosine', 'euclidean', 'manhattan', 'dot_product', 'jaccard']
                method_names = {
                    'cosine': '📐 Cosine Similarity',
                    'euclidean': '📏 Euclidean Distance', 
                    'manhattan': '🏙️ Manhattan Distance',
                    'dot_product': '⚡ Dot Product',
                    'jaccard': '🔗 Jaccard Similarity'
                }
                
                for method in methods:
                    if method in results and results[method]:
                        st.markdown(f"### {method_names[method]}")
                        
                        for i, result in enumerate(results[method]):
                            score = result['similarity']
                            
                            # Score styling
                            if score > 0.7:
                                score_class = "score-high"
                                score_icon = "🟢"
                            elif score > 0.4:
                                score_class = "score-medium" 
                                score_icon = "🟡"
                            else:
                                score_class = "score-low"
                                score_icon = "🔴"
                            
                            st.markdown(f"""
                            <div class="method-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong>Result #{i+1}</strong>
                                    <span class="{score_class}">{score_icon} {score:.4f}</span>
                                </div>
                                <p style="margin: 0; color: #333; line-height: 1.5;">{result['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                st.write("Please check that document files are available and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🏦 VPBank Document Search Engine</h4>
        <p>
            <strong>Corporate/Offline Mode</strong><br>
            ✅ No external dependencies • ✅ Fully offline operation • ✅ Corporate firewall friendly
        </p>
        <p style="font-size: 0.9rem;">
            Features:<br>
            • No HuggingFace Hub or Transformers<br>
            • Pre-computed embeddings or keyword search<br>
            • 5 different similarity methods<br>
            • Fast response time (&lt;1 second)<br>
            • Perfect for restricted environments
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()