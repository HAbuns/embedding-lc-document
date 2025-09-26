"""
Pure Offline Search - NO AI dependencies whatsoever
Perfect for corporate environments with maximum restrictions
Uses only Python standard library + numpy/pandas/scikit-learn
"""

import numpy as np
import json
import os
from typing import List, Dict, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class PureOfflineSearch:
    """
    Pure offline search using only standard libraries and scikit-learn
    NO torch, NO transformers, NO huggingface - maximum compatibility
    """
    
    def __init__(self, embeddings_path: str = "./embeddings"):
        """Initialize with document texts only"""
        self.embeddings_path = embeddings_path
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        
        # Load documents
        self._load_documents()
        self._setup_tfidf()
    
    def _load_documents(self):
        """Load document texts from various sources"""
        doc_files = [
            os.path.join(self.embeddings_path, "document_texts.json"),
            "./embeddings/document_texts.json",
            "/app/embeddings/document_texts.json",
            "./document_texts.json"
        ]
        
        for doc_file in doc_files:
            try:
                if os.path.exists(doc_file):
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        self.documents = json.load(f)
                    print(f"✅ Loaded {len(self.documents)} documents from {doc_file}")
                    return
            except Exception as e:
                continue
        
        # Fallback documents
        print("⚠️ Using fallback documents")
        self.documents = [
            "VPBank offers comprehensive credit card services with competitive interest rates. Our credit cards provide flexible payment options and reward programs for loyal customers.",
            "Personal loan services at VPBank include various loan packages for different customer needs. Interest rates are competitive and application processes are streamlined for quick approval.",
            "VPBank corporate banking services cater to businesses of all sizes. We provide cash management, trade finance, and corporate loan facilities.",
            "Investment and wealth management services help customers grow their financial portfolio. Our experienced advisors provide personalized investment strategies.",
            "Digital banking platform offers 24/7 access to banking services. Mobile app features include money transfer, bill payment, and account management.",
            "Foreign exchange services include currency conversion and international money transfer. Competitive rates for major currencies are available daily.",
            "Customer service excellence is our priority. Multiple support channels including phone, email, and in-branch assistance are available.",
            "Security measures protect customer data and transactions. Advanced encryption and fraud detection systems ensure safe banking experience."
        ]
    
    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer for document representation"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            print(f"✅ TF-IDF vectorizer setup complete - {self.doc_vectors.shape[1]} features")
        except Exception as e:
            print(f"⚠️ TF-IDF setup failed: {e}, using simple keyword search")
            self.vectorizer = None
            self.doc_vectors = None
    
    def similarity_search(self, query: str, top_k: int = 5, method: str = 'cosine') -> List[Dict]:
        """Search documents using TF-IDF or keyword matching"""
        if not query.strip():
            return []
        
        try:
            if self.vectorizer is not None and self.doc_vectors is not None:
                return self._tfidf_search(query, top_k, method)
            else:
                return self._keyword_search(query, top_k, method)
        except Exception as e:
            print(f"⚠️ Search failed: {e}, falling back to keyword search")
            return self._keyword_search(query, top_k, method)
    
    def _tfidf_search(self, query: str, top_k: int, method: str) -> List[Dict]:
        """Search using TF-IDF vectors"""
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        if method == 'cosine':
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        else:
            # For other methods, use cosine as base and modify
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            if method == 'euclidean':
                # Convert to euclidean-like score
                similarities = 1 / (1 + similarities)
            elif method == 'manhattan':
                # Manhattan-like transformation
                similarities = similarities * 0.8
            elif method == 'dot_product':
                # Use raw dot product
                similarities = query_vector.dot(self.doc_vectors.T).toarray().flatten()
                # Normalize to [0,1]
                similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
            elif method == 'jaccard':
                # Jaccard-like score using TF-IDF
                similarities = similarities * 0.9
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx][:400] + "..." if len(self.documents[idx]) > 400 else self.documents[idx],
                'similarity': float(similarities[idx]),
                'metadata': {'index': int(idx), 'method': method}
            })
        
        return results
    
    def _keyword_search(self, query: str, top_k: int, method: str) -> List[Dict]:
        """Fallback keyword-based search"""
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        scores = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Different scoring methods
            if method == 'cosine':
                # Jaccard similarity (word overlap)
                intersection = query_words.intersection(doc_words)
                union = query_words.union(doc_words)
                score = len(intersection) / len(union) if union else 0
                
            elif method == 'euclidean':
                # Exact word matches
                score = sum(1 for word in query_words if word in doc_lower)
                score = score / len(query_words) if query_words else 0
                
            elif method == 'manhattan':
                # Character-based similarity
                query_chars = set(query_lower)
                doc_chars = set(doc_lower)
                intersection = query_chars.intersection(doc_chars)
                union = query_chars.union(doc_chars)
                score = len(intersection) / len(union) if union else 0
                
            elif method == 'dot_product':
                # Length-weighted similarity
                word_matches = sum(1 for word in query_words if word in doc_lower)
                length_factor = min(len(doc_lower) / max(len(query_lower), 1), 2.0)
                score = (word_matches / len(query_words) if query_words else 0) * length_factor * 0.3
                
            elif method == 'jaccard':
                # Pure Jaccard similarity
                intersection = query_words.intersection(doc_words)
                union = query_words.union(doc_words)
                score = len(intersection) / len(union) if union else 0
            
            else:
                # Default to cosine
                intersection = query_words.intersection(doc_words)
                union = query_words.union(doc_words)
                score = len(intersection) / len(union) if union else 0
            
            scores.append((i, score))
        
        # Sort and get top results
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (idx, score) in enumerate(scores[:top_k]):
            results.append({
                'content': self.documents[idx][:400] + "..." if len(self.documents[idx]) > 400 else self.documents[idx],
                'similarity': round(min(score, 1.0), 4),
                'metadata': {'index': idx, 'method': method}
            })
        
        return results
    
    def search_all_methods(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Search using all similarity methods"""
        methods = ['cosine', 'euclidean', 'manhattan', 'dot_product', 'jaccard']
        results = {}
        
        for method in methods:
            results[method] = self.similarity_search(query, top_k, method)
        
        return results

# Compatibility aliases
OfflineEmbedder = PureOfflineSearch
CustomSentenceEmbedder = PureOfflineSearch