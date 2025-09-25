"""
Multiple Similarity Methods for Document Search
Implements various similarity calculation methods for comparison
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from typing import List, Dict, Tuple, Any
import time

class MultiSimilarityCalculator:
    """
    Calculator for multiple similarity methods
    Provides comparison between different similarity algorithms
    """
    
    def __init__(self):
        self.methods = {
            'cosine': self._cosine_similarity,
            'euclidean': self._euclidean_similarity, 
            'manhattan': self._manhattan_similarity,
            'dot_product': self._dot_product_similarity,
            'jaccard': self._jaccard_similarity
        }
        
    def _cosine_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Standard cosine similarity"""
        return cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
    
    def _euclidean_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Euclidean distance converted to similarity (1 / (1 + distance))"""
        distances = []
        for doc_emb in document_embeddings:
            # Manual euclidean distance calculation
            dist = np.sqrt(np.sum((query_embedding - doc_emb) ** 2))
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            distances.append(similarity)
        return np.array(distances)
    
    def _manhattan_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Manhattan distance converted to similarity"""
        distances = []
        for doc_emb in document_embeddings:
            # Manual manhattan distance calculation
            dist = np.sum(np.abs(query_embedding - doc_emb))
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            distances.append(similarity)
        return np.array(distances)
    
    def _dot_product_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Dot product similarity (normalized)"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Calculate dot product
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def _jaccard_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Jaccard similarity for binary vectors (threshold-based)"""
        # Convert to binary vectors (threshold = mean)
        query_binary = (query_embedding > np.mean(query_embedding)).astype(int)
        
        similarities = []
        for doc_emb in document_embeddings:
            doc_binary = (doc_emb > np.mean(doc_emb)).astype(int)
            
            # Manual Jaccard similarity calculation
            intersection = np.sum(query_binary & doc_binary)
            union = np.sum(query_binary | doc_binary)
            
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union
            
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def calculate_all_similarities(self, query_embedding: np.ndarray, 
                                 document_embeddings: np.ndarray, 
                                 top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate similarities using all methods
        
        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors
            top_k: Number of top results to return
            
        Returns:
            Dictionary with results from all methods
        """
        results = {}
        timing = {}
        
        for method_name, method_func in self.methods.items():
            start_time = time.time()
            
            try:
                similarities = method_func(query_embedding, document_embeddings)
                
                # Create results with indices and scores
                method_results = []
                for idx, score in enumerate(similarities):
                    method_results.append({
                        'index': idx,
                        'score': float(score),
                        'method': method_name
                    })
                
                # Sort by score and get top K
                method_results.sort(key=lambda x: x['score'], reverse=True)
                results[method_name] = method_results[:top_k]
                
                timing[method_name] = time.time() - start_time
                
            except Exception as e:
                print(f"Error calculating {method_name}: {str(e)}")
                results[method_name] = []
                timing[method_name] = 0.0
        
        results['timing'] = timing
        return results
    
    def get_method_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all similarity methods"""
        return {
            'cosine': 'Cosine Similarity - Measures angle between vectors (0-1)',
            'euclidean': 'Euclidean Distance - L2 distance converted to similarity',
            'manhattan': 'Manhattan Distance - L1 distance converted to similarity', 
            'dot_product': 'Dot Product - Normalized dot product similarity',
            'jaccard': 'Jaccard Similarity - Binary vector similarity (0-1)'
        }
    
    def compare_methods(self, query_embedding: np.ndarray, 
                       document_embeddings: np.ndarray,
                       chunks: List[str],
                       top_k: int = 5) -> Dict[str, Any]:
        """
        Compare all methods and return comprehensive results
        
        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors  
            chunks: Text chunks corresponding to embeddings
            top_k: Number of results per method
            
        Returns:
            Comprehensive comparison results
        """
        # Calculate similarities with all methods
        similarity_results = self.calculate_all_similarities(
            query_embedding, document_embeddings, top_k
        )
        
        # Prepare comparison data
        comparison = {
            'methods': {},
            'timing': similarity_results.pop('timing'),
            'descriptions': self.get_method_descriptions(),
            'summary': {}
        }
        
        # Process results for each method
        for method_name, method_results in similarity_results.items():
            method_data = []
            
            for result in method_results:
                idx = result['index']
                method_data.append({
                    'rank': len(method_data) + 1,
                    'index': idx,
                    'score': result['score'],
                    'text': chunks[idx] if idx < len(chunks) else "N/A",
                    'preview': chunks[idx][:200] + "..." if idx < len(chunks) and len(chunks[idx]) > 200 else chunks[idx] if idx < len(chunks) else "N/A"
                })
            
            comparison['methods'][method_name] = method_data
        
        # Calculate summary statistics
        all_scores = []
        for method_results in similarity_results.values():
            scores = [r['score'] for r in method_results]
            all_scores.extend(scores)
        
        if all_scores:
            comparison['summary'] = {
                'total_methods': len(self.methods),
                'avg_score': np.mean(all_scores),
                'max_score': np.max(all_scores),
                'min_score': np.min(all_scores),
                'std_score': np.std(all_scores)
            }
        
        return comparison

def test_similarity_methods():
    """Test function for similarity methods"""
    print("🧪 Testing Multiple Similarity Methods...")
    
    # Create test data
    np.random.seed(42)
    query_emb = np.random.randn(384)
    doc_embs = np.random.randn(10, 384)
    chunks = [f"Test document chunk {i+1} with some sample text content." for i in range(10)]
    
    # Initialize calculator
    calculator = MultiSimilarityCalculator()
    
    # Run comparison
    results = calculator.compare_methods(query_emb, doc_embs, chunks, top_k=3)
    
    print("\n📊 Method Descriptions:")
    for method, desc in results['descriptions'].items():
        print(f"  • {method}: {desc}")
    
    print(f"\n⏱️  Timing Results:")
    for method, time_taken in results['timing'].items():
        print(f"  • {method}: {time_taken:.4f}s")
    
    print(f"\n📈 Summary Statistics:")
    summary = results['summary']
    print(f"  • Average Score: {summary['avg_score']:.4f}")
    print(f"  • Max Score: {summary['max_score']:.4f}")
    print(f"  • Min Score: {summary['min_score']:.4f}")
    
    print(f"\n🏆 Top Results by Method:")
    for method, method_results in results['methods'].items():
        print(f"\n  {method.upper()}:")
        for result in method_results[:2]:  # Show top 2
            print(f"    #{result['rank']}: Score {result['score']:.4f} - {result['preview']}")
    
    return results

if __name__ == "__main__":
    test_results = test_similarity_methods()
    print("\n✅ Multi-similarity testing completed!")