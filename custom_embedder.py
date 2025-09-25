"""
Custom Embedding Solution for GitHub-only Environment
This replaces sentence-transformers with direct transformers + torch usage
Works without sentence-transformers package
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from typing import List, Union
import os
import json

class CustomSentenceEmbedder:
    """
    Custom implementation to replace SentenceTransformer
    Uses only transformers and torch - no sentence-transformers dependency
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize custom embedder
        
        Args:
            model_name: Hugging Face model name or local path
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model"""
        try:
            print(f"Loading tokenizer and model: {self.model_name}")
            
            # Check if offline mode is enabled
            offline_mode = os.getenv('TRANSFORMERS_OFFLINE', '0') == '1' or os.getenv('HF_HUB_OFFLINE', '0') == '1'
            
            # Try local paths first
            local_paths = [
                "/app/local_models/sentence-transformer",
                "./local_models/sentence-transformer", 
                "./local_models/custom-sentence-transformer",
                "/app/local_models/custom-sentence-transformer"
            ]
            
            model_loaded = False
            
            # Try loading from local paths first
            for local_path in local_paths:
                if os.path.exists(local_path):
                    try:
                        print(f"🔧 Attempting to load from local path: {local_path}")
                        
                        # Load tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            local_path,
                            local_files_only=True
                        )
                        
                        # Load model
                        self.model = AutoModel.from_pretrained(
                            local_path,
                            local_files_only=True
                        )
                        
                        print(f"✅ Successfully loaded from local path: {local_path}")
                        model_loaded = True
                        break
                        
                    except Exception as local_e:
                        print(f"⚠️  Failed to load from {local_path}: {str(local_e)}")
                        continue
            
            # If local loading failed and not in offline mode, try Hugging Face
            if not model_loaded and not offline_mode:
                print(f"🌐 Loading from Hugging Face: {self.model_name}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False
                )
                
                # Load model
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=False
                )
                
                model_loaded = True
                print(f"✅ Successfully loaded from Hugging Face")
            
            # If still not loaded, raise error
            if not model_loaded:
                raise Exception("Could not load model from any source (local or remote)")
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings
        Takes attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, sentences: Union[str, List[str]], 
               batch_size: int = 32, 
               show_progress_bar: bool = True,
               convert_to_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress (compatibility)
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Sentence embeddings as numpy array or torch tensor
        """
        
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # Tokenize batch
            encoded_input = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Compute embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
                # Perform pooling
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                
                all_embeddings.append(sentence_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def save(self, path: str):
        """
        Save model and tokenizer to local path
        
        Args:
            path: Directory to save model
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(path)
            
            # Save model  
            self.model.save_pretrained(path)
            
            # Save configuration info
            config = {
                'model_name': self.model_name,
                'embedding_dimension': self.get_sentence_embedding_dimension(),
                'device': str(self.device)
            }
            
            with open(os.path.join(path, 'custom_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Model saved to {path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_local(cls, path: str):
        """
        Load model from local path
        
        Args:
            path: Directory containing saved model
            
        Returns:
            CustomSentenceEmbedder instance
        """
        instance = cls.__new__(cls)
        instance.model_name = path
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load from local path
        instance.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        instance.model = AutoModel.from_pretrained(path, local_files_only=True)
        
        # Move to device
        instance.model.to(instance.device)
        instance.model.eval()
        
        print(f"✅ Model loaded from local path: {path}")
        return instance
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model is None:
            return 384  # Default for MiniLM
        
        # Get dimension from model config
        return self.model.config.hidden_size

# Compatibility function to replace SentenceTransformer
def SentenceTransformer(model_name_or_path: str):
    """
    Drop-in replacement for SentenceTransformer
    
    Args:
        model_name_or_path: Model name or local path
        
    Returns:
        CustomSentenceEmbedder instance
    """
    if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
        # Load from local path
        return CustomSentenceEmbedder.load_local(model_name_or_path)
    else:
        # Load from Hugging Face
        return CustomSentenceEmbedder(model_name_or_path)

# Test function
def test_custom_embedder():
    """Test the custom embedder"""
    try:
        print("🧪 Testing Custom Sentence Embedder...")
        
        # Initialize embedder
        embedder = CustomSentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        
        # Test sentences
        test_sentences = [
            "This is a test sentence.",
            "Another test sentence for embedding.",
            "Banking and finance documentation."
        ]
        
        # Generate embeddings
        embeddings = embedder.encode(test_sentences)
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        print(f"✅ Embedding dimension: {embedder.get_sentence_embedding_dimension()}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        print(f"✅ Similarity matrix shape: {similarities.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = test_custom_embedder()
    
    if success:
        print("\n🎉 Custom embedder working successfully!")
        print("💡 This can replace sentence-transformers in your applications")
    else:
        print("\n❌ Custom embedder test failed")