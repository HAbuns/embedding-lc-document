"""
Re-generate embeddings using custom embedder (GitHub-only)
This replaces the embeddings created with sentence-transformers
"""

import os
import sys
sys.path.append('.')

import numpy as np
import json
import PyPDF2
from custom_embedder import SentenceTransformer

class GitHubEmbeddingGenerator:
    def __init__(self):
        """Initialize with custom embedder"""
        print("🚀 Initializing GitHub-only embedding generator...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Custom embedder loaded successfully!")
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
                print(f"✅ Extracted {len(text)} characters from {os.path.basename(pdf_path)}")
                return text
                
        except Exception as e:
            print(f"❌ Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text, chunk_size=512, overlap=50):
        """Split text into chunks"""
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                    
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - overlap
            
        return chunks
    
    def generate_embeddings(self, pdf_path, doc_name):
        """Generate embeddings using custom embedder"""
        print(f"\n📄 Processing: {doc_name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"❌ No text extracted from {pdf_path}")
            return None, [], {}
        
        # Create chunks
        chunks = self.chunk_text(text)
        print(f"📝 Created {len(chunks)} chunks")
        
        if not chunks:
            return None, [], {}
        
        # Generate embeddings using custom embedder
        print("🤖 Generating embeddings with custom embedder...")
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Create metadata
        metadata = {
            'document_name': doc_name,
            'num_chunks': len(chunks),
            'embedding_dimension': embeddings.shape[1],
            'model_used': 'Custom Embedder (GitHub-only)',
            'total_embeddings': len(embeddings)
        }
        
        print(f"✅ Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        return embeddings, chunks, metadata
    
    def save_embeddings(self, embeddings, chunks, metadata, output_dir, doc_name):
        """Save embeddings and chunks"""
        if embeddings is None or len(embeddings) == 0:
            print(f"❌ No embeddings to save for {doc_name}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_file = f"{output_dir}/{doc_name}_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save chunks
        chunks_file = f"{output_dir}/{doc_name}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata_file = f"{output_dir}/{doc_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved to:")
        print(f"   - {embeddings_file}")
        print(f"   - {chunks_file}")
        print(f"   - {metadata_file}")

def main():
    """Main function to regenerate embeddings with custom embedder"""
    
    print("=" * 70)
    print("🔧 REGENERATING EMBEDDINGS WITH CUSTOM EMBEDDER (GITHUB-ONLY)")
    print("=" * 70)
    
    # Initialize generator
    generator = GitHubEmbeddingGenerator()
    
    # Document paths
    documents = [
        ("./document/isbp-745.pdf", "isbp-745"),
        ("./document/UCP600-1.pdf", "UCP600-1")
    ]
    
    output_dir = "./embeddings_github"
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Process each document
    for pdf_path, doc_name in documents:
        if os.path.exists(pdf_path):
            # Generate embeddings
            embeddings, chunks, metadata = generator.generate_embeddings(pdf_path, doc_name)
            
            # Save results
            generator.save_embeddings(embeddings, chunks, metadata, output_dir, doc_name)
            
            print(f"✅ Completed: {doc_name}")
            print("-" * 50)
        else:
            print(f"❌ File not found: {pdf_path}")
    
    print("\n" + "=" * 70)
    print("🎉 EMBEDDING REGENERATION COMPLETED!")
    print("=" * 70)
    
    print(f"""
📊 Summary:
   - Model: Custom Embedder (no sentence-transformers)
   - Output: {output_dir}/
   - Dependencies: transformers + torch only
   - GitHub-friendly: ✅
   
🚀 Ready for GitHub-only deployment!

📝 Next steps:
   1. Copy {output_dir}/ to embeddings/ 
   2. Use Dockerfile.github for deployment
   3. Run with docker-compose.github.yml
    """)

if __name__ == "__main__":
    main()