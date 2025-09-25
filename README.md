# 📚 VPBank Document Search Engine

**Advanced multi-method semantic document search system with beautiful UI and offline capabilities**

![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Quick Demo

```bash
# One-command deployment
git clone https://github.com/HAbuns/embedding-lc-document.git
cd embedding-lc-document
docker-compose -f docker-compose.github.yml up --build
open http://localhost:8501
```

## ✨ Key Features

- 🔍 **Advanced Semantic Search** - Vector-based document retrieval
- 📊 **5 Similarity Methods** - Cosine, Euclidean, Manhattan, Dot Product, Jaccard  
- 🎯 **Offline Operation** - No internet required after setup
- 🐳 **Docker Ready** - One-click containerized deployment
- 🚀 **Local Dev Friendly** - Simple Python virtual environment
- 📈 **Beautiful UI** - Interactive Streamlit interface with comparison tables
- 🔒 **Production Ready** - Security hardened with health checks

## 🎯 Similarity Methods Comparison

| Method | Best For | Score Range | Description |
|--------|----------|-------------|-------------|
| **Cosine** | General semantic search | [0, 1] | Angle between vectors |
| **Euclidean** | Exact matches | [0, 1] | Geometric distance |
| **Manhattan** | Robust search | [0, 1] | Sum of absolute differences |
| **Dot Product** | Magnitude-aware | [-∞, ∞] | Vector multiplication |
| **Jaccard** | Binary features | [0, 1] | Set similarity |

## 📊 Score Interpretation

- 🟢 **High (>0.7)**: Excellent match, highly relevant
- 🟡 **Medium (0.4-0.7)**: Good match, moderately relevant  
- 🔴 **Low (<0.4)**: Weak match, less relevant

## 🚀 Deployment Options

### Option 1: Docker (Recommended)
```bash
git clone https://github.com/HAbuns/embedding-lc-document.git
cd embedding-lc-document
docker-compose -f docker-compose.github.yml up --build
```

### Option 2: Local Development
```bash
git clone https://github.com/HAbuns/embedding-lc-document.git
cd embedding-lc-document
python3 -m venv venv && source venv/bin/activate
pip install streamlit torch transformers scikit-learn numpy pandas
streamlit run streamlit_search_app_github.py
```

## 🏗️ Project Structure

```
embedding-lc-document/
├── streamlit_search_app_github.py    # Main application
├── custom_embedder.py               # Custom embedding engine
├── similarity_methods.py            # 5 similarity algorithms
├── Dockerfile.github               # Docker configuration
├── docker-compose.github.yml       # Docker Compose
├── requirements_docker.txt         # Dependencies
├── embeddings/                     # Pre-generated embeddings
├── local_models/                   # Transformer models
└── document/                       # Source documents
```

