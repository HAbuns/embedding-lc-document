# 🏢 VPBank Corporate Document Search# 📚 VPBank Document Search Engine



**Enterprise-grade document search system designed for corporate environments with restricted internet access****Advanced multi-method semantic document search system with beautiful UI and offline capabilities**



![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)

![Corporate](https://img.shields.io/badge/corporate-friendly-green.svg)![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

![Offline](https://img.shields.io/badge/offline-ready-orange.svg)![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-red.svg)

![No AI Deps](https://img.shields.io/badge/no%20AI%20deps-red.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)



## 🎯 Perfect for Corporate Environments## 🚀 Quick Demo



✅ **NO** transformers, huggingface-hub, torch dependencies  ```bash

✅ **NO** external model downloads or API calls  # One-command deployment

✅ **NO** internet connection required after deployment  git clone https://github.com/HAbuns/embedding-lc-document.git

✅ **Firewall friendly** - works behind corporate proxies  cd embedding-lc-document

✅ **Lightweight** - minimal dependencies  docker-compose -f docker-compose.github.yml up --build

open http://localhost:8501

## 🚀 Quick Start```



```bash## ✨ Key Features

# One-command deployment

git clone https://github.com/HAbuns/embedding-lc-document.git- 🔍 **Advanced Semantic Search** - Vector-based document retrieval

cd embedding-lc-document- 📊 **5 Similarity Methods** - Cosine, Euclidean, Manhattan, Dot Product, Jaccard  

docker-compose -f docker-compose.github.yml up --build- 🎯 **Offline Operation** - No internet required after setup

- 🐳 **Docker Ready** - One-click containerized deployment

# Access at http://localhost:8502- 🚀 **Local Dev Friendly** - Simple Python virtual environment

```- 📈 **Beautiful UI** - Interactive Streamlit interface with comparison tables

- 🔒 **Production Ready** - Security hardened with health checks

## ✨ Features

## 🎯 Similarity Methods Comparison

- 🔍 **Smart Text Search** - TF-IDF vectorization + keyword matching

- 📊 **5 Similarity Methods** - Cosine, Euclidean, Manhattan, Dot Product, Jaccard| Method | Best For | Score Range | Description |

- ⚡ **Fast Response** - Sub-second search performance|--------|----------|-------------|-------------|

- 🎨 **Beautiful UI** - Modern Streamlit interface with charts| **Cosine** | General semantic search | [0, 1] | Angle between vectors |

- 🔒 **Corporate Security** - No external dependencies or data transfer| **Euclidean** | Exact matches | [0, 1] | Geometric distance |

- 📱 **Responsive Design** - Works on desktop and mobile| **Manhattan** | Robust search | [0, 1] | Sum of absolute differences |

| **Dot Product** | Magnitude-aware | [-∞, ∞] | Vector multiplication |

## 🛠️ Technology Stack| **Jaccard** | Binary features | [0, 1] | Set similarity |



### Core Dependencies (Minimal Set)## 📊 Score Interpretation

```

streamlit==1.50.0      # Web interface- 🟢 **High (>0.7)**: Excellent match, highly relevant

numpy==2.3.3           # Numerical operations  - 🟡 **Medium (0.4-0.7)**: Good match, moderately relevant  

pandas==2.3.2          # Data handling- 🔴 **Low (<0.4)**: Weak match, less relevant

scikit-learn==1.7.2    # TF-IDF vectorization only

plotly==6.3.0           # Interactive charts## 🚀 Deployment Options

```

### Option 1: Docker (Recommended)

### What's NOT Included```bash

```git clone https://github.com/HAbuns/embedding-lc-document.git

❌ torch               # Too heavy, often blockedcd embedding-lc-document

❌ transformers        # Requires external model downloadsdocker-compose -f docker-compose.github.yml up --build

❌ huggingface-hub     # Blocked by corporate firewalls```

❌ sentence-transformers # Not available in restricted environments

```### Option 2: Local Development

```bash

## 📊 Search Methods Comparisongit clone https://github.com/HAbuns/embedding-lc-document.git

cd embedding-lc-document

| Method | Algorithm | Best For | Corporate Friendly |python3 -m venv venv && source venv/bin/activate

|--------|-----------|----------|-------------------|pip install streamlit torch transformers scikit-learn numpy pandas

| **Cosine** | TF-IDF vectors | General semantic search | ✅ |streamlit run streamlit_search_app_github.py

| **Euclidean** | Word matching | Exact term matches | ✅ |```

| **Manhattan** | Character similarity | Fuzzy matching | ✅ |

| **Dot Product** | Weighted scoring | Length-aware search | ✅ |## 🏗️ Project Structure

| **Jaccard** | Set similarity | Keyword overlap | ✅ |

```

## 🏗️ Project Structureembedding-lc-document/

├── streamlit_search_app_github.py    # Main application

```├── custom_embedder.py               # Custom embedding engine

embedding-lc-document/├── similarity_methods.py            # 5 similarity algorithms

├── main_app.py                     # Main Streamlit application├── Dockerfile.github               # Docker configuration

├── pure_offline_search.py          # Search engine (NO AI deps)├── docker-compose.github.yml       # Docker Compose

├── similarity_methods.py           # 5 similarity algorithms├── requirements_docker.txt         # Dependencies

├── requirements_docker.txt         # Minimal dependencies├── embeddings/                     # Pre-generated embeddings

├── Dockerfile.github              # Corporate-friendly container├── local_models/                   # Transformer models

├── docker-compose.github.yml      # Production deployment└── document/                       # Source documents

├── test_deployment.sh             # Automated testing```

├── embeddings/                    # Document data

└── document/                      # Source PDFs
```

## 🚀 Deployment Options

### Option 1: Docker (Recommended)
```bash
# Production deployment
docker-compose -f docker-compose.github.yml up --build

# Access application
open http://localhost:8502
```

### Option 2: Local Development
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal dependencies
pip install streamlit numpy pandas scikit-learn plotly requests PyPDF2 Pillow

# Run application
streamlit run main_app.py
```

## 🧪 Testing & Validation

```bash
# Run automated tests
chmod +x test_deployment.sh
./test_deployment.sh

# Manual testing
python3 -c "
from pure_offline_search import PureOfflineSearch
search = PureOfflineSearch()
results = search.similarity_search('banking services', top_k=3)
print(f'Found {len(results)} results')
"
```

## 🔒 Corporate Security Features

### Data Privacy
- ✅ All processing happens locally
- ✅ No data sent to external services
- ✅ No telemetry or tracking
- ✅ No external API calls

### Network Security
- ✅ Works completely offline
- ✅ No outbound internet connections
- ✅ Firewall friendly
- ✅ Proxy server compatible

## 🐳 Docker Configuration

```yaml
services:
  vpbank-search-corporate:
    build: .
    ports: ["8502:8501"]
    environment:
      - OFFLINE_MODE=1
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

## 🔧 Troubleshooting

### Common Issues

**Corporate Firewall Issues**
- ✅ This system works completely offline
- ✅ No external connections required
- ✅ All dependencies pre-installed in Docker
- ✅ No model downloads needed

**No Search Results**
```bash
# Check document files
ls -la document/ embeddings/

# Test search engine
python3 -c "from pure_offline_search import PureOfflineSearch; PureOfflineSearch()"
```

## 📄 License

MIT License - Perfect for corporate use.

---

<div align="center">

**🏢 Built for Corporate Excellence**

*Secure • Offline • Fast • Reliable*

</div>