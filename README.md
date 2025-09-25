# 📚 VPBank Document Search Engine# 📚 VPBank Document Search Engine



A sophisticated document search engine with **multi-method similarity comparison** for financial documents. Supports both **offline local development** and **Docker deployment** with pre-trained embeddings.A sophisticated document search engine with **multi-method similarity comparison** for financial documents. Supports both **offline local development** and **Docker deployment** with pre-trained embeddings.



## ✨ Features## ✨ Features



- 🔍 **Semantic Document Search** - Advanced vector-based document retrieval- 🔍 **Semantic Document Search** - Advanced vector-based document retrieval

- 📊 **Multi-Method Similarity** - Compare 5 different similarity algorithms- 📊 **Multi-Method Similarity** - Compare 5 different similarity algorithms

- 🎯 **Offline-First Design** - Works without internet connectivity- 🎯 **Offline-First Design** - Works without internet connectivity

- 🐳 **Docker Ready** - Easy containerized deployment- 🐳 **Docker Ready** - Easy containerized deployment

- 🚀 **Local Development** - Python virtual environment setup- 🚀 **Local Development** - Python virtual environment setup

- 📈 **Interactive UI** - Beautiful Streamlit interface with comparison tables- 📈 **Interactive UI** - Beautiful Streamlit interface with comparison tables

- **Offline Capable**: Pre-downloaded models for restricted environments

## 🏗️ Project Structure

## 📁 Repository Structure

```

embedding-lc-document/```

├── streamlit_search_app_github.py    # Main Streamlit application├── custom_embedder.py              # Custom embedding implementation

├── custom_embedder.py               # Custom embedding implementation├── regenerate_embeddings_github.py # Generate embeddings with custom embedder

├── similarity_methods.py            # 5 similarity calculation methods├── streamlit_search_app_github.py  # Main Streamlit application

├── Dockerfile.github               # Docker configuration├── Dockerfile.github               # Docker configuration for GitHub-only

├── docker-compose.github.yml       # Docker Compose setup├── docker-compose.github.yml       # Docker Compose orchestration

├── requirements_docker.txt         # Docker dependencies├── deploy_github.sh               # Automated deployment script

├── embeddings/                     # Pre-generated document embeddings├── verify_deployment.sh           # Deployment verification script

├── embeddings_github/              # GitHub-ready embeddings├── requirements_docker.txt        # Python dependencies

├── local_models/                   # Local transformer models├── GITHUB_DEPLOYMENT_GUIDE.md    # Detailed deployment guide

├── document/                       # Source documents (PDF)├── document/                     # PDF source documents

├── deploy_github.sh               # Docker deployment script├── embeddings_github/           # Generated embeddings data

└── verify_deployment.sh           # Deployment verification└── local_models/               # Pre-downloaded model files

``````



## 🚀 Quick Start## 🚀 Quick Start



### Option 1: Docker Deployment (Recommended)### Prerequisites

- Docker and Docker Compose installed

```bash- Git installed

# 1. Clone repository- At least 4GB free disk space

git clone https://github.com/HAbuns/embedding-lc-document.git

cd embedding-lc-document### 1. Clone Repository

```bash

# 2. Build and run with Docker Composegit clone https://github.com/HAbuns/embedding-lc-document.git

docker-compose -f docker-compose.github.yml up --buildcd embedding-lc-document

```

# 3. Access application

open http://localhost:8501### 2. Deploy with Docker

``````bash

chmod +x deploy_github.sh

**Or using Docker directly:**./deploy_github.sh

```bash```

# Build image

docker build -f Dockerfile.github -t vpbank-search-engine:latest .### 3. Access Application

Open browser: `http://localhost:8502`

# Run container

docker run -p 8501:8501 --name vpbank-search vpbank-search-engine:latest## 🔧 Manual Setup



# Check logs### Build Docker Image

docker logs vpbank-search```bash

```docker-compose -f docker-compose.github.yml build

```

### Option 2: Local Development

### Run Application

```bash```bash

# 1. Clone repositorydocker-compose -f docker-compose.github.yml up -d

git clone https://github.com/HAbuns/embedding-lc-document.git```

cd embedding-lc-document

### Verify Deployment

# 2. Create virtual environment```bash

python3 -m venv venv./verify_deployment.sh

source venv/bin/activate  # On Windows: venv\Scripts\activate```



# 3. Install dependencies## 📊 Usage

pip install streamlit torch transformers scikit-learn numpy pandas

1. **Search Documents**: Enter your query in the search box

# 4. Run application2. **View Results**: See top 5 most relevant document chunks

streamlit run streamlit_search_app_github.py3. **Response Time**: Monitor search performance

4. **Similarity Scores**: Review relevance confidence

# 5. Access application

open http://localhost:8501## 🛠️ Technical Details

```

- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

## 🎯 Similarity Methods- **Embedding Method**: Mean pooling of token embeddings

- **Search Method**: Cosine similarity

The engine supports **5 different similarity calculation methods**:- **Framework**: Streamlit + transformers + torch

- **Deployment**: Docker with offline model caching

| Method | Description | Range | Best For |

|--------|-------------|-------|----------|## 📋 Requirements

| **Cosine Similarity** | Angle between vectors | [0, 1] | General purpose, semantic similarity |

| **Euclidean Distance** | L2 distance converted to similarity | [0, 1] | Geometric similarity, exact matches |See `requirements_docker.txt` for complete dependency list:

| **Manhattan Distance** | L1 distance converted to similarity | [0, 1] | Robust to outliers |- transformers==4.56.2

| **Dot Product** | Vector multiplication | [-∞, ∞] | Magnitude-aware similarity |- torch==2.8.0

| **Jaccard Similarity** | Set intersection over union | [0, 1] | Binary/categorical features |- streamlit==1.50.0

- numpy==2.3.3

## 📊 Score Interpretation- scikit-learn==1.7.2

- plotly==6.3.0

- **🟢 High Score (> 0.7)**: Excellent match, highly relevant- PyPDF2==3.0.1

- **🟡 Medium Score (0.4 - 0.7)**: Good match, moderately relevant  

- **🔴 Low Score (< 0.4)**: Weak match, less relevant## 🐳 Docker Environment



## 🐳 Docker ConfigurationThe system runs in a containerized environment with:

- Offline transformers mode

### Environment Variables- Local model caching

```bash- Read-only document/embedding volumes

TRANSFORMERS_OFFLINE=1      # Enable offline mode- Health checks

HF_HUB_OFFLINE=1           # Disable HuggingFace hub downloads- Auto-restart capability

TRANSFORMERS_CACHE=/app/cache  # Cache directory

```## 📖 Documentation



### Volume Mounts- **[GITHUB_DEPLOYMENT_GUIDE.md](GITHUB_DEPLOYMENT_GUIDE.md)**: Complete deployment instructions

```yaml- **Custom Embedder**: See `custom_embedder.py` for implementation details

volumes:- **Regenerate Data**: Use `regenerate_embeddings_github.py` to update embeddings

  - ./embeddings:/app/embeddings:ro       # Document embeddings

  - ./local_models:/app/local_models:ro   # Pre-trained models## 🔒 Security & Corporate Environment

  - ./document:/app/document:ro           # Source documents

```This system is designed for restricted corporate environments:

- No external API calls during runtime

### Health Check- Pre-downloaded model files

```bash- GitHub-only code dependencies

# Check container health- Offline transformer operations

docker exec vpbank-search curl -f http://localhost:8501/_stcore/health- Docker isolation



# View application logs## 🚀 Advanced Usage

docker logs -f vpbank-search

```### Regenerate Embeddings

```bash

## 🛠️ Developmentpython regenerate_embeddings_github.py

```

### File Organization

### Add New Documents

- **`streamlit_search_app_github.py`** - Main Streamlit UI and search logic1. Place PDF files in `document/` folder

- **`custom_embedder.py`** - Custom transformer implementation (no sentence-transformers)2. Run embedding regeneration script

- **`similarity_methods.py`** - Multi-method similarity calculations3. Rebuild Docker image

- **`embeddings/`** - Pre-generated embeddings for offline search

- **`local_models/`** - Downloaded transformer models for offline inference### Custom Configuration

Edit environment variables in `docker-compose.github.yml`

### Key Components

## 🤝 Contributing

1. **GitHubOnlyVectorSearchEngine** - Main search engine class

2. **CustomSentenceEmbedder** - Custom embedding without external dependencies1. Fork the repository

3. **MultiSimilarityCalculator** - 5 similarity methods in one class2. Create feature branch: `git checkout -b feature-name`

3. Commit changes: `git commit -am 'Add feature'`

### Adding New Documents4. Push branch: `git push origin feature-name`

5. Submit pull request

```bash

# 1. Add PDF to document/ folder## 📄 License

cp new_document.pdf document/

This project is proprietary software for VPBank internal use.

# 2. Regenerate embeddings

python regenerate_embeddings_github.py## 🆘 Support



# 3. Restart applicationFor issues and questions:

streamlit run streamlit_search_app_github.py1. Check deployment logs: `docker logs vpbank-document-search-github`

```2. Run verification script: `./verify_deployment.sh`

3. Review deployment guide: `GITHUB_DEPLOYMENT_GUIDE.md`

## 🔧 Troubleshooting

---

### Common Issues**Built for VPBank** | **GitHub-Only Deployment** | **Docker Ready**

**1. Model Loading Errors**
```bash
# Check if local models exist
ls -la local_models/sentence-transformer/

# Regenerate if missing
python regenerate_embeddings_github.py
```

**2. Docker Build Issues**
```bash
# Clean build cache
docker system prune -f

# Rebuild with no cache
docker build --no-cache -f Dockerfile.github -t vpbank-search-engine:latest .
```

**3. Port Already in Use**
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run streamlit_search_app_github.py --server.port=8502
```

**4. Empty Search Results**
```bash
# Verify embeddings exist
ls -la embeddings/
ls -la embeddings_github/

# Check document files
ls -la document/

# Regenerate embeddings if needed
python regenerate_embeddings_github.py
```

## 📈 Performance Optimization

### Local Development
- Use SSD storage for faster model loading
- Increase RAM allocation for Docker Desktop
- Enable GPU acceleration if available

### Docker Production
```yaml
# docker-compose.github.yml
services:
  vpbank-search:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## 🔒 Security Considerations

- Container runs as non-root user
- No external network calls in offline mode
- Environment variables for sensitive configuration
- Health checks for container monitoring

## 📝 API Usage

### Search Endpoints
```python
# Single method search
results = search_engine.search(query="credit card rules", top_k=5)

# Multi-method comparison
results = search_engine.search_with_multiple_methods(query="banking terms", top_k=5)
```

### Response Format
```python
{
    'cosine': [
        {
            'content': 'Document chunk text...',
            'similarity': 0.8542,
            'metadata': {'source': 'document.pdf', 'page': 1}
        }
    ],
    'euclidean': [...],
    'manhattan': [...],
    'dot_product': [...],
    'jaccard': [...]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For issues and questions:
- 🐛 Issues: [GitHub Issues](https://github.com/HAbuns/embedding-lc-document/issues)
- 📖 Documentation: [GitHub Wiki](https://github.com/HAbuns/embedding-lc-document/wiki)

---

**Made with ❤️ for VPBank Document Search**