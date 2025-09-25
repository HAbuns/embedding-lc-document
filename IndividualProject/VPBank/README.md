# VPBank Document Embedding & Search System

A semantic search system for VPBank documents using custom embeddings and Streamlit interface. This system is designed for GitHub-only deployment in restricted corporate environments.

## 🎯 Features

- **Custom Embedding**: Uses transformers + torch (no sentence-transformers dependency)
- **Semantic Search**: Find relevant document chunks using cosine similarity
- **Streamlit Interface**: Clean, responsive web interface
- **Docker Deployment**: Containerized for easy deployment
- **GitHub-Only**: Works without pip install, only with GitHub code
- **Offline Capable**: Pre-downloaded models for restricted environments

## 📁 Repository Structure

```
├── custom_embedder.py              # Custom embedding implementation
├── regenerate_embeddings_github.py # Generate embeddings with custom embedder
├── streamlit_search_app_github.py  # Main Streamlit application
├── Dockerfile.github               # Docker configuration for GitHub-only
├── docker-compose.github.yml       # Docker Compose orchestration
├── deploy_github.sh               # Automated deployment script
├── verify_deployment.sh           # Deployment verification script
├── requirements_docker.txt        # Python dependencies
├── GITHUB_DEPLOYMENT_GUIDE.md    # Detailed deployment guide
├── document/                     # PDF source documents
├── embeddings_github/           # Generated embeddings data
└── local_models/               # Pre-downloaded model files
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git installed
- At least 4GB free disk space

### 1. Clone Repository
```bash
git clone https://github.com/HAbuns/embedding-lc-document.git
cd embedding-lc-document
```

### 2. Deploy with Docker
```bash
chmod +x deploy_github.sh
./deploy_github.sh
```

### 3. Access Application
Open browser: `http://localhost:8502`

## 🔧 Manual Setup

### Build Docker Image
```bash
docker-compose -f docker-compose.github.yml build
```

### Run Application
```bash
docker-compose -f docker-compose.github.yml up -d
```

### Verify Deployment
```bash
./verify_deployment.sh
```

## 📊 Usage

1. **Search Documents**: Enter your query in the search box
2. **View Results**: See top 5 most relevant document chunks
3. **Response Time**: Monitor search performance
4. **Similarity Scores**: Review relevance confidence

## 🛠️ Technical Details

- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Embedding Method**: Mean pooling of token embeddings
- **Search Method**: Cosine similarity
- **Framework**: Streamlit + transformers + torch
- **Deployment**: Docker with offline model caching

## 📋 Requirements

See `requirements_docker.txt` for complete dependency list:
- transformers==4.56.2
- torch==2.8.0
- streamlit==1.50.0
- numpy==2.3.3
- scikit-learn==1.7.2
- plotly==6.3.0
- PyPDF2==3.0.1

## 🐳 Docker Environment

The system runs in a containerized environment with:
- Offline transformers mode
- Local model caching
- Read-only document/embedding volumes
- Health checks
- Auto-restart capability

## 📖 Documentation

- **[GITHUB_DEPLOYMENT_GUIDE.md](GITHUB_DEPLOYMENT_GUIDE.md)**: Complete deployment instructions
- **Custom Embedder**: See `custom_embedder.py` for implementation details
- **Regenerate Data**: Use `regenerate_embeddings_github.py` to update embeddings

## 🔒 Security & Corporate Environment

This system is designed for restricted corporate environments:
- No external API calls during runtime
- Pre-downloaded model files
- GitHub-only code dependencies
- Offline transformer operations
- Docker isolation

## 🚀 Advanced Usage

### Regenerate Embeddings
```bash
python regenerate_embeddings_github.py
```

### Add New Documents
1. Place PDF files in `document/` folder
2. Run embedding regeneration script
3. Rebuild Docker image

### Custom Configuration
Edit environment variables in `docker-compose.github.yml`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is proprietary software for VPBank internal use.

## 🆘 Support

For issues and questions:
1. Check deployment logs: `docker logs vpbank-document-search-github`
2. Run verification script: `./verify_deployment.sh`
3. Review deployment guide: `GITHUB_DEPLOYMENT_GUIDE.md`

---
**Built for VPBank** | **GitHub-Only Deployment** | **Docker Ready**