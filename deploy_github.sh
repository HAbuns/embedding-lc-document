#!/bin/bash

# GitHub-only deployment script for VPBank Document Search Engine
# This version uses custom embedder instead of sentence-transformers

set -e

echo "📦 VPBank Document Search Engine - GitHub-Only Deployment"
echo "========================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check Docker
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Check required files for GitHub-only deployment
required_files=(
    "Dockerfile.github"
    "docker-compose.github.yml" 
    "requirements_docker.txt"
    "custom_embedder.py"
    "streamlit_search_app_github.py"
)

echo ""
echo "🔍 Checking GitHub-only deployment files..."
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "$file exists"
    else
        print_error "$file is missing!"
        exit 1
    fi
done

# Check if we have GitHub-generated embeddings or need to use existing ones
if [[ -d "embeddings_github" ]]; then
    print_info "Using GitHub-generated embeddings from embeddings_github/"
    if [[ ! -d "embeddings" ]] || [[ "$1" == "--use-github-embeddings" ]]; then
        print_info "Copying GitHub embeddings to embeddings/ directory..."
        cp -r embeddings_github/* embeddings/ 2>/dev/null || mkdir -p embeddings && cp -r embeddings_github/* embeddings/
        print_success "GitHub embeddings copied"
    fi
else
    print_warning "No embeddings_github/ found, using existing embeddings/"
    if [[ ! -d "embeddings" ]]; then
        print_error "No embeddings directory found! Please run regenerate_embeddings_github.py first"
        exit 1
    fi
fi

# Check embeddings
if [[ -f "embeddings/isbp-745_embeddings.npy" ]] && [[ -f "embeddings/UCP600-1_embeddings.npy" ]]; then
    print_success "Embeddings files found"
else
    print_error "Embeddings files missing! Please run regenerate_embeddings_github.py first"
    exit 1
fi

echo ""
echo "🏗️  Building GitHub-only Docker image..."
echo "This uses custom embedder instead of sentence-transformers..."

# Build the Docker image using GitHub-specific files
if docker-compose -f docker-compose.github.yml build --no-cache; then
    print_success "Docker image built successfully with custom embedder"
else
    print_error "Failed to build Docker image"
    exit 1
fi

echo ""
echo "🚀 Starting GitHub-only Docker container..."

# Stop any existing container
docker-compose -f docker-compose.github.yml down 2>/dev/null || true

# Start the container
if docker-compose -f docker-compose.github.yml up -d; then
    print_success "Docker container started successfully"
else
    print_error "Failed to start Docker container"
    exit 1
fi

echo ""
echo "⏳ Waiting for GitHub-only application to start..."
sleep 15

# Test application accessibility
echo "🧪 Testing GitHub-only application..."
max_attempts=12
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8502/_stcore/health > /dev/null 2>&1; then
        print_success "GitHub-only application is healthy and accessible!"
        break
    else
        echo "Attempt $attempt/$max_attempts: Waiting for application..."
        sleep 5
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    print_error "Application failed to start properly"
    echo ""
    echo "📋 Container logs:"
    docker-compose -f docker-compose.github.yml logs
    exit 1
fi

# Show container status
echo ""
echo "📊 Container Status:"
docker-compose -f docker-compose.github.yml ps

# Show application logs
echo ""
echo "📋 Application Logs (last 20 lines):"
docker-compose -f docker-compose.github.yml logs --tail=20

echo ""
echo "🎉 SUCCESS! GitHub-only deployment completed successfully!"
echo ""
echo "🌐 Application is now running at:"
echo "   • Local:    http://localhost:8502"
echo "   • Network:  http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo 'YOUR_IP'):8502"
echo ""
echo "🔧 Docker Commands:"
echo "   • View logs:     docker-compose -f docker-compose.github.yml logs -f"
echo "   • Stop:          docker-compose -f docker-compose.github.yml down"
echo "   • Restart:       docker-compose -f docker-compose.github.yml restart"
echo "   • Rebuild:       docker-compose -f docker-compose.github.yml build --no-cache && docker-compose -f docker-compose.github.yml up -d"
echo ""
echo "📦 GITHUB-ONLY MODE CONFIRMED:"
echo "   ✅ Custom embedder (no sentence-transformers)"
echo "   ✅ Uses only transformers + torch"
echo "   ✅ Perfect for GitHub-only environments"
echo "   ✅ Drop-in replacement functionality"
echo ""
echo "🧪 Test the app:"
echo "   1. Open http://localhost:8502"
echo "   2. Try search: 'letter of credit requirements'"
echo "   3. Verify custom embedder is working"
echo ""
echo "========================================================="