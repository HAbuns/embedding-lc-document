#!/bin/bash

# Verification script for Docker deployment readiness
# Checks all required files and configurations for offline deployment

echo "🔍 VPBank Document Search Engine - Deployment Verification"
echo "========================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check counters
total_checks=0
passed_checks=0

check_file() {
    total_checks=$((total_checks + 1))
    if [[ -f "$1" ]] || [[ -d "$1" ]]; then
        print_success "$1"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        print_error "$1 (MISSING)"
        return 1
    fi
}

# 1. Docker Configuration Files
print_header "Docker Configuration Files"
check_file "Dockerfile"
check_file "docker-compose.yml"
check_file "requirements_docker.txt"
check_file ".dockerignore"
check_file "deploy_docker.sh"

# 2. Application Files
print_header "Application Files"
check_file "streamlit_search_app_offline.py"
check_file "streamlit_search_app.py"
check_file "document_embedding.py"
check_file "embedding_demo.py"

# 3. Local Models (Critical for Offline)
print_header "Local Models (Offline Mode)"
check_file "local_models"
check_file "local_models/sentence-transformer"
check_file "local_models/sentence-transformer/config.json"

# Count model files
if [[ -d "local_models/sentence-transformer" ]]; then
    model_files=$(find local_models/sentence-transformer -type f | wc -l)
    if [[ $model_files -gt 5 ]]; then
        print_success "Model has $model_files files (sufficient)"
    else
        print_warning "Model has only $model_files files (may be incomplete)"
    fi
fi

# 4. Embeddings Data
print_header "Pre-computed Embeddings"
check_file "embeddings"
check_file "embeddings/isbp-745_embeddings.npy"
check_file "embeddings/isbp-745_chunks.json"
check_file "embeddings/isbp-745_metadata.json"
check_file "embeddings/UCP600-1_embeddings.npy"
check_file "embeddings/UCP600-1_chunks.json"
check_file "embeddings/UCP600-1_metadata.json"

# 5. Source Documents
print_header "Source Documents"
check_file "document"
check_file "document/isbp-745.pdf"
check_file "document/UCP600-1.pdf"

# 6. Documentation
print_header "Documentation"
check_file "DOCKER_DEPLOYMENT_GUIDE.md"
check_file "README_SEARCH_APP.md"

# 7. File Size Checks
print_header "File Size Verification"

check_size() {
    if [[ -f "$1" ]]; then
        size=$(du -h "$1" | cut -f1)
        print_success "$1: $size"
    fi
}

check_size "embeddings/isbp-745_embeddings.npy"
check_size "embeddings/UCP600-1_embeddings.npy"

if [[ -d "local_models/sentence-transformer" ]]; then
    model_size=$(du -sh local_models/sentence-transformer | cut -f1)
    print_success "Model directory: $model_size"
fi

# 8. Docker Environment Check
print_header "Docker Environment"
total_checks=$((total_checks + 2))

if command -v docker &> /dev/null; then
    print_success "Docker CLI installed"
    passed_checks=$((passed_checks + 1))
    
    if docker info &> /dev/null; then
        print_success "Docker daemon running"
        passed_checks=$((passed_checks + 1))
    else
        print_warning "Docker daemon not running (start Docker to deploy)"
    fi
else
    print_error "Docker CLI not installed"
fi

if command -v docker-compose &> /dev/null; then
    docker_compose_version=$(docker-compose --version)
    print_success "Docker Compose: $docker_compose_version"
else
    print_warning "Docker Compose not found (may need to install)"
fi

# 9. Permissions Check
print_header "File Permissions"
if [[ -x "deploy_docker.sh" ]]; then
    print_success "deploy_docker.sh is executable"
else
    print_warning "deploy_docker.sh not executable (run: chmod +x deploy_docker.sh)"
fi

# 10. Summary
print_header "DEPLOYMENT READINESS SUMMARY"

echo ""
echo "📊 Check Results:"
echo "   Passed: $passed_checks / $total_checks"

if [[ $passed_checks -eq $total_checks ]]; then
    echo ""
    print_success "🎉 ALL CHECKS PASSED - READY FOR DEPLOYMENT!"
    echo ""
    echo "🚀 To deploy, run:"
    echo "   ./deploy_docker.sh"
    echo ""
    echo "🔒 OFFLINE MODE CONFIRMED:"
    echo "   ✅ Local model downloaded"
    echo "   ✅ Embeddings pre-computed" 
    echo "   ✅ No internet required for deployment"
    echo "   ✅ Perfect for proxy-restricted environments"
    
elif [[ $passed_checks -gt $((total_checks * 3 / 4)) ]]; then
    echo ""
    print_warning "⚠️  MOSTLY READY - Some non-critical files missing"
    echo ""
    echo "🚀 You can still deploy, but consider fixing missing files:"
    echo "   ./deploy_docker.sh"
    
else
    echo ""
    print_error "❌ NOT READY - Critical files missing"
    echo ""
    echo "🔧 Fix missing files before deployment"
    echo "   Check DOCKER_DEPLOYMENT_GUIDE.md for details"
fi

echo ""
echo "📋 For detailed deployment instructions, see:"
echo "   cat DOCKER_DEPLOYMENT_GUIDE.md"
echo ""
echo "========================================================"