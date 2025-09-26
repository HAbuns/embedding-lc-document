#!/bin/bash

# Docker build debug script for corporate environments
echo "🏢 VPBank Docker Build Diagnostics"
echo "=================================="

echo ""
echo "🔍 Checking corporate network requirements..."

# Check if behind corporate proxy
if [[ -n "$HTTP_PROXY" ]] || [[ -n "$http_proxy" ]]; then
    echo "✅ Corporate proxy detected: $HTTP_PROXY$http_proxy"
    PROXY_ARGS="--build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy"
else
    echo "ℹ️  No proxy environment variables detected"
    PROXY_ARGS=""
fi

echo ""
echo "🐳 Testing Docker build strategies..."

echo ""
echo "Strategy 1: Standard build with corporate compatibility"
echo "----------------------------------------------------"
docker build $PROXY_ARGS -f Dockerfile.github -t vpbank-corporate-standard . 2>&1 | tee build-standard.log

if [ $? -eq 0 ]; then
    echo "✅ Standard build successful!"
    docker tag vpbank-corporate-standard vpbank-corporate:latest
    echo "🎉 Ready to use: docker run -p 8502:8501 vpbank-corporate:latest"
    exit 0
fi

echo ""
echo "❌ Standard build failed. Trying fallback..."
echo ""
echo "Strategy 2: Alpine-based fallback build"
echo "---------------------------------------"
docker build $PROXY_ARGS -f Dockerfile.corporate-fallback -t vpbank-corporate-fallback . 2>&1 | tee build-fallback.log

if [ $? -eq 0 ]; then
    echo "✅ Fallback build successful!"
    docker tag vpbank-corporate-fallback vpbank-corporate:latest
    echo "🎉 Ready to use: docker run -p 8502:8501 vpbank-corporate:latest"
    exit 0
fi

echo ""
echo "❌ All Docker builds failed. Analyzing logs..."
echo ""

echo "🔍 Common corporate build issues:"
echo ""

if grep -q "timeout" build-*.log; then
    echo "⏰ TIMEOUT ISSUE DETECTED"
    echo "   Solution: Add corporate proxy settings"
    echo "   export HTTP_PROXY=http://your-proxy:port"
    echo "   export HTTPS_PROXY=http://your-proxy:port"
    echo ""
fi

if grep -q "SSL" build-*.log; then
    echo "🔒 SSL CERTIFICATE ISSUE DETECTED"
    echo "   Solution: Add --trusted-host flags or corporate certificates"
    echo "   Ask IT for corporate pip configuration"
    echo ""
fi

if grep -q "network" build-*.log; then
    echo "🌐 NETWORK ACCESS ISSUE DETECTED"
    echo "   Solution: Build on machine with internet access"
    echo "   Or ask IT to whitelist PyPI domains"
    echo ""
fi

if grep -q "permission" build-*.log; then
    echo "🔐 PERMISSION ISSUE DETECTED"
    echo "   Solution: Run Docker as administrator/sudo"
    echo "   Or check Docker Desktop settings"
    echo ""
fi

echo ""
echo "🛠️ Alternative solutions:"
echo "1. Use local Python installation (see README)"
echo "2. Ask IT to pre-build Docker image"
echo "3. Use portable Python distribution"
echo ""

echo "📋 Build logs saved to:"
echo "   - build-standard.log"
echo "   - build-fallback.log"
echo ""

echo "💡 For immediate testing without Docker:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install streamlit numpy pandas scikit-learn"  
echo "   streamlit run main_app.py"