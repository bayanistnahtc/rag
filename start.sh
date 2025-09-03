#!/bin/bash

# MedRAG Assistant - Startup Script
echo "🚀 Starting MedRAG Assistant..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p vector_store
mkdir -p cached_recordings

echo "📁 Creating necessary directories..."

# Build and start the services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose ps

echo ""
echo "✅ MedRAG Assistant is starting up!"
echo ""
echo "🌐 Services will be available at:"
echo "   - Backend API: http://localhost:8000"
echo "   - Frontend UI: http://localhost:8501"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
echo ""
echo "🎉 Enjoy using MedRAG Assistant!" 