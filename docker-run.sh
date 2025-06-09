#!/bin/bash

# LinPDE-GP Docker Runner Script

set -e

echo "🐳 LinPDE-GP Docker Runner"
echo "=========================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image"
    echo "  run       - Run the container interactively"
    echo "  test      - Run tests inside the container"
    echo "  jupyter   - Start Jupyter notebook server"
    echo "  shell     - Open a shell in the running container"
    echo "  stop      - Stop the running container"
    echo "  clean     - Remove container and image"
    echo ""
}

# Function to build the Docker image
build_image() {
    echo "🔨 Building Docker image..."
    docker-compose build
    echo "✅ Build complete!"
}

# Function to run the container
run_container() {
    echo "🚀 Starting LinPDE-GP container..."
    docker-compose up -d
    echo "✅ Container started!"
    echo "💡 Use './docker-run.sh shell' to access the container"
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    docker-compose exec linpde-gp pytest
}

# Function to start Jupyter
start_jupyter() {
    echo "📓 Starting Jupyter notebook..."
    docker-compose exec -d linpde-gp jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    echo "✅ Jupyter started!"
    echo "🌐 Access at: http://localhost:8888"
}

# Function to open shell
open_shell() {
    echo "🐚 Opening shell in container..."
    docker-compose exec linpde-gp /bin/bash
}

# Function to stop container
stop_container() {
    echo "🛑 Stopping container..."
    docker-compose down
    echo "✅ Container stopped!"
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up..."
    docker-compose down --rmi all --volumes
    echo "✅ Cleanup complete!"
}

# Main script logic
case "${1:-}" in
    "build")
        build_image
        ;;
    "run")
        build_image
        run_container
        ;;
    "test")
        run_tests
        ;;
    "jupyter")
        start_jupyter
        ;;
    "shell")
        open_shell
        ;;
    "stop")
        stop_container
        ;;
    "clean")
        cleanup
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "")
        echo "❌ No command specified!"
        echo ""
        show_usage
        exit 1
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 