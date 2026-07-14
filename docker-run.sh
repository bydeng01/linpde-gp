#!/bin/bash

# LinPDE-GP Docker Runner Script
# Copyright (c) 2025, Boyuan Deng

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
    echo "  nbconvert <notebook.ipynb> - Headlessly execute a notebook (output: <name>.executed.ipynb)"
    echo "  metrics [N_PDE]            - Run the instrumented 3D Helmholtz metrics script (default N_PDE=10)"
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
    echo "⏳ Waiting for server to initialize..."
    sleep 3
    echo "🔑 Jupyter Token:"
    docker-compose exec linpde-gp jupyter notebook list
    echo "🌐 Access at: http://localhost:8888"
}

# Function to headlessly execute a notebook
run_nbconvert() {
    local notebook="$1"
    if [ -z "$notebook" ]; then
        echo "❌ No notebook specified!"
        echo "   Usage: $0 nbconvert experiments/your_notebook.ipynb"
        exit 1
    fi
    local output="${notebook%.ipynb}.executed.ipynb"
    echo "📓 Executing $notebook headlessly (no kernel timeout)..."
    echo "   Output -> $output"
    # --rm: discard the throwaway container; the repo is bind-mounted, so the
    #       executed notebook is written straight back to the host.
    # timeout=-1: never kill long-running cells (Gram assembly can be slow).
    docker-compose run --rm linpde-gp \
        python -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        "$notebook" --output "$(basename "$output")"
    echo "✅ Done. Executed notebook: $output"
}

# Function to run the instrumented 3D Helmholtz metrics script
run_metrics() {
    local n_pde="${1:-10}"
    echo "📐 Running 3D Helmholtz metrics (N_pde=${n_pde})..."
    echo "   Watch the last [RSS ...] line: if it gets killed, that line is where memory spiked."
    # -u: unbuffered, so progress/RSS lines flush before any OOM kill.
    docker-compose run --rm linpde-gp \
        python -u experiments/compute_metrics_helmholtz3d.py --n-pde "${n_pde}"
    echo "✅ Done. Metrics JSON: experiments/helmholtz3d_complex_metrics.json"
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
    "nbconvert")
        run_nbconvert "$2"
        ;;
    "metrics")
        run_metrics "$2"
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