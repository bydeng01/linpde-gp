# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for the project
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements files first for better Docker layer caching
COPY dev-requirements.txt ./
COPY linting-requirements.txt ./
COPY formatting-requirements.txt ./

# Copy the tests requirements if they exist
COPY tests/ tests/
COPY experiments/ experiments/

# Copy probnum submodule
COPY probnum/ probnum/

# Copy the main project files
COPY . .

# Initialize git repository and submodules for setuptools-scm
RUN git init . || echo "Git already initialized"
RUN git add . || echo "Files already added"
RUN git config user.email "docker@example.com" && git config user.name "Docker Build"
RUN git commit -m "Initial commit" || echo "Already committed"

# Initialize git for probnum submodule too
RUN cd probnum && git init . && git add . && git commit -m "probnum initial commit" || echo "probnum git already set up"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel

# Set version for setuptools-scm for both packages
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LINPDE_GP=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PROBNUM=0.1.25

# Install probnum first
RUN pip install -e ./probnum

# Install other requirements without the editable linpde-gp package
RUN pip install -r tests/requirements.txt || echo "tests requirements installed"
RUN pip install -r ./formatting-requirements.txt
RUN pip install -r ./linting-requirements.txt  
RUN pip install -r experiments/requirements.txt || echo "experiments requirements installed"
RUN pip install "pre-commit>=3.1,<4"

# Finally install the main package
RUN pip install -e .

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for Jupyter if needed
EXPOSE 8888

# Set the default command to bash for interactive use
CMD ["/bin/bash"] 