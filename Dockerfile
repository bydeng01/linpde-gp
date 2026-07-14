# Use Python 3.11 slim image as base
# Copyright (c) 2025, Boyuan Deng
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
    fontconfig \
    fonts-freefont-ttf \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-science \
    cm-super \
    dvipng \
    ghostscript \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# --- R + R packages for gp_constr's R-backed constrained GP (truncated-MVN) ---
# The constrained arms (quotes+C, quotes+E+C) call gp_constr's R backend via
# rpy2; without R baked in here the backend silently degrades to NaN and only
# the unconstrained arms produce numbers. Install Debian's own r-base (no CRAN
# apt repo: it avoids a codename/key dependency that breaks the build, and on a
# trixie base this is already R 4.5.x; bookworm ships 4.2/4.3).
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages for the truncated-MVN sampler. install.packages takes the current
# CRAN versions; if the exact paper versions are required (mvtnorm 1.3-6,
# tmvtnorm 1.7, TruncatedNormal 2.3, truncnorm 1.0-9, R 4.5.0) pin them with
# remotes::install_version() and use the CRAN apt repo for the R 4.5 line.
RUN Rscript -e 'install.packages(c("mvtnorm","tmvtnorm","TruncatedNormal","truncnorm"), repos="https://cloud.r-project.org")'

# Copy the requirements files first for better Docker layer caching
COPY dev-requirements.txt ./
COPY linting-requirements.txt ./
COPY formatting-requirements.txt ./

# Copy the tests requirements if they exist
COPY tests/ tests/
COPY experiments/ experiments/

# Copy submodules
COPY probnum/ probnum/
COPY gp_constr/ gp_constr/

# Copy the main project files
COPY . .

# Initialize git repository and submodules for setuptools-scm
RUN git init . || echo "Git already initialized"
RUN git add . || echo "Files already added"
RUN git config user.email "docker@example.com" && git config user.name "Docker Build"
RUN git commit -m "Initial commit" || echo "Already committed"

# Initialize git for submodules too
RUN cd probnum && git init . && git add . && git commit -m "probnum initial commit" || echo "probnum git already set up"
RUN cd gp_constr && git init . && git add . && git commit -m "gp_constr initial commit" || echo "gp_constr git already set up"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel

# Set version for setuptools-scm for both packages
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LINPDE_GP=0.1.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PROBNUM=0.1.25

# Install submodules first
RUN pip install -e ./probnum
# [r] extra pulls rpy2 (>=3.5,<3.6); R is installed above so rpy2 builds/links.
RUN pip install -e './gp_constr[r]'

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