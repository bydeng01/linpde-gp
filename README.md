<div align="center">

# LinPDE-GP

**Physics-Informed Gaussian Process Regression for Linear PDE Solvers**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/bydeng01/linpde-gp/ci.yml?branch=main&label=CI)](https://github.com/bydeng01/linpde-gp/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2212.12474-b31b1b.svg)](https://arxiv.org/abs/2212.12474)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

</div>

---

The following work was done by Boyuan Deng:

- Development of the Helmholtz equation module (including the Helmholtz equation, Helmholtz operator, and related components)
- Updating the software and its submodules to use modern dependencies
- Creation and maintenance of Docker files

The software was developed by the author(s) of the paper
"Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers".
If you are using this software in your research, I strongly recommend visiting the
[original repository](https://github.com/marvinpfoertner/linpde-gp).

## Submodules

This project depends on two Git submodules:

| Submodule | Description |
| :--- | :--- |
| [`probnum`](https://github.com/bydeng01/probnum) | Probabilistic numerics library (forked for compatibility) |
| [`gp_constr`](https://github.com/bydeng01/gp_constr) | Gaussian Process regression with linear operator constraints |

Both are fetched automatically when you clone with `--recurse-submodules` (see below).

## Getting Started

### Usage (Docker)

> **Prerequisites:** Install [Docker](https://docs.docker.com/get-docker/) (with Docker Compose).
> On macOS/Windows, Docker Desktop already includes Compose.

**Clone with submodules** (if you haven't already):

```bash
git clone --recurse-submodules https://github.com/bydeng01/linpde-gp.git
cd linpde-gp
```

If you've already cloned it without submodules, you can fetch them afterward by doing:

```bash
cd linpde-gp
git submodule update --init --recursive
```

#### Quick Start (recommended)

**Start the container** (builds if needed and runs in the background):

```bash
./docker-run.sh run
```

**Open a shell inside the running container:**

```bash
./docker-run.sh shell
```

**Run tests:**

```bash
./docker-run.sh test
```

**Start Jupyter Notebook** (accessible on your host at `http://localhost:8888`):

```bash
./docker-run.sh jupyter
```

> **Tip:** If Jupyter asks for a token, show the container logs to find the URL:
>
> ```bash
> docker-compose logs -f linpde-gp
> ```

**Stop the container:**

```bash
./docker-run.sh stop
```

**Clean up everything** (container, image, and volumes like pip cache):

```bash
./docker-run.sh clean
```

#### Notes

- Your project directory is mounted into the container at `/app`, so local code edits are immediately available inside the container.
- Ports `8888` (Jupyter) and `8000` are published by default; adjust `docker-compose.yml` if you need different ports.

<details>
<summary><strong>Manual Docker Compose equivalents</strong></summary>

```bash
# Build
docker-compose build

# Run (detached)
docker-compose up -d

# Shell
docker-compose exec linpde-gp /bin/bash

# Stop
docker-compose down
```

</details>

---

### Usage (Local)

If you prefer to work without Docker, install the submodules as editable packages after cloning:

```bash
pip install -e ./probnum
pip install -e ./gp_constr
pip install -e .
```

## Citation

If you refer to the Helmholtz equation example, please cite the following reference:

```bibtex

```

If you use this software, please cite the original publication:

```bibtex
@misc{Pfoertner2022LinPDEGP,
  author = {Pf\"ortner, Marvin and Steinwart, Ingo and Hennig, Philipp and Wenger, Jonathan},
  title = {Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers},
  year = {2022},
  publisher = {arXiv},
  doi = {10.48550/arxiv.2212.12474},
  url = {https://arxiv.org/abs/2212.12474}
}
```

---

<div align="center">
<sub>Originally developed by Marvin Pf&ouml;rtner et al. &mdash; Extended with Helmholtz equation support by Boyuan Deng.</sub>
</div>
