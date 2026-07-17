<div align="center">

# LinPDE-GP

**Physics-Informed Gaussian Process Regression for Linear PDE Solvers**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/bydeng01/linpde-gp/ci.yml?branch=main&label=CI)](https://github.com/bydeng01/linpde-gp/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2607.14193-b31b1b.svg)](https://arxiv.org/abs/2607.14193)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

</div>

---

This fork builds on the original `linpde-gp` framework in two directions.

>Research extensions:

- Helmholtz equation support, including the Helmholtz operator and related components
- An application to *in vivo* brain magnetic resonance elastography (MRE)
- Additional research modules are in preparation...

>Maintenance and reproducibility:

- Updates to the software and submodules for modern dependencies
- A maintained Docker build

The original framework was developed by the author(s) of
"Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers." If you use
this software in your research, we strongly recommend also visiting the
[original repository](https://github.com/marvinpfoertner/linpde-gp).

## Projects

Each project is self-contained and has its own README with the science, the data it needs,
and exact reproduce steps. Start at the front door here (install once), then follow the
relevant guide.

| Project | Status | Guide |
| :--- | :--- | :--- |
| **Helmholtz equation & Brain MRE** — physics-informed GP solver for the inhomogeneous Helmholtz BVP (real & complex, 1D/2D/3D) with FDM/PINN baselines, and its application to *in vivo* brain magnetic resonance elastography | Open-source | [`docs/HELMHOLTZ.md`](docs/HELMHOLTZ.md) |

## Submodules

This project depends on two Git submodules:

| Submodule | Description |
| :--- | :--- |
| [`probnum`](https://github.com/bydeng01/probnum) | Probabilistic numerics library (forked for compatibility) |

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

If you use or refer to the Helmholtz equation module or the brain MRE application in this fork, please cite our paper.

```bibtex
@misc{deng2026operatorinformedgaussianprocessescomplex,
      title={Operator-Informed Gaussian Processes for Complex Helmholtz Wavefields: From Synthetic Benchmarks to In Vivo Brain Elastography}, 
      author={Boyuan Deng and Kshitiz Upadhyay and Michael Shields},
      year={2026},
      eprint={2607.14193},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2607.14193}, 
}
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

## License

Released under the [MIT License](LICENSE).

---

<!-- <div align="center">
<sub></sub>
</div> -->