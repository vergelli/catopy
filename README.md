# Catopy

![Catopy project banner](docs/assets/banner_2.png)


A Python library for zippy tensor operations on CUDA devices.

[![Status](https://img.shields.io/badge/status-WIP-orange)](#status)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![CUDA](https://img.shields.io/badge/CUDA-12.0.140-brightgreen?logo=nvidia&logoColor=white)](#requirements)
[![OS](https://img.shields.io/badge/OS-Linux-informational)](#requirements)

## Table of Contents
- [Overview](#overview)
- [Project intent](#project-intent)
- [Status](#status)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API at a Glance](#api-at-a-glance)
- [Testing and Profiling](#testing-and-profiling)
- [Current Limitations and Roadmap](#current-limitations-and-roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Catopy is a Python library for high-performance tensor operations on CUDA-enabled devices. Right now, it’s in early development (WIP, very WIP), handling basic vector ops only

## Project intent

_This project began as an educational/experimental playground. It does not intent to repleace any mature frameworks_

Contributions that improve clarity, docs, and educational value are welcome! 💚

## Status

**Early Development** - Core vector operations and memory management work, but it’s a **work in progress (WIP)**.

## Requirements

- **CUDA:** 12.0+ (tested on CUDA 12.0.140)
- **GPU:** CUDA-capable GPU, Compute Capability 8.0+ (tested on `sm_80`, Ampere for now.)
- **Python:** 3.10+
- **OS:** Linux (tested on Ubuntu 22.04)
- **Tools:** `uv` (optional, but we love it), `meson`, `ninja`, and `make`

## Installation

### Prerequisites
- Install CUDA 12.0+. Check out [NVIDIA’s CUDA installation guide](https://developer.nvidia.com/cuda-downloads).
- A GPU with compute capability 8.0+ (Ampere or better).
- System dependencies: `libspdlog-dev` (we’ll handle this for you).

### From Source
Clone the repo and let `make` work:

```bash
git clone https://github.com/vergelli/catopy.git
cd catopy

# (recommended) create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

make install-dependencies  # Installs system deps (needs sudo)
make config               # Sets up Meson (run once or after config changes)
make build                # Compiles the C++/CUDA code
make install             # Installs the Python module with uv otherwise `pip install .` as usual.
```

> **Note**: We use `uv` for speedy installs (`pip install uv` to get it). You can swap it for `pip`, `conda`, or `poetry` if you like. See [uv docs](https://github.com/astral-sh/uv) for more.

## Quick Start

#### Assign/access operations

```python
import cato as ca

# Create a vector with 1000 constant values
v = ca.vector(1000, ca.constant(3.14))

v[0] = 42.0
print(v[0])  # Outputs: 42.0

# If you ever need it for some reason.
# It's not the main goal but it's there.
v.ensure_on_gpu()

# Enable debug logging if you want to inspect internals
# This is very verbose so be warned.
ca.logger(True)
```

#### Operations within vectors/scalars

Using normal distribution $\mathbf{v}_i \sim \mathcal{N}(\mu,\sigma^2)$ for example

```python
import cato as ca

ca.vector(5, ca.normal(10, 12))

A=ca.vector(1000000, ca.normal(10, 0.7))
# A is : [8.762258,..., 9.626155], size=1000000

B=ca.vector(1000000, ca.normal(2, 0.3))
# B is : [2.210217,..., 2.046339], size=1000000

A*B
# Output: [19.366493,...,19.698373], size=1000000

A*B*A*B*B
# Output: [828.966325,...,794.032381], size=1000000

A*0
# Output: [0.000000,...,0.000000], size=1000000

A-A
# Output: [0.000000,...,0.000000], size=1000000

A+B
# Output: [10.972475,...,11.672494], size=1000000

```


## API at a Glance

- Initialization: `zeros()`, `ones()`, `constant(c)`, `random(seed?)`, `uniform(a,b,seed?)`, `normal(μ,σ,seed?)`, `box_muller(μ,σ,seed?)`, `sequence(start, step)`, `arange(start, stop, step)`, `sine(freq, phase)`
- Operations: `vecmul(a,b)`, `vecadd(a,b)`, `vecsub(a,b)`, `vecmul_scalar(a,s)`, `vecadd_scalar(a,s)`; Python operators: `*`, `+`, `-`
- See the compact reference: [Vector initialization and ops](docs/VectorOperations.md)


## License

Licensed under the [MIT License](LICENSE).
