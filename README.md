# Catopy

![Catopy project banner](docs/assets/banner.png)

A Python library for zippy tensor operations on CUDA devices. Born as a joke, but now I'm kinda serious about it

## Table of Contents
- [Overview](#overview)
- [Status](#status)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Testing and Profiling](#testing-and-profiling)
- [Current Limitations and Roadmap](#current-limitations-and-roadmap)
- [License](#license)

## Overview

Catopy is a Python library for high-performance tensor operations on CUDA-enabled devices. Right now, it’s in early development (WIP, very WIP), handling basic vector ops, but dreaming of full tensor support and all the fancy math stuff. Let’s see where this takes us 😆

## Status

**Early Development** - Core vector operations and memory management work, but it’s a **work in progress (WIP)**. The repo itself is a bit bare-bones (no fancy `LICENSE` or `docs/` yet), but we’re getting there..

## Requirements

- **CUDA:** 12.0+ (tested on CUDA 12.0.140)
- **GPU:** Compute Capability 8.0+ (e.g., NVIDIA Ampere, tested on `sm_80`)
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
make install-dependencies  # Installs system deps (needs sudo)
make config               # Sets up Meson (run once or after config changes)
make build                # Compiles the C++/CUDA code
make install             # Installs the Python module with uv
```

> **Note**: We use `uv` for speedy installs (`pip install uv` to get it). You can swap it for `pip`, `conda`, or `poetry` if you like. See [uv docs](https://github.com/astral-sh/uv) for more.

### Future Plans
The idea is to make it a PyPI package for easier installation, but lets see how it goes though 😅

## Quick Start

```python
import cato as ca

# Create a vector with 1000 random values (normal dist, mean=0, std=1)
v = ca.vector(1000, ca.normal(0, 1))

v[0] = 42.0
print(v[0])  # Outputs: 42.0

# If you ever need it for some reason.
# It snot the main goal but its there.
v.ensure_on_gpu()
```

## Testing and Profiling

Catopy’s got some testing and profiling setup:
- **Run tests**: `make test-frontend` for Python tests (Most of the test, literally), back-end tests are coming soon, probably....
- **Profile performance**: Try `make profile-memory-transfer` to see how data moves between CPU and GPU. Use `make profile-auto-open` to fire up NVIDIA Nsight Systems and geek out on the results. (Check your nsys and nsys-ui apps for this one)

Run `make help` to see all the `make` commands we’ve got. It's on **Spanish** I KNOW.

## Current Limitations and Roadmap

Catopy’s still a kitten, so it’s got some growing pains:
- Only supports **1D vectors** (tensors are coming, maybe ...).
- Just **basic operations** for now (fancy math like `+`, `-`, `*`, `/` **is on the way**).
- **Synchronous transfers** (async streams, not yet..).

### Planned Features
- Full tensor support (2D and beyond).
- Math ops  (matmuls, reductions, etc).
- Async GPU transfers with CUDA streams.
- Playing nice with ML frameworks like PyTorch or NumPy.


## License

Licensed under the [MIT License](LICENSE).
