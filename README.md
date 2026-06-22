# FastBDT

Stochastic gradient-boosted decision trees for binary classification, with interfaces for Python, C++ and C.
FastBDT is a speed-optimised, cache-friendly implementation that is about an order of magnitude faster than general-purpose implementations such as TMVA, scikit-learn and XGBoost, at both training and application.
It is used extensively in high-energy physics by the Belle II Collaboration.

**Check the paper on arXiv**: [FastBDT: A speed-optimized and cache-friendly implementation of stochastic gradient-boosted decision trees for multivariate classification](http://arxiv.org/abs/1609.06119)

---

### Important

This repository is a fork maintained by the Belle II Collaboration.
It is guaranteed to compile with modern compilers and the unit tests and main examples are fully functional, unless stated otherwise.

The original repository can be found at: https://github.com/thomaskeck/FastBDT

---

### Installation

#### From `conda-forge` (recommended for users)

FastBDT is packaged on [`conda-forge`](https://conda-forge.org/), which ships the Python bindings together with the pre-compiled C++ and C libraries, so no manual build is required.
Two packages are available, differing only in the internal weight precision (see [Weight type and numerical precision](#weight-type-and-numerical-precision)):

| Package | Weight | Downloads | Version | Platforms |
| --- | --- | --- | --- | --- |
| `fastbdt` | `float` | [![](https://img.shields.io/conda/dn/conda-forge/fastbdt.svg)](https://anaconda.org/conda-forge/fastbdt) | [![](https://img.shields.io/conda/vn/conda-forge/fastbdt.svg)](https://anaconda.org/conda-forge/fastbdt) | [![](https://img.shields.io/conda/pn/conda-forge/fastbdt.svg)](https://anaconda.org/conda-forge/fastbdt) |
| `fastbdt-double-weight` | `double` | [![](https://img.shields.io/conda/dn/conda-forge/fastbdt-double-weight.svg)](https://anaconda.org/conda-forge/fastbdt-double-weight) | [![](https://img.shields.io/conda/vn/conda-forge/fastbdt-double-weight.svg)](https://anaconda.org/conda-forge/fastbdt-double-weight) | [![](https://img.shields.io/conda/pn/conda-forge/fastbdt-double-weight.svg)](https://anaconda.org/conda-forge/fastbdt-double-weight) |

Using `conda`:

```bash
conda install -c conda-forge fastbdt
# or, for the double-precision build:
conda install -c conda-forge fastbdt-double-weight
```

Using [`pixi`](https://pixi.sh/) inside a project (`conda-forge` is the default channel):

```bash
pixi add fastbdt
# or, for the double-precision build:
pixi add fastbdt-double-weight
```

To install it into a global pixi environment instead of a project, use

```bash
pixi global install fastbdt
# or, for the double-precision build:
pixi global install fastbdt-double-weight
```

#### From source

To build and install FastBDT from source, use the following commands:

```bash
mkdir -p build install && cd build
cmake ..
make
make install
```

This will also install the Python bindings automatically if CMake detects a valid `python3` interpreter during the configuration step.
To build the double-precision variant from source, see [Weight type and numerical precision](#weight-type-and-numerical-precision).

---

### Usage

Typically, you will want to use FastBDT as a library integrated directly into your application. Available interfaces:

- the Python library `PyFastBDT/FastBDT.py` (see `examples/iris_example.py` and `examples/generic_example.py`)
- the C++ shared/static library (see `examples/IRISExample.cxx`)
- the C shared library

For a broader description of how FastBDT works, its configuration options, and the Python, C++ and C APIs, see **[DOCS.md](DOCS.md)**.

---

### Weight type and numerical precision

By default, FastBDT uses **single-precision floating point** (`float`) as type for internal weights in the C++ implementation. This choice is made for performance reasons and is sufficient for most use cases.
If higher numerical precision is required, a **double-precision floating point** (`double`) build is available in two ways:

- **from `conda-forge`**: install the `fastbdt-double-weight` package instead of `fastbdt` (see [Installation](#installation)).
- **from source**: enable the corresponding CMake option at configuration time:

  ```bash
  cmake .. -DUSE_DOUBLE_WEIGHT=ON
  ```

Either way, this changes the internal weight type used throughout the FastBDT codebase.

#### Weight type in Python

The Python interface automatically handles the internal weight type and requires no user action.
Switching between single and double precision is entirely transparent to Python users.

#### Weight type in C++

When working with FastBDT in C++, it is strongly recommended to use the type alias `FastBDT::Weight`, which is available via the header `FastBDT.h`, for all weight-related variables, rather than explicitly using `float` or `double`.
This ensures that user code remains compatible regardless of whether FastBDT is built with single or double precision.

---

### Further reading

This work is mostly based on the papers by Jerome H. Friedman
  * https://jerryfriedman.su.domains/ftp/trebst.pdf
  * https://jerryfriedman.su.domains/ftp/stobst.pdf

FastBDT also implements the uniform gradient boosting techniques to boost to flatness:
  * https://arxiv.org/abs/1410.4140