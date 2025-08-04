# FastBDT

Stochastic gradient-boosted decision trees for multivariate classification, usable standalone and via Python interface.

**Check the paper on ArXiv: [FastBDT: A speed-optimized and cache-friendly implementation of stochastic gradient-boosted decision trees for multivariate classification](http://arxiv.org/abs/1609.06119)**

Stochastic gradient-boosted decision trees are widely employed for multivariate classification and regression tasks.
This paper presents a speed-optimized and cache-friendly implementation for multivariate classification called FastBDT.
FastBDT is one order of magnitude faster during the fitting and application phases compared to popular implementations in frameworks like TMVA, scikit-learn, and XGBoost.
The concepts used to optimize execution time and performance are discussed in detail in this paper. Key ideas include:

- equal-frequency binning on the input data, which allows replacing expensive floating-point operations with integer operations while improving classification quality;
- a cache-friendly linear access pattern to the input data, in contrast to typical implementations that exhibit random access patterns.

FastBDT provides interfaces to C/C++ and Python.
It is extensively used in high energy physics by the Belle II Collaboration.

---

### Warning

This repository is a fork maintained by the Belle II Collaboration.
It is guaranteed to compile with modern compilers and the unit tests and main examples are fully functional, unless stated otherwise.
However, **no further development of this fork is currently planned**.

The original repository can be found at: https://github.com/thomaskeck/FastBDT

---

### Installation

To build and install FastBDT, use the following commands:

```bash
mkdir -p build install && cd build
cmake ..
make
make install
```

This will also install the Python bindings automatically if CMake detects a valid `python3` interpreter during the configuration step.

---

### Usage

Typically, you will want to use FastBDT as a library integrated directly into your application. Available interfaces:

- the C++ shared/static library (see `examples/IRISExample.cxx`)
- the C shared library
- the Python library `PyFastBDT/FastBDT.py` (see `examples/iris_example.py` and `examples/generic_example.py`)

---

### Further reading

This work is mostly based on the papers by Jerome H. Friedman
  * https://jerryfriedman.su.domains/ftp/trebst.pdf
  * https://jerryfriedman.su.domains/ftp/stobst.pdf

FastBDT also implements the uniform gradient boosting techniques to boost to flatness:
  * https://arxiv.org/abs/1410.4140