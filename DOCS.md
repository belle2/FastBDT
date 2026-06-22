# FastBDT - User Guide

This document describes how FastBDT works and how to use it from Python, C++ and C.
For the algorithmic background see the [FastBDT paper on arXiv](http://arxiv.org/abs/1609.06119).

---

### 1. What FastBDT does

FastBDT trains a **stochastic gradient-boosted decision tree** (BDT) classifier for binary classification: given training events with real-valued features and a binary label (signal / background), it learns an ensemble of shallow decision trees whose summed, shrunk output estimates how signal-like an event is.

It is optimised for speed, both when fitting and when applying the model:

- **Equal-frequency binning**: each feature is mapped once to a small integer bin index (`2^nCutLevels` bins).
  Training then operates on integers instead of floats, and the cut search is a cheap histogram scan.
- **Cache-friendly layout**: events are stored as a flat, contiguous array of bin indices, accessed linearly.

The result is typically about an order of magnitude faster than general-purpose implementations such as TMVA, scikit-learn and XGBoost, which is why it is used heavily in high-energy physics applications.

#### Key concepts

- **Feature binning**: for each feature the training data is binned into `2^nCutLevels` bins by quantiles (equal frequency).
  Bin `0` is reserved for **NaN**, which FastBDT treats as a genuine "missing / not measured" value and routes through the tree separately - you can pass `NaN` features at training and application time.
- **Trees**: each tree has a fixed depth (`depth` cut layers: up to `2^depth` leaves).
  At each node the best Gini-gain cut over all features/bins is chosen.
- **Boosting**: `nTrees` trees are trained sequentially with gradient boosting; each tree's contribution is scaled by the `shrinkage` (learning rate).
- **Stochastic bagging**: each tree is trained on a random `subsample` fraction of the events.
  `subsample = 1.0` disables bagging and makes training fully deterministic.
- **Output**: the raw model output `F` is the shrunk sum of the boost weights of the leaves an event falls into.
  With `transform2probability` (the default) the output is mapped to a signal probability in `[0, 1]` via `p = 1 / (1 + exp(-2 F))`.

---

### 2. Hyperparameters

The same hyperparameters are exposed by all interfaces.
Defaults differ slightly between the C++ interface and the Python one (noted below).

| Parameter | Meaning | C++ default | Python default |
| --- | --- | --- | --- |
| `nTrees` | Number of boosting iterations (trees). | - (required) | `100` |
| `depth` | Tree depth = number of cut layers; up to `2^depth` leaves. | - (required) | `3` |
| `binning` / `nCutLevels` | Per-feature binning level `N`; the feature gets `2^N` bins. C++ takes one value per feature; Python auto-fills `8` per feature if the list is empty. | - (required) | `[]` -> `8` per feature |
| `shrinkage` | Learning rate; smaller is slower but more stable. | `0.1` | `0.1` |
| `subsample` | Fraction of events used per tree (`1.0` = deterministic). | `1.0` | `0.5` |
| `transform2probability` | Map the score to a probability in `[0, 1]`. | `true` | `True` |
| `purityTransformation` | Per-feature flag to add a purity-ordered version of the feature (see [§6](#specialised-features)). | `{}` | `[]` |
| `sPlot` | Treat the sample weights as sPlot weights (see [§6](#specialised-features)). | `false` | `False` |
| `flatnessLoss` | Strength of uniform boosting; `> 0` enables it (see [§6](#specialised-features)). | `-1.0` | `-1.0` |
| `numberOfFlatnessFeatures` | Number of trailing *spectator* features used only for the flatness penalty (see [§6](#specialised-features)). | `0` | `0` |

---

### 3. Python interface

The Python interface (`PyFastBDT.FastBDT`) is a thin, sklearn-like wrapper over the C API.
It selects the correct weight precision automatically - single vs. double precision is transparent to Python users.

```python
import numpy as np
from PyFastBDT import FastBDT

# X: shape (nEvents, nFeatures), row-major; y: binary labels; weights: optional
X = np.random.normal(size=(10000, 4)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0)

clf = FastBDT.Classifier(nTrees=100, depth=3, binning=[8, 8, 8, 8], shrinkage=0.1)
clf.fit(X, y)                       # optional third arg: per-event weights

p = clf.predict(X)                  # np.array of signal probabilities, shape (nEvents,)
p0 = clf.predict_single(X[0])       # single event -> float

clf.save("model.weightfile")
clf2 = FastBDT.Classifier()
clf2.load("model.weightfile")       # also refreshes clf2's hyperparameter attributes
```

Notes:

- **Input shape**: Python uses **row-major** `X` of shape `(nEvents, nFeatures)`.
  `fit` and `predict` coerce any array-like (lists, `float64`, ...) to the required `float32` internally, so you do not need to pre-convert.
- **`binning` as a plain list works**: `binning=[8, 8, 8, 8]` is fine; the wrapper coerces it to the exact C type.
- **Flatness features**: the spectator columns go as the **last** `numberOfFlatnessFeatures` columns of `X` at `fit` time, and `predict` is then called with only the ordinary feature columns.

#### Feature importance

```python
clf.internFeatureImportance()         # global gain-based ranking, normalised {feature: importance}
clf.individualFeatureImportance(row)  # ranking along one event's decision path
clf.externFeatureImportance(X, y)     # retrain-and-drop importance (Python-only)
```

`internFeatureImportance` and `individualFeatureImportance` wrap the C++ `GetVariableRanking` and `GetIndividualVariableRanking` functions (see below) and return the same quantities.
`externFeatureImportance` is Python-only: it retrains the model while dropping each feature, so it relies on the **default (auto) binning**, and an explicit per-feature `binning` is best left unset when using it.

#### ROC AUC helper

```python
auc = FastBDT.calculate_roc_auc(p, y)  # optional weights as third argument
```

This integrates purity against efficiency (a precision–recall-style area); it is `~0.5` for a random classifier and approaches `1.0` for a perfect one.

---

### 4. C++ interface

From C++ you work with the high-level `FastBDT::Classifier` class, which is declared in `Classifier.h` (this header also pulls in `FastBDT.h`).
For weight variables it is best to use the `FastBDT::Weight` typedef, so that your code stays correct in both single- and double-precision builds.

The snippet below is the C++ counterpart of the Python example above.

```cpp
#include <Classifier.h>

// X is feature-major: X[iFeature][iEvent]
std::vector<std::vector<float>> X = /* nFeatures vectors, each nEvents long */;
std::vector<bool>   y = /* nEvents labels */;
std::vector<FastBDT::Weight> w(y.size(), 1.0);       // weights are required in C++ (1.0 = unweighted)

FastBDT::Classifier clf(100, 3, {8, 8, 8, 8}, 0.1);  // nTrees, depth, binning, shrinkage
clf.fit(X, y, w);

float p = clf.predict({x0, x1, x2, x3});             // signal probability; called once per event

std::ofstream file("model.weightfile");
file << clf;                                         // serialise
std::ifstream in("model.weightfile");
FastBDT::Classifier loaded(in);                      // deserialise
```

Unlike the Python interface, the C++ `fit` takes `X` in **feature-major** order (`X[iFeature][iEvent]`), while `predict` takes a single event as a vector of feature values.

#### Feature importance

```cpp
// global gain-based ranking, normalised {feature index: importance}
std::map<unsigned int, double> ranking = clf.GetVariableRanking();
// ranking along one event's decision path
std::map<unsigned int, double> ind = clf.GetIndividualVariableRanking({x0, x1, x2, x3});
```

These are the same quantities the Python wrapper exposes as `internFeatureImportance` and `individualFeatureImportance` (see above).

---

### 5. C interface

For use from C - or from any language with a C foreign-function interface - FastBDT also offers a plain `extern "C"` API, declared in `FastBDT_C_API.h` and built as `libFastBDT_CInterface`.
This is the same layer the Python bindings are built on.

The API follows the same workflow as `FastBDT::Classifier`.
The snippet below is the C counterpart of the Python and C++ examples above.

```c
#include <FastBDT_C_API.h>

void* clf = Create();
unsigned int binning[4] = {8, 8, 8, 8};
SetBinning(clf, binning, 4);
SetNTrees(clf, 100);
SetDepth(clf, 3);
SetShrinkage(clf, 0.1);

/* data_ptr: nEvents * nFeatures floats, row-major (as in Python). */
/* target_ptr: nEvents labels. weight_ptr may be NULL for unweighted training. */
Fit(clf, data_ptr, weight_ptr, target_ptr, nEvents, nFeatures);

float p = Predict(clf, features);  /* features: nFeatures floats; returns the signal probability */
Save(clf, "model.weightfile");
Delete(clf);                       /* releases the handle */
```

Every `Classifier` method has a direct counterpart here - `Fit`, `Predict`, `PredictArray`, `Save`, `Load`, the `Set*` configuration functions, and the variable-ranking helpers.
`IsWeightFloat()` tells you whether the build uses single- or double-precision weights, which is handy when you fill the `weight_ptr` passed to `Fit`.

---

### 6. Specialised features

- **Purity transformation** (`purityTransformation`): for a selected feature, FastBDT can add a companion feature whose bins are re-ordered by signal purity.
  This can help when a feature's relationship to the label is non-monotonic.
  It makes inference somewhat slower and is off by default.
- **Uniform boosting / flatness loss** (`flatnessLoss`, `numberOfFlatnessFeatures`): when `flatnessLoss > 0`, the boosting adds a penalty that keeps the classifier output **flat** (uniform) with respect to the trailing spectator ("flatness") features, which is useful when a distribution must not be sculpted (e.g. an invariant mass).
  Further detail is in [arXiv:1410.4140](https://arxiv.org/abs/1410.4140).
- **sPlot** (`sPlot`): this enables the special handling needed when you train on sPlot-weighted data, where signal and background are represented statistically via per-event weights rather than hard labels.

---

### 7. Serialisation and weight files

A trained model is serialised to a plain-text **weight file** that stores the hyperparameters, the per-feature binnings, and the forest.
Serialisation is available as `clf.save(path)` / `clf.load(path)` in Python, `operator<<` / the stream constructor in C++, and `Save` / `Load` in the C API.
Loading reconstructs an inference-ready classifier, and in Python `load` also refreshes the object's hyperparameter attributes from the stored model.

Weight files are portable between the Python, C++ and C interfaces of a build with the same weight precision.

---

### 8. Performance

Keeping FastBDT fast is a core project requirement - several tests in `src/test_Performance.cxx` assert that training time scales as expected, and `examples/Benchmark.cxx` measures training, inference, and (de)serialisation across dataset sizes.
Running it before and after changes is a good way to catch regressions:

```bash
g++ examples/Benchmark.cxx -o Benchmark -O3 -lFastBDT_static \
    -L install/lib -I install/include
./Benchmark            # optional: number of repetitions, e.g. ./Benchmark 5
```

Inference uses a "fast forest" whose cuts are stored in float space and laid out contiguously, which keeps large forests cache-resident.

---

### 9. Examples

You can find further examples under `examples/`:

| File | Interface | What it shows |
| --- | --- | --- |
| `IRISExample.cxx` | C++ | End-to-end train/predict on the Iris dataset. |
| `iris_example.py` | Python | The same, via the Python interface. |
| `generic_example.py` | Python | Synthetic Gaussian data; also prints the feature importances. |
| `Benchmark.cxx` | C++ | Training, inference and serialisation timing benchmark. |