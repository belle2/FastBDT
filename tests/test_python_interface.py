"""
Regression tests for the PyFastBDT ctypes wrapper (PyFastBDT/FastBDT.py).

These guard against bugs in the Python <-> C-API boundary that the C++
GoogleTest suite cannot catch, in particular:

  * binning passed as a plain Python list must be coerced to uint32 before
    reaching the C API (otherwise a 64-bit numpy default is reinterpreted as
    32-bit and the binning is silently corrupted / aborts).
  * purityTransformation must reach the C API as a bool* (1 byte per element).
  * calculate_roc_auc must work on numpy < 2.0 (np.trapezoid fallback).
  * predict_single must coerce its input dtype like predict does.
  * load() must refresh the Python hyperparameter attributes from the C model.

The build copies the PyFastBDT package (with the shared libraries next to it)
into the CMake binary directory. Put that directory on PYTHONPATH so that
``import PyFastBDT`` resolves against the freshly built libraries, e.g. from the
repository root after building in ``build/``::

    PYTHONPATH=build python3 -m pytest tests/test_python_interface.py

Alternatively, run against an installed PyFastBDT package (e.g. after
``make install``) without setting PYTHONPATH.
"""

import numpy as np
import pytest

from PyFastBDT import FastBDT


# Deterministic training (subsample=1.0) so two classifiers with identical
# configuration produce bit-identical forests.
DETERMINISTIC = dict(nTrees=30, depth=3, subsample=1.0)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0)
    return X, y


def test_binning_plain_list_does_not_crash_and_trains(data):
    # The documented interface is "a list of numbers"; this used to abort
    # because the int64 list was reinterpreted as uint32 -> [4, 0, 4, 0].
    X, y = data
    clf = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    clf.fit(X, y)
    p = clf.predict(X)
    assert p.shape == (len(y),)
    assert np.all(np.isfinite(p))


def test_binning_plain_list_matches_uint32_array(data):
    X, y = data
    clf_list = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    clf_list.fit(X, y)
    clf_arr = FastBDT.Classifier(binning=np.array([4, 4, 4, 4], dtype=np.uint32), **DETERMINISTIC)
    clf_arr.fit(X, y)
    np.testing.assert_array_equal(clf_list.predict(X), clf_arr.predict(X))


def test_purity_transformation_bool_list_matches_int_list(data):
    X, y = data
    clf_bool = FastBDT.Classifier(binning=[4, 4, 4, 4],
                                  purityTransformation=[True, False, True, False],
                                  **DETERMINISTIC)
    clf_bool.fit(X, y)
    clf_int = FastBDT.Classifier(binning=[4, 4, 4, 4],
                                 purityTransformation=[1, 0, 1, 0],
                                 **DETERMINISTIC)
    clf_int.fit(X, y)
    np.testing.assert_array_equal(clf_bool.predict(X), clf_int.predict(X))


def test_calculate_roc_auc_works(data):
    # Exercises the np.trapezoid -> np.trapz fallback on numpy < 2.0.
    X, y = data
    clf = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    clf.fit(X, y)
    auc = FastBDT.calculate_roc_auc(clf.predict(X), y)
    assert 0.5 < auc <= 1.0


def test_predict_single_accepts_float64_row(data):
    X, y = data
    clf = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    clf.fit(X, y)
    expected = clf.predict(X)[0]
    # A float64 row must give the same answer as the float32 batch path.
    got = clf.predict_single(X[0].astype(np.float64))
    assert np.isclose(got, expected, atol=1e-5)


def test_load_refreshes_hyperparameters(tmp_path, data):
    X, y = data
    clf = FastBDT.Classifier(binning=[6, 6, 6, 6], nTrees=42, depth=4,
                             shrinkage=0.05, subsample=0.7,
                             transform2probability=False)
    clf.fit(X, y)
    weightfile = str(tmp_path / "model.weightfile")
    clf.save(weightfile)

    # Fresh classifier with all-default hyperparameters.
    loaded = FastBDT.Classifier()
    assert loaded.nTrees == 100 and loaded.depth == 3  # defaults before load
    loaded.load(weightfile)

    # After load the attributes must reflect the saved model, not the defaults.
    assert loaded.nTrees == 42
    assert loaded.depth == 4
    assert np.isclose(loaded.shrinkage, 0.05)
    assert np.isclose(loaded.subsample, 0.7)
    assert loaded.transform2probability is False


def test_save_load_round_trip_predictions(tmp_path, data):
    X, y = data
    clf = FastBDT.Classifier(binning=[6, 6, 6, 6], **DETERMINISTIC)
    clf.fit(X, y)
    weightfile = str(tmp_path / "model.weightfile")
    clf.save(weightfile)

    loaded = FastBDT.Classifier()
    loaded.load(weightfile)
    # Within float32 serialisation tolerance (matches the C++ EXPECT_FLOAT_EQ tests).
    np.testing.assert_allclose(clf.predict(X), loaded.predict(X), atol=1e-5)
