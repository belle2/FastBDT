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

Beyond those regression guards, the suite also exercises the general behaviour
of the interface: that the model actually learns, the probability transform,
batch/single-prediction consistency, dtype/list robustness of fit and predict,
sample weights, NaN-as-missing handling, the feature-importance helpers,
calculate_roc_auc, flatness features, determinism and file save/load.

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


# ---------------------------------------------------------------------------
# Additional fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    """Two well-separated Gaussian blobs along feature 0 (AUC ~ 1)."""
    rng = np.random.RandomState(1)
    n = 500
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = np.zeros(n, dtype=bool)
    y[: n // 2] = True
    X[y, 0] += 3.0
    X[~y, 0] -= 3.0
    return X, y


@pytest.fixture
def importance_data():
    """Label is fully determined by feature 0; features 1 and 2 are noise."""
    rng = np.random.RandomState(2)
    n = 600
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] > 0.0)
    return X, y


# ---------------------------------------------------------------------------
# Learning / output behaviour
# ---------------------------------------------------------------------------

def test_model_learns_on_separable_data(separable_data):
    X, y = separable_data
    clf = FastBDT.Classifier(binning=[8, 8], nTrees=50, depth=3, subsample=1.0)
    clf.fit(X, y)
    assert FastBDT.calculate_roc_auc(clf.predict(X), y) > 0.99


def test_single_feature_works():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(400, 1)).astype(np.float32)
    y = (X[:, 0] > 0.0)
    clf = FastBDT.Classifier(binning=[6], nTrees=20, depth=2, subsample=1.0)
    clf.fit(X, y)
    p = clf.predict(X)
    assert p.shape == (len(y),)
    assert FastBDT.calculate_roc_auc(p, y) > 0.95


def test_probabilities_in_unit_interval_and_transform_relationship(data):
    # transform2probability=True must give probabilities in [0, 1]; the raw
    # score path must satisfy prob == 1 / (1 + exp(-2 * raw)).
    X, y = data
    clf_prob = FastBDT.Classifier(binning=[4, 4, 4, 4], transform2probability=True, **DETERMINISTIC)
    clf_prob.fit(X, y)
    clf_raw = FastBDT.Classifier(binning=[4, 4, 4, 4], transform2probability=False, **DETERMINISTIC)
    clf_raw.fit(X, y)

    prob = clf_prob.predict(X)
    raw = clf_raw.predict(X)
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)
    np.testing.assert_allclose(prob, 1.0 / (1.0 + np.exp(-2.0 * raw)), atol=1e-4)


def test_training_is_deterministic_with_full_subsample(data):
    X, y = data
    a = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    a.fit(X, y)
    b = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# ---------------------------------------------------------------------------
# Input robustness (dtype / container coercion)
# ---------------------------------------------------------------------------

def test_predict_accepts_list_and_float64(data):
    X, y = data
    clf = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    clf.fit(X, y)
    ref = clf.predict(X)
    np.testing.assert_allclose(clf.predict(X.tolist()), ref, atol=1e-6)
    np.testing.assert_allclose(clf.predict(X.astype(np.float64)), ref, atol=1e-6)


def test_predict_single_matches_batch(data):
    X, y = data
    clf = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    clf.fit(X, y)
    batch = clf.predict(X)
    for i in (0, 1, 137, len(X) - 1):
        assert np.isclose(clf.predict_single(X[i]), batch[i], atol=1e-5)


def test_fit_accepts_float64_X_and_integer_labels(data):
    X, y = data
    a = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    a.fit(X.astype(np.float64), y.astype(np.int32))
    b = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-6)


def test_nan_feature_is_treated_as_missing(data):
    # FastBDT maps NaN to the dedicated missing bin; predictions stay finite.
    X, y = data
    clf = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    clf.fit(X, y)
    Xn = X.copy()
    Xn[:5, 0] = np.nan
    assert np.all(np.isfinite(clf.predict(Xn)))


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def test_sample_weights_change_the_model(data):
    X, y = data
    base = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    base.fit(X, y)
    w = np.where(y, 5.0, 1.0).astype(np.float32)
    weighted = FastBDT.Classifier(binning=[4, 4, 4, 4], **DETERMINISTIC)
    weighted.fit(X, y, w)
    pw = weighted.predict(X)
    assert np.all(np.isfinite(pw))
    assert not np.allclose(base.predict(X), pw)


# ---------------------------------------------------------------------------
# Flatness (spectator) features
# ---------------------------------------------------------------------------

def test_flatness_features_train_and_predict(importance_data):
    # The last feature is a spectator used only for the flatness penalty; the
    # model is trained on real + spectator features but predicts on the real
    # features (GetNFeatures = total - numberOfFlatnessFeatures).
    X, y = importance_data  # 3 columns: features 0,1 real, 2 used as flatness
    clf = FastBDT.Classifier(binning=[6, 6, 6], nTrees=20, depth=2, subsample=1.0,
                             flatnessLoss=1.0, numberOfFlatnessFeatures=1)
    clf.fit(X, y)
    p = clf.predict(X[:, :2])
    assert p.shape == (len(y),)
    assert np.all(np.isfinite(p))


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------

def test_intern_feature_importance_ranks_informative_feature(importance_data):
    X, y = importance_data
    clf = FastBDT.Classifier(binning=[8, 8, 8], nTrees=50, depth=3, subsample=1.0)
    clf.fit(X, y)
    ranking = clf.internFeatureImportance()
    assert isinstance(ranking, dict) and len(ranking) >= 1
    assert all(v >= 0.0 for v in ranking.values())
    assert sum(ranking.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0 in ranking and ranking[0] == max(ranking.values())


def test_individual_feature_importance(importance_data):
    X, y = importance_data
    clf = FastBDT.Classifier(binning=[8, 8, 8], nTrees=50, depth=3, subsample=1.0)
    clf.fit(X, y)
    ranking = clf.individualFeatureImportance(X[0])
    assert isinstance(ranking, dict) and len(ranking) >= 1
    assert all(v >= 0.0 for v in ranking.values())
    assert sum(ranking.values()) == pytest.approx(1.0, abs=1e-6)


def test_extern_feature_importance_identifies_informative_feature(importance_data):
    # externFeatureImportance retrains while dropping each feature, so it relies
    # on the default (auto) binning to adapt to the reduced feature count.
    X, y = importance_data
    clf = FastBDT.Classifier(nTrees=20, depth=2, subsample=1.0)
    clf.fit(X, y)
    importances = clf.externFeatureImportance(X, y)
    assert isinstance(importances, dict) and len(importances) >= 1
    assert all(np.isfinite(v) for v in importances.values())
    # Removing the only informative feature must hurt the most.
    assert max(importances, key=importances.get) == 0


# ---------------------------------------------------------------------------
# calculate_roc_auc
# ---------------------------------------------------------------------------

def test_calculate_roc_auc_separation_behaviour():
    rng = np.random.RandomState(3)
    n = 2000
    t = (np.arange(n) % 2 == 0).astype(float)
    good = np.where(t > 0.5, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    rand = rng.uniform(0.0, 1.0, n)

    auc_good = FastBDT.calculate_roc_auc(good, t)
    auc_bad = FastBDT.calculate_roc_auc(1.0 - good, t)
    auc_rand = FastBDT.calculate_roc_auc(rand, t)

    assert auc_good > 0.97
    assert auc_bad < auc_rand < auc_good
    assert 0.4 < auc_rand < 0.6

    # A weighted call must run and stay within [0, 1].
    w = rng.uniform(0.5, 2.0, n)
    assert 0.0 <= FastBDT.calculate_roc_auc(good, t, w) <= 1.0


# ---------------------------------------------------------------------------
# Save / load to a real file
# ---------------------------------------------------------------------------

def test_save_load_predicts_on_fresh_data(tmp_path, data):
    X, y = data
    clf = FastBDT.Classifier(binning=[5, 5, 5, 5], **DETERMINISTIC)
    clf.fit(X, y)
    weightfile = str(tmp_path / "model.weightfile")
    clf.save(weightfile)

    loaded = FastBDT.Classifier()
    loaded.load(weightfile)

    rng = np.random.RandomState(9)
    Xnew = rng.normal(size=(64, 4)).astype(np.float32)
    np.testing.assert_allclose(clf.predict(Xnew), loaded.predict(Xnew), atol=1e-5)
