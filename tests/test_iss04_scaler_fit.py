"""ISS-04: StandardScaler must be fit on all rows, not the first batch only."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preproc_scale import fit_transform_scaled


def test_scaler_sees_all_rows_not_just_first_batch():
    # First batch is all zeros; later batches are large — a first-batch-only fit
    # would learn mean≈0/std≈0 for early columns and distort later values.
    n = 25
    batch = 10
    X = pd.DataFrame(
        {
            "a": np.concatenate([np.zeros(batch), np.linspace(100, 200, n - batch)]),
            "b": np.arange(n, dtype=float),
        }
    )

    scaled, scaler = fit_transform_scaled(
        X, StandardScaler(), batch_size=batch, show_progress=False
    )

    assert scaler.n_samples_seen_ == n
    np.testing.assert_allclose(scaler.mean_, X.mean().to_numpy())
    # Scaled column b should be approximately zero-mean unit-variance.
    np.testing.assert_allclose(scaled["b"].mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled["b"].std(ddof=0), 1.0, atol=1e-9)


def test_legacy_first_batch_fit_would_differ():
    n = 25
    batch = 10
    X = pd.DataFrame({"a": np.concatenate([np.zeros(batch), np.ones(n - batch) * 50])})

    correct, correct_scaler = fit_transform_scaled(
        X, StandardScaler(), batch_size=batch, show_progress=False
    )

    legacy = StandardScaler()
    legacy.fit(X.iloc[:batch])
    assert legacy.n_samples_seen_ == batch
    assert correct_scaler.n_samples_seen_ == n
    assert not np.allclose(legacy.mean_, correct_scaler.mean_)
    assert correct.shape == X.shape
