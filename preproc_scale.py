"""Batch-friendly feature scaling helpers (ISS-04)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def fit_transform_scaled(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    batch_size: int = 10_000,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit ``StandardScaler`` on **all** rows, then transform in batches.

    Historically trainers called ``fit_transform`` on the first batch only and
    ``transform`` on the rest, so mean/variance ignored most of the data.
    """
    if scaler is None:
        scaler = StandardScaler()

    # Fit uses the full frame (already in memory after CSV load/concat).
    scaler.fit(X)

    scaled_batches = []
    batch_iter = range(0, len(X), batch_size)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc="Scaling batches")

    for start in batch_iter:
        batch = X.iloc[start : start + batch_size]
        scaled_batches.append(scaler.transform(batch))

    scaled = pd.DataFrame(np.vstack(scaled_batches), columns=X.columns, index=X.index)
    return scaled, scaler
