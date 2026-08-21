"""Rolling-origin cross-validation used for model selection.

Five folds, a 52-week forecast horizon, a 13-week (one-quarter) stride, and an
expanding training window, as described in the paper's Evaluation Protocol.
No fold sees future weeks; this yields the leakage-free splits over which the
per-series CV metrics are accumulated.
"""

from __future__ import annotations

from typing import Iterator

HORIZON = 52
STRIDE = 13
N_FOLDS = 5
MIN_TRAIN = 110


def rolling_origin_folds(
    n: int,
    horizon: int = HORIZON,
    stride: int = STRIDE,
    n_folds: int = N_FOLDS,
    min_train: int = MIN_TRAIN,
) -> Iterator[tuple[slice, slice]]:
    """Yield (train_slice, test_slice) pairs for a series of length `n`.

    Origins step back by `stride`; the most recent fold tests the last
    `horizon` weeks. Folds whose training window is shorter than `min_train`
    are skipped.
    """
    for k in range(n_folds):
        origin = n - horizon - stride * k
        if origin < min_train:
            break
        yield slice(0, origin), slice(origin, origin + horizon)


if __name__ == "__main__":
    for tr, te in rolling_origin_folds(626):
        print(f"train[0:{tr.stop}]  test[{te.start}:{te.stop}]")
