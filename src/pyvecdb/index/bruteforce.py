from __future__ import annotations

import numpy as np

from .base import Index


class BruteForce(Index):
    """Exact search. The ground truth every approximate index is measured against."""

    kind = "bruteforce"

    def _insert(self, rows):
        pass

    def _search(self, q, k, **_):
        return self._top_k(q, np.arange(self._n), k)
