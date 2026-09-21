"""Tests for the public backend-neutral JIT helper."""

from __future__ import annotations

from typing import cast

import numpy as np

from fast_simus import jit
from fast_simus.utils._array_api import ArrayNamespace


def test_jit_is_noop_without_backend_compiler() -> None:
    """Array API namespaces without a JIT compiler keep eager execution."""

    def add_one(value):
        return value + 1

    compiled = jit(add_one, xp=cast(ArrayNamespace, np))

    assert compiled is add_one
    np.testing.assert_array_equal(compiled(np.asarray([1, 2])), np.asarray([2, 3]))
