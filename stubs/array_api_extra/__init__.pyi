"""Stubs for array-api-extra Array type to work with FastSIMUS Array.

References:
    https://data-apis.org/array-api-extra/api-reference.html
"""

from fast_simus.utils._array_api import Array, ArrayOrScalar, _ArrayNamespace

def isclose(
    a: ArrayOrScalar,
    b: ArrayOrScalar,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    xp: _ArrayNamespace | None = None,
) -> Array: ...
def sinc(x: Array, *, xp: _ArrayNamespace | None = None) -> Array: ...
