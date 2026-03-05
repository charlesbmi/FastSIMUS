"""Array-API utilities."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self, cast, runtime_checkable

from array_api_compat import array_namespace as xpc_array_namespace


@runtime_checkable
class LinAlg(Protocol):
    """Protocol for linear algebra extension conforming to Array API standard.

    This is an optional extension - not all array libraries implement it.
    See: https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html
    """

    def vector_norm(self, x: Array, *, axis: Any = None, keepdims: bool = False, ord: Any = None) -> Array: ...


@runtime_checkable
class FFT(Protocol):
    """Protocol for FFT extension conforming to Array API standard.

    This is an optional extension - not all array libraries implement it.
    See: https://data-apis.org/array-api/2023.12/extensions/fourier_transform_functions.html

    This protocol includes only the FFT functions commonly used in PyMUST:
    - rfft/irfft: Real FFT for efficient processing of real-valued signals
    - rfftfreq: Frequency bins for real FFT
    - fftshift: Shift zero-frequency component to center
    """

    def rfft(self, x: Array, /, *, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array: ...
    def irfft(self, x: Array, /, *, n: int | None = None, axis: int = -1, norm: str = "backward") -> Array: ...
    def rfftfreq(self, n: int, /, *, d: float = 1.0, device: Any = None) -> Array: ...
    def fftshift(self, x: Array, /, *, axes: int | tuple[int, ...] | None = None) -> Array: ...


@runtime_checkable
class _ArrayNamespace(Protocol):
    """Protocol for array namespaces that conform to the Array API standard.

    This covers the common operations and data types used throughout the FastSIMUS codebase.
    Based on the Array API specification: https://data-apis.org/array-api/latest/

    Notes:
        - This base protocol does NOT include optional extensions (linalg, fft).
            Use the extended protocols for namespaces that have these extensions.
        - We remove some Array API standard functions that are not used in the codebase,
            to be lenient about other array libraries like MLX
    """

    # Data types
    float32: Any
    float64: Any
    complex64: Any
    int8: Any
    int16: Any
    int32: Any
    int64: Any
    uint8: Any
    uint16: Any
    uint32: Any
    uint64: Any

    # Constants
    pi: float

    # Creation functions
    def asarray(self, obj: Any, *, dtype: Any = None, device: Any = None, copy: bool = False) -> Array: ...
    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Any = None,
        device: Any = None,
    ) -> Array: ...
    def linspace(
        self,
        start: int | float | complex,
        stop: int | float | complex,
        /,
        num: int,
        *,
        dtype: Any = None,
        device: Any = None,
        endpoint: bool = True,
    ) -> Array: ...
    def meshgrid(self, *arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> list[Array]: ...
    def ones(self, shape: int | tuple[int, ...], *, dtype: Any = None, device: Any = None) -> Array: ...
    def ones_like(self, x: Array, /, *, dtype: Any = None, device: Any = None) -> Array: ...
    def zeros(self, shape: Any, *, dtype: Any = None, device: Any = None) -> Array: ...
    def zeros_like(self, x: Array, /, *, dtype: Any = None, device: Any = None) -> Array: ...

    # Element-wise functions
    def abs(self, x: Array) -> Array: ...
    def asin(self, x: Array, /) -> Array: ...
    def atan2(self, x1: Array, x2: Array, /) -> Array: ...
    def cos(self, x: Array) -> Array: ...
    def exp(self, x: Array, /) -> Array: ...
    def floor(self, x: Array, /) -> Array: ...
    def isfinite(self, x: Array, /) -> Array: ...
    def isnan(self, x: Array, /) -> Array: ...
    def log(self, x: Array, /) -> Array: ...
    def log10(self, x: Array, /) -> Array: ...
    def max(self, x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array: ...
    def mean(self, x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array: ...
    def min(self, x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array: ...
    def sign(self, x: Array) -> Array: ...
    def sin(self, x: Array) -> Array: ...
    def sqrt(self, x: Array) -> Array: ...
    def sum(self, x: Array, *, axis: Any = None, keepdims: bool = False) -> Array: ...

    # Indexing functions
    def take(self, x: Array, indices: Array, /, *, axis: int | None = None) -> Array: ...

    # Linear algebra functions (part of main API)
    def matmul(self, x1: Array, x2: Array) -> Array: ...
    def tensordot(self, x1: Array, x2: Array, *, axes: Any = 2) -> Array: ...

    # Manipulation functions
    def broadcast_to(self, x: Array, /, shape: tuple[int, ...]) -> Array: ...
    def reshape(self, x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Array: ...
    def stack(self, arrays: Any, *, axis: int = 0) -> Array: ...

    # Searching functions
    def argmax(self, x: Array, /, *, axis: int | None = None, keepdims: bool = False) -> Array: ...
    def where(self, condition: Array, x1: Array, x2: Array, /) -> Array: ...

    # Utility functions
    def all(self, x: Array, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array: ...


@runtime_checkable
class _ArrayNamespaceWithLinAlg(_ArrayNamespace, Protocol):
    """Extended _ArrayNamespace protocol that includes the linear algebra extension.

    Use this when you know the array namespace supports the linalg extension.
    Most code should use the base _ArrayNamespace and check with hasattr(xp, 'linalg').
    """

    linalg: LinAlg


@runtime_checkable
class _ArrayNamespaceWithFFT(_ArrayNamespace, Protocol):
    """Extended _ArrayNamespace protocol that includes the FFT extension.

    Use this when you know the array namespace supports the fft extension.
    Most code should use the base _ArrayNamespace and check with hasattr(xp, 'fft').
    """

    fft: FFT


@runtime_checkable
class _ArrayNamespaceWithLinAlgAndFFT(_ArrayNamespace, Protocol):
    """Extended _ArrayNamespace protocol that includes both linalg and fft extensions.

    Use this when you know the array namespace supports both extensions.
    Most code should use the base _ArrayNamespace and check with hasattr(xp, 'linalg')
    and hasattr(xp, 'fft') as needed.
    """

    linalg: LinAlg
    fft: FFT


@runtime_checkable
class Array(Protocol):
    """Protocol for arrays that conform to the Array API standard.

    This is a lightweight implementation that covers the basic operations
    needed by the FastSIMUS codebase. It will eventually be replaced by one of the
    following:
    https://github.com/magnusdk/spekk/commit/d17d5bbd3e2beac97142a9397ce25942b787a7ed
    https://github.com/data-apis/array-api/pull/589/
    https://github.com/data-apis/array-api-typing

    Note:
        https://data-apis.org/array-api/latest/API_specification/index.html
    """

    dtype: Any
    ndim: int
    shape: tuple[int, ...]
    size: int

    # Basic operations used in the codebase
    def __add__(self, other: int | float | complex | Self) -> Self: ...
    def __sub__(self, other: int | float | complex | Self) -> Self: ...
    def __mul__(self, other: int | float | complex | Self) -> Self: ...
    def __truediv__(self, other: int | float | complex | Self) -> Self: ...
    def __pow__(self, other: int | float | complex | Self) -> Self: ...
    def __matmul__(self, other: Self) -> Self: ...
    def __and__(self, other: Self) -> Self: ...
    def __or__(self, other: Self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __lt__(self, other: int | float | complex | Self) -> Self: ...
    def __gt__(self, other: int | float | complex | Self) -> Self: ...
    def __le__(self, other: int | float | complex | Self) -> Self: ...
    def __ge__(self, other: int | float | complex | Self) -> Self: ...
    def __eq__(self, other: int | float | complex | Self) -> Self: ...  # type: ignore[invalid-method-override]
    def __ne__(self, other: int | float | complex | Self) -> Self: ...  # type: ignore[invalid-method-override]
    def __getitem__(
        self, key: int | slice | ... | None | tuple[int | slice | ... | Self | None, ...] | Self, /
    ) -> Self: ...

    # Reflected operations for scalar op array
    def __radd__(self, other: int | float | complex | Self) -> Self: ...
    def __rsub__(self, other: int | float | complex | Self) -> Self: ...
    def __rmul__(self, other: int | float | complex | Self) -> Self: ...
    def __rtruediv__(self, other: int | float | complex | Self) -> Self: ...
    def __rpow__(self, other: int | float | complex | Self) -> Self: ...

    # Only defined for zero-dimensional arrays
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __bool__(self) -> bool: ...


ArrayOrScalar = Array | int | float | complex | bool


def array_namespace(
    *arrays: Any,
) -> _ArrayNamespace | _ArrayNamespaceWithLinAlg | _ArrayNamespaceWithFFT | _ArrayNamespaceWithLinAlgAndFFT:
    """Typed wrapper around array_api_compat.array_namespace.

    Returns the array namespace for the given arrays with proper type hints.
    This resolves static typing issues by providing an ArrayNamespace protocol.

    Args:
        *arrays: Arrays to get the namespace for

    Returns:
        The appropriate array namespace (numpy, cupy, jax.numpy, etc.).
        May include optional extensions (linalg, fft, or both).

    Note:
        Optional extensions may not be available in all array libraries.
        Code should check for their existence using hasattr() before use:
        - hasattr(xp, 'linalg') for linear algebra extension
        - hasattr(xp, 'fft') for FFT extension

    Examples:
        >>> xp = array_namespace(arr)
        >>> if hasattr(xp, 'linalg'):
        ...     norm = xp.linalg.vector_norm(arr)
        ... else:
        ...     # Fallback implementation
        ...     norm = xp.sqrt(xp.sum(arr * arr, axis=-1))
        >>>
        >>> if hasattr(xp, 'fft'):
        ...     spectrum = xp.fft.rfft(signal)
        ... else:
        ...     raise RuntimeError("FFT extension not available")
    """
    xp = xpc_array_namespace(*arrays)
    if getattr(xp, "__name__", "").startswith("mlx"):
        from fast_simus.backends.mlx import ensure_compat  # noqa: PLC0415

        ensure_compat(xp)
    return cast(
        _ArrayNamespace | _ArrayNamespaceWithLinAlg | _ArrayNamespaceWithFFT | _ArrayNamespaceWithLinAlgAndFFT,
        xp,
    )
