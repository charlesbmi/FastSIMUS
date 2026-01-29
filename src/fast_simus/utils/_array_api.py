"""Array-API utilities."""

from typing import Any, Protocol, cast, runtime_checkable

from array_api_compat import array_namespace as xpc_array_namespace


@runtime_checkable
class LinAlg(Protocol):
    """Protocol for linear algebra extension conforming to Array API standard.

    This is an optional extension - not all array libraries implement it.
    See: https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html
    """

    def vector_norm(self, x: "Array", *, axis: Any = None, keepdims: bool = False, ord: Any = None) -> "Array": ...


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

    def rfft(self, x: "Array", /, *, n: int | None = None, axis: int = -1, norm: str = "backward") -> "Array": ...
    def irfft(self, x: "Array", /, *, n: int | None = None, axis: int = -1, norm: str = "backward") -> "Array": ...
    def rfftfreq(self, n: int, /, *, d: float = 1.0, device: Any = None) -> "Array": ...
    def fftshift(self, x: "Array", /, *, axes: int | tuple[int, ...] | None = None) -> "Array": ...


@runtime_checkable
class _ArrayNamespace(Protocol):
    """Protocol for array namespaces that conform to the Array API standard.

    This covers the common operations and data types used throughout the FastSIMUS codebase.
    Based on the Array API specification: https://data-apis.org/array-api/latest/

    Note: This base protocol does NOT include optional extensions (linalg, fft).
    Use the extended protocols for namespaces that have these extensions.
    """

    # Data types
    float32: Any
    float64: Any
    complex64: Any
    complex128: Any
    int8: Any
    int16: Any
    int32: Any
    int64: Any
    uint8: Any
    uint16: Any
    uint32: Any
    uint64: Any

    # Core linear algebra functions (part of main API)
    def matmul(self, x1: "Array", x2: "Array") -> "Array": ...
    def tensordot(self, x1: "Array", x2: "Array", *, axes: Any = 2) -> "Array": ...
    def vecdot(self, x1: "Array", x2: "Array", *, axis: int = -1) -> "Array": ...

    # Mathematical functions
    def abs(self, x: "Array") -> "Array": ...
    def cos(self, x: "Array") -> "Array": ...
    def sign(self, x: "Array") -> "Array": ...
    def sin(self, x: "Array") -> "Array": ...
    def sqrt(self, x: "Array") -> "Array": ...
    def sum(self, x: "Array", *, axis: Any = None, keepdims: bool = False) -> "Array": ...

    # Array creation and manipulation
    def asarray(self, obj: Any, *, dtype: Any = None, device: Any = None, copy: bool = False) -> "Array": ...
    def stack(self, arrays: Any, *, axis: int = 0) -> "Array": ...
    def zeros(self, shape: Any, *, dtype: Any = None, device: Any = None) -> "Array": ...


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
    """

    dtype: Any
    shape: tuple[int, ...]

    # Basic operations used in the codebase
    def __add__(self, other: Any) -> "Array": ...
    def __sub__(self, other: Any) -> "Array": ...
    def __mul__(self, other: Any) -> "Array": ...
    def __truediv__(self, other: Any) -> "Array": ...
    def __neg__(self) -> "Array": ...
    def __pos__(self) -> "Array": ...
    def __lt__(self, other: Any) -> "Array": ...
    def __gt__(self, other: Any) -> "Array": ...
    def __getitem__(self, key: Any) -> "Array": ...


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
    return cast(
        _ArrayNamespace | _ArrayNamespaceWithLinAlg | _ArrayNamespaceWithFFT | _ArrayNamespaceWithLinAlgAndFFT,
        xpc_array_namespace(*arrays),
    )
