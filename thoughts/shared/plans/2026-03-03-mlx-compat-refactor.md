# MLX Backend Compatibility -- Approach C Refactor

## Overview

Refactor the MLX compatibility patches from inline code in `_array_api.py` into a dedicated `src/fast_simus/backends/mlx.py` module. This separates concerns (protocols vs backend compat), fixes re-wrapping bugs in the current stashed implementation, and makes the MLX shim easy to remove when `array_api_compat` gains native MLX support (tracking: https://github.com/data-apis/array-api-compat/issues/162).

## Current State

**Branch:** `feature/mlx` (2 commits ahead of `origin/main`)

**Already committed:**
- `pyproject.toml`: `mlx>=0.31` added to test deps (Darwin/Linux gated)
- `pfield.py`: `freq_mask` removed from `_FrequencyPlan`, boolean indexing replaced with slice indexing (MLX compat)
- `test_pfield_helpers.py`: `freq_mask` assertion removed

**Stashed (uncommitted WIP):**
- `_array_api.py`: ~100 lines of MLX patching code mixed into the protocol/namespace file
- `tests/backend/test_mlx.py`: MLX compile test

**Problems with stashed implementation:**
1. `asarray` gets re-wrapped on every `array_namespace()` call (no idempotency guard)
2. Top-level imports of `array_api_compat.common._helpers` and `array_api_extra._lib._utils._compat` execute even when MLX is never used
3. Protocols and MLX compat are mixed in one 366-line file
4. `_ArrayNamespace` protocol was trimmed (removed `complex128`, `empty`, `empty_like`, `acos`, `vecdot`, `nonzero`) and `Array` protocol was trimmed (`__pos__`, `__index__`) -- these removals need to be preserved

## Desired End State

- `src/fast_simus/utils/_array_api.py` (~240 lines): protocols + `array_namespace()` with a 3-line MLX check
- `src/fast_simus/backends/__init__.py`: empty
- `src/fast_simus/backends/mlx.py` (~80 lines): all MLX patching, lazy imports, idempotent with sentinels
- `tests/backend/test_mlx.py` (~45 lines): `mx.compile` pfield test
- All 76+ tests pass (`poe test`)
- Lint clean (`poe lint`)

### Verification:
```bash
poe test      # all tests pass
poe lint      # no lint/type errors
```

## What We're NOT Doing

- Adding MLX to the parametrized `xp` fixture in `conftest.py` (future work)
- Benchmarking MLX vs NumPy/JAX (future work)
- Supporting MLX boolean indexing (MLX limitation, tracked separately)
- Adding MLX to `array_api_compat` upstream (external dependency)

## Implementation Approach

Extract-and-improve: take the stashed `_array_api.py` MLX code, move it to `backends/mlx.py` with bug fixes, then slim `_array_api.py` back down. The test file from the stash is reused as-is.

---

## Phase 1: Create `backends/mlx.py`

### Overview
Create the dedicated MLX compatibility module with all patching logic, lazy imports, and sentinel-based idempotency.

### Changes Required:

#### 1. New file: `src/fast_simus/backends/__init__.py`
Empty package init.

#### 2. New file: `src/fast_simus/backends/mlx.py`

All MLX compatibility code in one place. Key improvements over stashed version:
- All `array_api_compat` / `array_api_extra` private-API imports are **lazy** (inside function body)
- `asarray` wrapping uses a `_fastsimus_wrapped` sentinel to prevent stacking
- `device()` patching uses a `_fastsimus_mlx` sentinel to prevent stacking
- `try/except ImportError` on `array_api_extra` internals for graceful degradation

```python
"""Array API compatibility shim for MLX.

Temporary until array_api_compat gains native MLX support.
Tracking: https://github.com/data-apis/array-api-compat/issues/162
"""

from __future__ import annotations

from typing import Any

_MLX_ARRAY_API_ALIASES: dict[str, str] = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan2": "arctan2",
    "bool": "bool_",
}

_MLX_ISDTYPE_KIND_MAP: dict[str, str] = {
    "bool": "bool_",
    "signed integer": "signedinteger",
    "unsigned integer": "unsignedinteger",
    "integral": "integer",
    "real floating": "floating",
    "complex floating": "complexfloating",
    "numeric": "number",
}


def _make_isdtype(xp: Any) -> Any:
    def isdtype(dtype: Any, kind: Any) -> bool:
        if isinstance(kind, str):
            category = _MLX_ISDTYPE_KIND_MAP.get(kind)
            if category is None:
                msg = f"Unrecognized dtype kind: {kind!r}"
                raise ValueError(msg)
            return bool(xp.issubdtype(dtype, getattr(xp, category)))
        if isinstance(kind, tuple):
            return any(isdtype(dtype, k) for k in kind)
        return dtype == kind
    return isdtype


def _patch_namespace(xp: Any) -> None:
    """Add Array API aliases to mlx.core (idempotent)."""
    for standard_name, mlx_name in _MLX_ARRAY_API_ALIASES.items():
        if not hasattr(xp, standard_name) and hasattr(xp, mlx_name):
            setattr(xp, standard_name, getattr(xp, mlx_name))

    if not hasattr(xp, "isdtype") and hasattr(xp, "issubdtype"):
        xp.isdtype = _make_isdtype(xp)

    if not hasattr(xp, "astype"):
        def _astype(x: Any, dtype: Any, /, *, copy: bool = False) -> Any:
            return x.astype(dtype)
        xp.astype = _astype

    if not getattr(xp.asarray, "_fastsimus_wrapped", False):
        _original = xp.asarray
        def _asarray(a: Any, *, dtype: Any = None, **_kwargs: Any) -> Any:
            if dtype is not None:
                return _original(a, dtype=dtype)
            return _original(a)
        _asarray._fastsimus_wrapped = True
        xp.asarray = _asarray


def _patch_device(xp: Any) -> None:
    """Patch array_api_compat device() for MLX unified memory."""
    import array_api_compat
    import array_api_compat.common._helpers as _helpers

    _original = _helpers.device
    if getattr(_original, "_fastsimus_mlx", False):
        return

    def _device_with_mlx(x: Any, /) -> Any:
        if type(x).__module__.startswith("mlx"):
            return xp.default_device()
        return _original(x)
    _device_with_mlx._fastsimus_mlx = True

    _helpers.device = _device_with_mlx
    array_api_compat.device = _device_with_mlx

    try:
        import array_api_extra._lib._utils._compat as _xpx_compat
        _xpx_compat.device = _device_with_mlx
    except ImportError:
        pass


def ensure_compat(xp: Any) -> None:
    """Apply all MLX compatibility patches (idempotent)."""
    _patch_namespace(xp)
    _patch_device(xp)
```

### Success Criteria:

#### Automated:
- [x] `ruff check src/fast_simus/backends/mlx.py` passes
- [x] `ty check` passes (may need `# ty: ignore` on sentinel assignments)
- [x] Module is importable: `python -c "from fast_simus.backends.mlx import ensure_compat"`

---

## Phase 2: Update `_array_api.py`

### Overview
Remove all MLX patching code. Trim protocols (same removals as stashed version). Add 3-line MLX detection in `array_namespace()`.

### Changes Required:

#### 1. `src/fast_simus/utils/_array_api.py`

**Remove from top-level imports:**
- `import array_api_compat`
- `import array_api_compat.common._helpers as _apc_helpers`
- `import array_api_extra._lib._utils._compat as _xpx_compat`

**Remove from `_ArrayNamespace` protocol:**
- `complex128` (MLX doesn't support it)
- `empty`, `empty_like` (not used in codebase)
- `acos` (not used in codebase; `arccos` alias handled by backend)
- `vecdot` (not used in codebase)
- `nonzero` (not used in codebase, breaks JAX JIT)

**Remove from `Array` protocol:**
- `__pos__` (MLX arrays lack it, not used)
- `__index__` (MLX arrays lack it, not used)

**Remove entirely:**
- `_MLX_COMPAT_REGISTERED` flag
- `_register_mlx_with_array_api_compat()`
- `_MLX_ARRAY_API_ALIASES` dict
- `_MLX_ISDTYPE_KIND_MAP` dict
- `_make_mlx_isdtype()`
- `_patch_mlx_namespace()`

**Modify `array_namespace()`:**
```python
def array_namespace(*arrays):
    xp = xpc_array_namespace(*arrays)
    if getattr(xp, "__name__", "").startswith("mlx"):
        from fast_simus.backends.mlx import ensure_compat  # noqa: PLC0415
        ensure_compat(xp)
    return cast(..., xp)
```

### Success Criteria:

#### Automated:
- [x] `_array_api.py` is ~240 lines (down from 366 stashed) - now 268 lines
- [x] `ruff check src/fast_simus/utils/_array_api.py` passes
- [x] `ty check` passes
- [x] No top-level imports of `array_api_compat.common._helpers` or `array_api_extra._lib._utils._compat`

---

## Phase 3: Add MLX test and verify

### Overview
Create `tests/backend/test_mlx.py` (same as stashed version) and run the full test suite.

### Changes Required:

#### 1. `tests/backend/test_mlx.py`

Reuse the stashed test file as-is. It tests `pfield_precompute` + `mx.compile(pfield_compute)` end-to-end.

### Success Criteria:

#### Automated:
- [x] `poe test` -- all tests pass (76+)
- [x] `poe lint` -- clean
- [x] MLX test specifically: `pytest tests/backend/test_mlx.py -v` passes

#### Manual:
- [ ] Review that `_array_api.py` contains zero MLX-specific code
- [ ] Review that `backends/mlx.py` has no top-level `array_api_compat` imports

---

## Phase 4: Commit

### Overview
Stage all changes and commit with a descriptive message.

### Changes Required:

Single commit covering all files:
- `src/fast_simus/backends/__init__.py` (new)
- `src/fast_simus/backends/mlx.py` (new)
- `src/fast_simus/utils/_array_api.py` (modified)
- `tests/backend/test_mlx.py` (new)

Commit message:
```
feat: add MLX backend support with dedicated compat module

Extract MLX Array API patches into src/fast_simus/backends/mlx.py
for clean separation from protocol definitions. Fixes asarray
re-wrapping bug and makes array_api_compat imports lazy.
```

### Success Criteria:

#### Automated:
- [x] `poe test` passes post-commit
- [x] `poe lint` passes post-commit

---

## Testing Strategy

### Existing tests (must not regress):
- `tests/test_pfield.py` -- NumPy pfield correctness vs PyMUST
- `tests/test_pfield_helpers.py` -- array-api-strict unit tests
- `tests/backend/test_jax.py` -- JAX JIT compilation
- All other tests in `tests/`

### New test:
- `tests/backend/test_mlx.py` -- `mx.compile` + full pfield pipeline

### What's covered by the test:
- MLX arrays flow through `array_namespace()` and trigger `ensure_compat()`
- `pfield_precompute` works with MLX arrays (frequencies, spectra, slice indexing)
- `pfield_compute` works under `mx.compile` (complex arithmetic, sinc, exp, sum)
- Output shape and non-negativity assertions

## File Summary

| File | Action | Lines (approx) |
|------|--------|----------------|
| `src/fast_simus/backends/__init__.py` | Create | 1 |
| `src/fast_simus/backends/mlx.py` | Create | ~80 |
| `src/fast_simus/utils/_array_api.py` | Modify (slim down) | ~240 |
| `tests/backend/test_mlx.py` | Create | ~45 |

## References

- Previous session: [MLX compat WIP](a5f7daad-ba21-42bf-8930-6933d5f7bef3)
- `array_api_compat` MLX tracking: https://github.com/data-apis/array-api-compat/issues/162
- MLX docs: https://ml-explore.github.io/mlx/build/html/
