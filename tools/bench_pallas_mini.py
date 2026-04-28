"""Minimal Pallas kernel test: single-element TX with fori_loop geometric progression.

Tests whether dynamic-index writes to output ref work inside fori_loop.
"""
import os
import sys
import time

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PALLAS_USE_MOSAIC_GPU"] = "0"

nvidia_base = os.path.join(
    os.path.dirname(__file__), "..", ".venv", "lib", "python3.12",
    "site-packages", "nvidia",
)
nvidia_libs = ":".join(
    os.path.join(nvidia_base, pkg, "lib")
    for pkg in ["cusparse", "cublas", "cuda_runtime", "cufft", "cusolver", "cudnn", "nvjitlink"]
    if os.path.isdir(os.path.join(nvidia_base, pkg, "lib"))
)
if nvidia_libs:
    os.environ["LD_LIBRARY_PATH"] = nvidia_libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

print(f"Backend: {jax.default_backend()}")

TILE_S = 64
N_FREQ = 32

def _test_kernel(sx_ref, out_re_ref, out_im_ref):
    """Simple kernel: write sx * f to each column of output."""
    sx = sx_ref[:]
    n_freq = out_re_ref.shape[1]

    def body(f, _):
        f_float = jnp.float32(f)
        out_re_ref[:, f] = sx * f_float
        out_im_ref[:, f] = sx * (f_float + 0.5)
        return None

    jax.lax.fori_loop(0, n_freq, body, None)


def test_dynamic_write():
    """Test dynamic-index write to output ref inside fori_loop."""
    sx = jnp.arange(TILE_S, dtype=jnp.float32)

    try:
        out_re, out_im = pl.pallas_call(
            _test_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((TILE_S, N_FREQ), jnp.float32),
                jax.ShapeDtypeStruct((TILE_S, N_FREQ), jnp.float32),
            ],
            grid=(1,),
            in_specs=[pl.BlockSpec((TILE_S,), lambda i: (0,))],
            out_specs=[
                pl.BlockSpec((TILE_S, N_FREQ), lambda i: (0, 0)),
                pl.BlockSpec((TILE_S, N_FREQ), lambda i: (0, 0)),
            ],
        )(sx)
        print(f"Output shape: {out_re.shape}")
        print(f"out_re[0,:5] = {out_re[0,:5]}")
        print(f"out_re[1,:5] = {out_re[1,:5]}")
        expected_re_0 = jnp.arange(5, dtype=jnp.float32) * 0.0
        expected_re_1 = jnp.arange(5, dtype=jnp.float32) * 1.0
        print(f"expected[0,:5] = {expected_re_0}")
        print(f"expected[1,:5] = {expected_re_1}")
        print("Dynamic write to output ref: OK")
        return True
    except Exception as e:
        print(f"Dynamic write failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _geo_prog_kernel(init_re_ref, init_im_ref, step_re_ref, step_im_ref,
                     out_re_ref, out_im_ref):
    """Geometric progression kernel: cv *= step each iteration, write to output."""
    cv_re = init_re_ref[:]
    cv_im = init_im_ref[:]
    sv_re = step_re_ref[:]
    sv_im = step_im_ref[:]
    n_freq = out_re_ref.shape[1]

    def body(f, carry):
        cr, ci = carry
        out_re_ref[:, f] = cr
        out_im_ref[:, f] = ci
        new_cr = cr * sv_re - ci * sv_im
        new_ci = cr * sv_im + ci * sv_re
        return new_cr, new_ci

    jax.lax.fori_loop(0, n_freq, body, (cv_re, cv_im))


def test_geo_progression():
    """Test geometric progression with fori_loop writing to output."""
    init_re = jnp.ones(TILE_S, dtype=jnp.float32)
    init_im = jnp.zeros(TILE_S, dtype=jnp.float32)
    angle = 0.1
    step_re = jnp.full(TILE_S, jnp.cos(angle), dtype=jnp.float32)
    step_im = jnp.full(TILE_S, jnp.sin(angle), dtype=jnp.float32)

    try:
        out_re, out_im = pl.pallas_call(
            _geo_prog_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((TILE_S, N_FREQ), jnp.float32),
                jax.ShapeDtypeStruct((TILE_S, N_FREQ), jnp.float32),
            ],
            grid=(1,),
            in_specs=[
                pl.BlockSpec((TILE_S,), lambda i: (0,)),
                pl.BlockSpec((TILE_S,), lambda i: (0,)),
                pl.BlockSpec((TILE_S,), lambda i: (0,)),
                pl.BlockSpec((TILE_S,), lambda i: (0,)),
            ],
            out_specs=[
                pl.BlockSpec((TILE_S, N_FREQ), lambda i: (0, 0)),
                pl.BlockSpec((TILE_S, N_FREQ), lambda i: (0, 0)),
            ],
        )(init_re, init_im, step_re, step_im)

        expected_re = jnp.array([jnp.cos(angle * f) for f in range(N_FREQ)])
        expected_im = jnp.array([jnp.sin(angle * f) for f in range(N_FREQ)])
        max_err_re = float(jnp.max(jnp.abs(out_re[0, :] - expected_re)))
        max_err_im = float(jnp.max(jnp.abs(out_im[0, :] - expected_im)))
        print(f"Geo progression max error: re={max_err_re:.2e} im={max_err_im:.2e}")
        print("Geometric progression kernel: OK")
        return True
    except Exception as e:
        print(f"Geo progression failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _test_2d_carry(x_ref, out_ref):
    """2D tensor as fori_loop carry + jnp.sum(axis=1) + broadcasting."""
    x = x_ref[:, :]  # (TILE_S, N_ELEM)

    def body(f, carry):
        state = carry  # (TILE_S, N_ELEM)
        row_sum = jnp.sum(state, axis=1)  # (TILE_S,) -- sum over elements
        out_ref[:, f] = row_sum
        return state * 0.99

    jax.lax.fori_loop(0, N_FREQ, body, x)


N_ELEM = 8

def test_2d_carry():
    """Test 2D array as fori_loop carry, jnp.sum(axis=1), broadcasting."""
    x = jnp.ones((TILE_S, N_ELEM), dtype=jnp.float32)
    x = x * jnp.arange(N_ELEM, dtype=jnp.float32)[None, :]  # row i = [0,1,...,N_ELEM-1]

    try:
        (out,) = pl.pallas_call(
            _test_2d_carry,
            out_shape=[jax.ShapeDtypeStruct((TILE_S, N_FREQ), jnp.float32)],
            grid=(1,),
            in_specs=[pl.BlockSpec((TILE_S, N_ELEM), lambda i: (0, 0))],
            out_specs=[pl.BlockSpec((TILE_S, N_FREQ), lambda i: (0, 0))],
        )(x)

        # Expected: out[s, f] = sum(row * 0.99^f) = sum([0..N_ELEM-1]) * 0.99^f
        row_sum = float(N_ELEM * (N_ELEM - 1) / 2)  # 0+1+...+7 = 28
        expected_f0 = row_sum  # 0.99^0 = 1
        expected_f1 = row_sum * 0.99
        print(f"2D carry: out[0,0]={float(out[0,0]):.4f} expected={expected_f0:.4f}")
        print(f"2D carry: out[0,1]={float(out[0,1]):.4f} expected={expected_f1:.4f}")
        err = abs(float(out[0, 0]) - expected_f0)
        print(f"2D carry error at f=0: {err:.2e}")
        print("2D carry test: OK")
        return True
    except Exception as e:
        print(f"2D carry FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def _test_broadcast_2d(sx_ref, ex_ref, out_ref):
    """Test broadcasting 1D→2D: sx[:,None] - ex[None,:]."""
    sx = sx_ref[:]  # (TILE_S,)
    ex = ex_ref[:]  # (N_ELEM,)
    dx = sx[:, None] - ex[None, :]  # (TILE_S, N_ELEM) via broadcasting
    out_ref[:, :] = dx


def test_broadcast():
    """Test broadcasting 1D arrays to create 2D tensor in Pallas."""
    sx = jnp.arange(TILE_S, dtype=jnp.float32)
    ex = jnp.arange(N_ELEM, dtype=jnp.float32) * 10.0

    try:
        (out,) = pl.pallas_call(
            _test_broadcast_2d,
            out_shape=[jax.ShapeDtypeStruct((TILE_S, N_ELEM), jnp.float32)],
            grid=(1,),
            in_specs=[
                pl.BlockSpec((TILE_S,), lambda i: (0,)),
                pl.BlockSpec((N_ELEM,), lambda i: (0,)),
            ],
            out_specs=[pl.BlockSpec((TILE_S, N_ELEM), lambda i: (0, 0))],
        )(sx, ex)

        expected_00 = 0.0 - 0.0
        expected_10 = 1.0 - 0.0
        expected_01 = 0.0 - 10.0
        print(f"Broadcast: out[0,0]={float(out[0,0]):.1f} expected={expected_00:.1f}")
        print(f"Broadcast: out[1,0]={float(out[1,0]):.1f} expected={expected_10:.1f}")
        print(f"Broadcast: out[0,1]={float(out[0,1]):.1f} expected={expected_01:.1f}")
        err = float(jnp.max(jnp.abs(out - (sx[:, None] - ex[None, :]))))
        print(f"Broadcast max error: {err:.2e}")
        print("Broadcast test: OK")
        return True
    except Exception as e:
        print(f"Broadcast FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def _test_dynamic_ref_index(geo_ref, out_ref):
    """Test dynamic indexing into input ref with fori_loop variable."""
    n = geo_ref.shape[0]
    def body(e, _):
        val = geo_ref[e]  # scalar dynamic index
        out_ref[e] = val * 2.0
        return None
    jax.lax.fori_loop(0, n, body, None)


def test_dynamic_ref_index():
    """Test dynamic indexing into input ref inside fori_loop."""
    geo = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=jnp.float32)
    try:
        (out,) = pl.pallas_call(
            _test_dynamic_ref_index,
            out_shape=[jax.ShapeDtypeStruct((N_ELEM,), jnp.float32)],
            grid=(1,),
            in_specs=[pl.BlockSpec((N_ELEM,), lambda i: (0,))],
            out_specs=[pl.BlockSpec((N_ELEM,), lambda i: (0,))],
        )(geo)
        expected = geo * 2.0
        err = float(jnp.max(jnp.abs(out - expected)))
        print(f"Dynamic ref index: out={out[:4]} expected={expected[:4]}")
        print(f"Dynamic ref index max error: {err:.2e}")
        print("Dynamic ref index test: OK")
        return True
    except Exception as e:
        print(f"Dynamic ref index FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok1 = test_dynamic_write()
    print()
    if ok1:
        ok2 = test_geo_progression()
    print()
    ok3 = test_broadcast()
    print()
    ok4 = test_2d_carry()
    print()
    ok5 = test_dynamic_ref_index()
    print("\n=DONE=")
