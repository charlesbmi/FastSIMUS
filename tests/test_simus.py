"""Tests for simus (RF signal simulation).

Reference tests compare FastSIMUS simus against PyMUST's simus output.
Tests are structured as invariants that must hold at every refactoring step.
"""

from typing import NamedTuple, cast

import array_api_strict
import numpy as np
import pymust
import pytest
from array_api_compat import is_jax_namespace

from fast_simus.medium_params import MediumParams
from fast_simus.simus import SimusResult, SimusStrategy, simus, simus_compute, simus_precompute
from fast_simus.transducer_presets import C5_2v, L11_5v, P4_2v
from fast_simus.utils._array_api import Array, _ArrayNamespace, is_mlx_namespace

xp = cast(_ArrayNamespace, array_api_strict)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SCATTERERS = 6


class SimusReferenceData(NamedTuple):
    """Typed reference data from PyMUST simus."""

    rf: np.ndarray
    spectrum: np.ndarray
    scatterers_x: np.ndarray
    scatterers_z: np.ndarray
    rc: np.ndarray
    delays: np.ndarray
    probe: str
    fs: float
    focus: tuple[float, float] | None = None


def _pymust_simus(
    probe_name: str,
    x: np.ndarray,
    z: np.ndarray,
    rc: np.ndarray,
    delays: np.ndarray,
    fs: float,
    db_thresh: float = -60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PyMUST simus and return (RF, spectrum)."""
    param = pymust.getparam(probe_name)
    param.fs = fs
    options = pymust.utils.Options()
    options.dBThresh = db_thresh
    rf, spectrum = pymust.simus(x, z, rc, delays, param, options)
    return rf, spectrum


# ---------------------------------------------------------------------------
# Reference data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("P4-2v-focused", id="P4-2v-focused"),
        pytest.param("L11-5v-plane", id="L11-5v-plane"),
    ]
)
def simus_reference(request: pytest.FixtureRequest) -> SimusReferenceData:
    """Parametrized reference data from PyMUST simus."""
    case = request.param
    if case == "P4-2v-focused":
        param = pymust.getparam("P4-2v")
        x0, z0 = 0.0, 0.03
        delays = pymust.txdelayFocused(param, x0, z0)
        x = np.zeros(N_SCATTERERS)
        z = np.linspace(1e-2, 8e-2, N_SCATTERERS)
        rc = np.ones(N_SCATTERERS)
        assert param.fc is not None
        fs = 4.0 * float(param.fc)
        rf, spectrum = _pymust_simus("P4-2v", x, z, rc, delays, fs)
        return SimusReferenceData(rf, spectrum, x, z, rc, delays, "P4-2v", fs, focus=(x0, z0))
    if case == "L11-5v-plane":
        param = pymust.getparam("L11-5v")
        tilt_rad = np.deg2rad(5.0)
        delays = pymust.txdelayPlane(param, tilt_rad)
        x = np.linspace(-1e-2, 1e-2, N_SCATTERERS)
        z = np.linspace(0.5e-2, 3e-2, N_SCATTERERS)
        rc = np.ones(N_SCATTERERS)
        assert param.fc is not None
        fs = 4.0 * float(param.fc)
        rf, spectrum = _pymust_simus("L11-5v", x, z, rc, delays, fs)
        return SimusReferenceData(rf, spectrum, x, z, rc, delays, "L11-5v", fs)
    raise ValueError(f"Unknown reference case: {case}")


def _preset_for_probe(probe: str):
    """Return preset callable for probe name."""
    return {"P4-2v": P4_2v, "L11-5v": L11_5v, "C5-2v": C5_2v}[probe]


# ---------------------------------------------------------------------------
# Phase 1: Reference sanity tests
# ---------------------------------------------------------------------------


class TestPyMUSTSimusReference:
    """Validate that PyMUST simus reference data is sane."""

    def test_rf_shape(self, simus_reference: SimusReferenceData):
        """RF output has expected shape (n_samples, n_elements)."""
        rf = simus_reference.rf
        assert rf.ndim == 2
        param = pymust.getparam(simus_reference.probe)
        assert rf.shape[1] == param.Nelements
        assert rf.shape[0] > 0

    def test_rf_not_all_zero(self, simus_reference: SimusReferenceData):
        """RF signals should not be all zeros."""
        assert np.max(np.abs(simus_reference.rf)) > 0

    def test_spectrum_shape(self, simus_reference: SimusReferenceData):
        """Spectrum has expected shape (n_freq, n_elements)."""
        spectrum = simus_reference.spectrum
        assert spectrum.ndim == 2
        param = pymust.getparam(simus_reference.probe)
        assert spectrum.shape[1] == param.Nelements


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

# Simus comparison uses peak-normalized absolute tolerance.
# PyMUST is float32 internally, so tolerance is limited by float32 precision.
# Additionally, the IFFT + smooth thresholding adds numerical differences.
_SIMUS_ATOL_PEAK = 5e-3  # -46 dB re peak; conservative for initial validation


def _compute_peak_normalized_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Compute peak-normalized max error and report in dB."""
    peak = max(np.max(np.abs(actual)), np.max(np.abs(expected)))
    if peak == 0:
        return 0.0, -np.inf
    max_error = np.max(np.abs(actual - expected))
    max_error_fraction = max_error / peak
    max_error_db = 20 * np.log10(max_error_fraction) if max_error_fraction > 0 else -np.inf
    return max_error_fraction, max_error_db


def _assert_simus_rf_close(
    actual: np.ndarray,
    expected: np.ndarray,
    atol_peak: float = _SIMUS_ATOL_PEAK,
    desc: str = "",
) -> None:
    """Assert RF signals match using peak-normalized metric."""
    # Match lengths (allow minor truncation differences)
    min_len = min(actual.shape[0], expected.shape[0])
    actual = actual[:min_len]
    expected = expected[:min_len]

    peak = max(np.max(np.abs(actual)), np.max(np.abs(expected)))
    if peak == 0:
        return

    max_error_fraction, max_error_db = _compute_peak_normalized_error(actual, expected)
    if desc:
        print(f"\n{desc}: RF max error = {max_error_db:.1f} dB ({max_error_fraction:.2e} of peak)")

    np.testing.assert_allclose(
        actual / peak,
        expected / peak,
        atol=atol_peak,
        rtol=0,
        err_msg=f"{desc}: RF max error {max_error_db:.1f} dB exceeds tolerance {20 * np.log10(atol_peak):.1f} dB",
    )


# ---------------------------------------------------------------------------
# FastSIMUS simus helper
# ---------------------------------------------------------------------------


def _fastsimus_simus(
    preset_fn,
    scatterers_x: np.ndarray,
    scatterers_z: np.ndarray,
    rc: np.ndarray,
    delays: np.ndarray,
    fs: float,
    db_thresh: float = -60.0,
) -> SimusResult:
    """Call FastSIMUS simus and return SimusResult."""
    params = preset_fn()
    scatterers = np.stack([scatterers_x, scatterers_z], axis=-1)
    scatterers_strict = xp.asarray(scatterers)
    rc_strict = xp.asarray(rc)
    delays_1d = xp.reshape(xp.asarray(delays), (-1,))
    return simus(scatterers_strict, rc_strict, delays_1d, params, fs=fs, db_thresh=db_thresh)


# ---------------------------------------------------------------------------
# Comparison tests: FastSIMUS simus vs PyMUST
# ---------------------------------------------------------------------------


class TestSimusMatchesPyMUST:
    """Compare FastSIMUS simus output against PyMUST reference."""

    def test_rf_matches_pymust(self, simus_reference: SimusReferenceData):
        """FastSIMUS simus RF signals must match PyMUST reference."""
        preset_fn = _preset_for_probe(simus_reference.probe)
        result = _fastsimus_simus(
            preset_fn,
            simus_reference.scatterers_x,
            simus_reference.scatterers_z,
            simus_reference.rc,
            simus_reference.delays,
            simus_reference.fs,
        )
        rf_np = np.asarray(result.rf)
        _assert_simus_rf_close(rf_np, simus_reference.rf, desc=simus_reference.probe)

    def test_rf_shape_matches(self, simus_reference: SimusReferenceData):
        """FastSIMUS RF output has same number of elements as PyMUST."""
        preset_fn = _preset_for_probe(simus_reference.probe)
        result = _fastsimus_simus(
            preset_fn,
            simus_reference.scatterers_x,
            simus_reference.scatterers_z,
            simus_reference.rc,
            simus_reference.delays,
            simus_reference.fs,
        )
        rf_np = np.asarray(result.rf)
        assert rf_np.shape[1] == simus_reference.rf.shape[1]


# ---------------------------------------------------------------------------
# Unit tests for simus API
# ---------------------------------------------------------------------------


class TestSimusAPI:
    """Tests for the simus public API."""

    def test_simus_returns_named_tuple(self):
        """Simus returns a SimusResult with rf and spectrum fields."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)
        result = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        assert isinstance(result, SimusResult)
        assert hasattr(result, "rf")
        assert hasattr(result, "spectrum")

    def test_rf_output_shape(self):
        """RF output has shape (n_samples, n_elements)."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)
        result = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        rf_np = np.asarray(result.rf)
        assert rf_np.ndim == 2
        assert rf_np.shape[1] == params.n_elements

    def test_spectrum_output_shape(self):
        """Spectrum output has shape (n_freq, n_elements)."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)
        result = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        spectrum_np = np.asarray(result.spectrum)
        assert spectrum_np.ndim == 2
        assert spectrum_np.shape[1] == params.n_elements

    def test_custom_sampling_frequency(self):
        """Custom fs produces more samples."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)

        result_default = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        result_high_fs = simus(
            xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params, fs=10 * params.freq_center
        )

        rf_default = np.asarray(result_default.rf)
        rf_high = np.asarray(result_high_fs.rf)
        assert rf_high.shape[0] > rf_default.shape[0]

    def test_precompute_compute_matches_simus(self):
        """simus_precompute + simus_compute must give identical output to simus."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)

        scatterers_strict = xp.asarray(scatterers)
        rc_strict = xp.asarray(rc)
        delays_strict = xp.asarray(delays)

        result_direct = simus(scatterers_strict, rc_strict, delays_strict, params)

        plan = simus_precompute(scatterers_strict, rc_strict, delays_strict, params)
        result_split = simus_compute(scatterers_strict, rc_strict, delays_strict, plan, params)

        np.testing.assert_array_equal(np.asarray(result_direct.rf), np.asarray(result_split.rf))
        np.testing.assert_array_equal(np.asarray(result_direct.spectrum), np.asarray(result_split.spectrum))


class TestSimusFrequencyCount:
    """Validate frequency counts under different db_thresh settings."""

    def test_default_db_thresh_frequency_count(self, simus_reference: SimusReferenceData):
        """Default db_thresh=-60 produces fewer frequencies than PyMUST's -100."""
        preset_fn = _preset_for_probe(simus_reference.probe)
        params = preset_fn()
        scatterers = np.stack([simus_reference.scatterers_x, simus_reference.scatterers_z], axis=-1)
        delays_1d = xp.reshape(xp.asarray(simus_reference.delays), (-1,))

        plan_60 = simus_precompute(
            xp.asarray(scatterers),
            xp.asarray(simus_reference.rc),
            delays_1d,
            params,
            fs=simus_reference.fs,
            db_thresh=-60.0,
        )
        plan_100 = simus_precompute(
            xp.asarray(scatterers),
            xp.asarray(simus_reference.rc),
            delays_1d,
            params,
            fs=simus_reference.fs,
            db_thresh=-100.0,
        )
        n_freq_60 = plan_60.selected_freqs.shape[0]
        n_freq_100 = plan_100.selected_freqs.shape[0]

        print(f"\n{simus_reference.probe}: n_freq at -60dB={n_freq_60}, -100dB={n_freq_100}")
        assert n_freq_60 < n_freq_100, "-60dB should select fewer frequencies than -100dB"
        assert n_freq_60 > 0

    @pytest.mark.parametrize(
        "probe_name,expected_n_freq_approx",
        [
            ("P4-2v", 812),
        ],
    )
    def test_p4_2v_frequency_count_at_80mm(self, probe_name: str, expected_n_freq_approx: int):
        """P4-2v at 80mm depth with db_thresh=-60 should produce ~812 frequencies."""
        preset_fn = _preset_for_probe(probe_name)
        params = preset_fn()
        n_scat = 6
        scatterers = np.stack([np.zeros(n_scat), np.linspace(1e-2, 8e-2, n_scat)], axis=-1)
        rc = np.ones(n_scat)
        param_pymust = pymust.getparam(probe_name)
        x0, z0 = 0.0, 0.03
        delays = pymust.txdelayFocused(param_pymust, x0, z0)
        assert param_pymust.fc is not None
        fs = 4.0 * float(param_pymust.fc)

        plan = simus_precompute(
            xp.asarray(scatterers),
            xp.asarray(rc),
            xp.reshape(xp.asarray(delays), (-1,)),
            params,
            fs=fs,
            db_thresh=-60.0,
        )
        n_freq = plan.selected_freqs.shape[0]
        print(f"\n{probe_name} at 80mm: n_freq={n_freq} (expected ~{expected_n_freq_approx})")
        assert abs(n_freq - expected_n_freq_approx) < 20, (
            f"Expected ~{expected_n_freq_approx} frequencies, got {n_freq}"
        )


class TestSimusEdgeCases:
    """Edge-case tests for simus."""

    def test_single_scatterer(self):
        """Single scatterer produces valid output."""
        params = P4_2v()
        scatterers = np.array([[0.0, 3e-2]])
        rc = np.ones(1)
        delays = np.zeros(params.n_elements)
        result = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        rf_np = np.asarray(result.rf)
        assert rf_np.shape[1] == params.n_elements
        assert np.max(np.abs(rf_np)) > 0

    def test_zero_rc_gives_zero_rf(self):
        """Zero reflection coefficients produce zero RF signals."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.zeros(3)
        delays = np.zeros(params.n_elements)
        result = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params)
        rf_np = np.asarray(result.rf)
        assert np.max(np.abs(rf_np)) < 1e-10

    def test_with_attenuation(self):
        """Simus with attenuation produces different (typically smaller) signals."""
        params = P4_2v()
        medium_no_atten = MediumParams(attenuation=0.0)
        medium_atten = MediumParams(attenuation=0.5)
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)

        result_no = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params, medium=medium_no_atten)
        result_yes = simus(xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params, medium=medium_atten)

        power_no = np.sum(np.asarray(result_no.rf) ** 2)
        power_yes = np.sum(np.asarray(result_yes.rf) ** 2)
        assert power_yes < power_no


# ---------------------------------------------------------------------------
# Cross-backend strategy tests
# ---------------------------------------------------------------------------


class TestSimusStrategyCrossBackend:
    """Test strategies across backends using the xp fixture."""

    def test_strategy_on_backend(self, xp, simus_strategy):
        """Each strategy produces valid output on each backend."""
        if simus_strategy == SimusStrategy.SCAN and not is_jax_namespace(xp):
            pytest.skip("scan requires JAX")
        if simus_strategy == SimusStrategy.METAL and not is_mlx_namespace(xp):
            pytest.skip("metal requires MLX")

        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)

        result = simus(
            xp.asarray(scatterers),
            xp.asarray(rc),
            xp.asarray(delays),
            params,
            strategy=simus_strategy,
        )
        rf_np = np.asarray(result.rf)
        assert rf_np.ndim == 2
        assert rf_np.shape[1] == params.n_elements
        assert np.max(np.abs(rf_np)) > 0


class TestSimusMetal:
    """Metal-specific tests: validate Metal kernel matches Python strategy."""

    @pytest.fixture(autouse=True)
    def _require_mlx(self):
        pytest.importorskip("mlx")

    def test_metal_matches_python(self):
        """Metal strategy must match Python strategy (peak-normalized)."""
        import mlx.core as _mx

        from fast_simus.backends.mlx import ensure_compat

        ensure_compat(_mx)

        params = P4_2v()
        scatterers_np = np.stack([np.zeros(N_SCATTERERS), np.linspace(1e-2, 5e-2, N_SCATTERERS)], axis=-1)
        rc_np = np.ones(N_SCATTERERS)
        delays_np = np.zeros(params.n_elements)

        result_python = simus(
            xp.asarray(scatterers_np), xp.asarray(rc_np), xp.asarray(delays_np), params, strategy=SimusStrategy.PYTHON
        )

        result_metal = simus(
            cast("Array", _mx.array(scatterers_np)),
            cast("Array", _mx.array(rc_np)),
            cast("Array", _mx.array(delays_np.astype(np.float32))),
            params,
            strategy=SimusStrategy.METAL,
        )

        rf_python = np.asarray(result_python.rf)
        rf_metal = np.asarray(result_metal.rf)

        _assert_simus_rf_close(
            rf_metal,
            rf_python,
            atol_peak=0.02,
            desc="Metal vs Python",
        )

    def test_metal_matches_pymust(self, simus_reference: SimusReferenceData):
        """Metal strategy must match PyMUST reference (peak-normalized)."""
        import mlx.core as _mx

        from fast_simus.backends.mlx import ensure_compat

        ensure_compat(_mx)

        preset_fn = _preset_for_probe(simus_reference.probe)
        params = preset_fn()
        scatterers = np.stack([simus_reference.scatterers_x, simus_reference.scatterers_z], axis=-1)

        result = simus(
            cast("Array", _mx.array(scatterers)),
            cast("Array", _mx.array(simus_reference.rc)),
            cast("Array", _mx.array(simus_reference.delays.astype(np.float32).ravel())),
            params,
            fs=simus_reference.fs,
            strategy=SimusStrategy.METAL,
        )

        rf_metal = np.asarray(result.rf)
        _assert_simus_rf_close(
            rf_metal,
            simus_reference.rf,
            atol_peak=0.02,
            desc=f"Metal vs PyMUST ({simus_reference.probe})",
        )

    def test_metal_chunked_matches_python(self):
        """Chunked Metal (>chunk_size scatterers) must match Python strategy."""
        import mlx.core as _mx

        from fast_simus.backends.mlx import ensure_compat

        ensure_compat(_mx)

        n_scat = 10_001
        params = P4_2v()
        np.random.seed(42)
        scatterers_np = np.stack(
            [np.random.uniform(-2e-2, 2e-2, n_scat), np.random.uniform(1e-3, 5e-2, n_scat)],
            axis=-1,
        ).astype(np.float32)
        rc_np = np.random.uniform(0.5, 1.5, n_scat).astype(np.float32)
        delays_np = np.zeros(params.n_elements, dtype=np.float32)

        result_python = simus(
            xp.asarray(scatterers_np), xp.asarray(rc_np), xp.asarray(delays_np), params, strategy=SimusStrategy.PYTHON
        )
        result_metal = simus(
            cast("Array", _mx.array(scatterers_np)),
            cast("Array", _mx.array(rc_np)),
            cast("Array", _mx.array(delays_np)),
            params,
            strategy=SimusStrategy.METAL,
        )

        rf_python = np.asarray(result_python.rf)
        rf_metal = np.asarray(result_metal.rf)

        _assert_simus_rf_close(
            rf_metal,
            rf_python,
            atol_peak=0.02,
            desc="Metal chunked (10K+1, odd SCAT_REDUCE tail) vs Python",
        )

    def test_metal_auto_selected_for_mlx(self):
        """Auto strategy selects METAL when arrays are MLX."""
        import mlx.core as _mx

        from fast_simus.backends.mlx import ensure_compat
        from fast_simus.simus import _select_simus_strategy

        ensure_compat(_mx)

        params = P4_2v()
        strategy = _select_simus_strategy(cast(_ArrayNamespace, _mx), params, False)
        assert strategy == SimusStrategy.METAL


class TestSimusStrategy:
    """Tests for SimusStrategy enum and dispatch."""

    def test_explicit_python_strategy(self):
        """Explicit PYTHON strategy produces valid output."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)
        result = simus(
            xp.asarray(scatterers), xp.asarray(rc), xp.asarray(delays), params, strategy=SimusStrategy.PYTHON
        )
        assert np.max(np.abs(np.asarray(result.rf))) > 0

    def test_simus_compute_accepts_strategy(self):
        """simus_compute accepts strategy kwarg."""
        params = P4_2v()
        scatterers = np.stack([np.zeros(3), np.linspace(1e-2, 5e-2, 3)], axis=-1)
        rc = np.ones(3)
        delays = np.zeros(params.n_elements)
        scatterers_strict = xp.asarray(scatterers)
        rc_strict = xp.asarray(rc)
        delays_strict = xp.asarray(delays)

        plan = simus_precompute(scatterers_strict, rc_strict, delays_strict, params)
        result = simus_compute(scatterers_strict, rc_strict, delays_strict, plan, params, strategy=SimusStrategy.PYTHON)
        assert np.max(np.abs(np.asarray(result.rf))) > 0
