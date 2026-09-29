"""FastSIMUS - Fast Simulator for Medical Ultrasound based on SIMUS/MUST."""

from fast_simus.aperture import RectangularAperture, matrix_aperture, transform_aperture
from fast_simus.execution import ExecutionOptions
from fast_simus.field_blocks import FieldBlock, iter_pfield_spectrum, iter_wavefield
from fast_simus.jit import jit
from fast_simus.lens import ElevationLens
from fast_simus.medium_params import MediumParams
from fast_simus.pfield import (
    PfieldPlan,
    PfieldSpectrumInfo,
    PfieldStrategy,
    pfield,
    pfield_compute,
    pfield_precompute,
    pfield_spectrum,
    pfield_spectrum_compute,
    rms_from_spectrum,
)
from fast_simus.plans import EchoPlan, FieldPlan, FieldSpectrumInfo
from fast_simus.scattered import (
    ScatteredFieldPlan,
    iter_scattered_pfield_spectrum,
    scattered_field_precompute,
    scattered_pfield_spectrum,
)
from fast_simus.sequence import (
    SequenceEvent,
    SequencePlan,
    SequenceResult,
    TransmitSequence,
    iter_simus_sequence,
    sequence_precompute,
    simus_sequence,
)
from fast_simus.simus import SimusPlan, SimusResult, SimusStrategy, simus, simus_compute, simus_precompute
from fast_simus.transducer import Transducer, transducer_from_params
from fast_simus.transducer_params import BaffleType, TransducerParams
from fast_simus.tx_delay import (
    diverging_wave,
    focus_delays,
    focused,
    plane_wave,
    plane_wave_delays,
)
from fast_simus.utils._array_api import default_namespace
from fast_simus.utils.geometry import element_positions
from fast_simus.wavefield import WavefieldResult, spectrum_to_wavefield, wavefield, wavefield_times

__all__ = [
    "BaffleType",
    "EchoPlan",
    "ElevationLens",
    "ExecutionOptions",
    "FieldBlock",
    "FieldPlan",
    "FieldSpectrumInfo",
    "MediumParams",
    "PfieldPlan",
    "PfieldSpectrumInfo",
    "PfieldStrategy",
    "RectangularAperture",
    "ScatteredFieldPlan",
    "SequenceEvent",
    "SequencePlan",
    "SequenceResult",
    "SimusPlan",
    "SimusResult",
    "SimusStrategy",
    "Transducer",
    "TransducerParams",
    "TransmitSequence",
    "WavefieldResult",
    "default_namespace",
    "diverging_wave",
    "element_positions",
    "focus_delays",
    "focused",
    "iter_pfield_spectrum",
    "iter_scattered_pfield_spectrum",
    "iter_simus_sequence",
    "iter_wavefield",
    "jit",
    "matrix_aperture",
    "pfield",
    "pfield_compute",
    "pfield_precompute",
    "pfield_spectrum",
    "pfield_spectrum_compute",
    "plane_wave",
    "plane_wave_delays",
    "rms_from_spectrum",
    "scattered_field_precompute",
    "scattered_pfield_spectrum",
    "sequence_precompute",
    "simus",
    "simus_compute",
    "simus_precompute",
    "simus_sequence",
    "spectrum_to_wavefield",
    "transducer_from_params",
    "transform_aperture",
    "wavefield",
    "wavefield_times",
]
