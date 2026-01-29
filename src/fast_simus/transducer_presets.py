"""Preset transducer configurations for common ultrasound probes.

These presets are based on Verasonics transducers and match the values
from the MUST toolbox (getparam.m).
"""

from math import inf

from fast_simus import TransducerParams


def P4_2v() -> TransducerParams:
    """Create P4-2v phased array transducer preset.

    Verasonics P4-2v phased array transducer for cardiac imaging:
    - 2.72 MHz center frequency
    - 64 elements
    - 300 µm (0.3 mm) pitch

    Returns:
        TransducerParams instance configured for P4-2v transducer.
    """
    return TransducerParams(
        freq_center=2.72e6,  # 2.72 MHz
        pitch=300e-6,  # 300 µm
        n_elements=64,
        kerf=50e-6,  # 50 µm
        bandwidth=0.74,  # 74%
        radius=inf,
        height=14e-3,  # 14 mm
        elev_focus=60e-3,  # 60 mm
    )


def L11_5v() -> TransducerParams:
    """Create L11-5v linear array transducer preset.

    Verasonics L11-5v high-frequency linear array for vascular imaging:
    - 7.6 MHz center frequency
    - 128 elements
    - 300 µm (0.3 mm) pitch

    Returns:
        TransducerParams instance configured for L11-5v transducer.
    """
    return TransducerParams(
        freq_center=7.6e6,  # 7.6 MHz
        pitch=300e-6,  # 300 µm
        n_elements=128,
        kerf=30e-6,  # 30 µm
        bandwidth=0.77,  # 77%
        radius=inf,
        height=5e-3,  # 5 mm
        elev_focus=18e-3,  # 18 mm
    )


def L12_3v() -> TransducerParams:
    """Create L12-3v linear array transducer preset.

    Verasonics L12-3v high-frequency linear array:
    - 7.54 MHz center frequency
    - 192 elements
    - 200 µm (0.2 mm) pitch

    Returns:
        TransducerParams instance configured for L12-3v transducer.
    """
    return TransducerParams(
        freq_center=7.54e6,  # 7.54 MHz
        pitch=200e-6,  # 200 µm
        n_elements=192,
        kerf=30e-6,  # 30 µm
        bandwidth=0.93,  # 93%
        radius=inf,
        height=5e-3,  # 5 mm
        elev_focus=20e-3,  # 20 mm
    )


def C5_2v() -> TransducerParams:
    """Create C5-2v convex array transducer preset.

    Verasonics C5-2v convex array for abdominal imaging:
    - 3.57 MHz center frequency
    - 128 elements
    - 508 µm (0.508 mm) pitch
    - 49.57 mm radius of curvature

    Returns:
        TransducerParams instance configured for C5-2v transducer.
    """
    return TransducerParams(
        freq_center=3.57e6,  # 3.57 MHz
        pitch=508e-6,  # 508 µm
        n_elements=128,
        kerf=48e-6,  # 48 µm
        bandwidth=0.79,  # 79%
        radius=49.57e-3,  # 49.57 mm
        height=13.5e-3,  # 13.5 mm
        elev_focus=60e-3,  # 60 mm
    )
