# Pinned PyMUST 3D reference

`planar_two_element.npz` contains raw outputs from PyMUST commit `df02b422bf06fe298d352cd608a083d775c89193`. The JSON
records source hashes, versions, input parameters, options and scaling. Coordinate and delay arrays, selected field
frequencies and mask, and the complete echo frequency grid are stored in the NPZ. No output has been normalized to its
own peak.

The field reference calls `pfield3` with x, y and z each shaped `(1, 1, 2)`. The echo reference calls `simus3` with
those coordinates and reflectivity shaped `(1, 1, 2)`. Each call receives fresh deep copies of parameters/options and a
copy of delays because the reference mutates inputs. Channel order follows the stored centers. NPZ field arrays flatten
the spatial dimensions to the point axis.

Field spectra are unscaled complex amplitudes. Their RMS is `sqrt(df * sum(abs(spectrum)**2, axis=-1))`. Echo spectra
contain the raw selected bins embedded in the full one-sided grid; PyMUST does not multiply by `df`. Compare FastSIMUS
outputs on these exact grids and undo any documented FastSIMUS integration scaling before comparing raw spectra. The RF
array includes PyMUST's smooth relative threshold after the inverse transform. Do not compare spectra from independently
chosen grids by index.

The independent complex128 oracle in `tests/_reference_3d.py`, using PyMUST's pulse/probe spectral responses at the
stored field frequencies, differs from the raw field spectrum by 4.60e-6 of peak. The independent two-leg contraction
differs from the stored echo spectrum by 2.90e-6 of peak after matching its selected frequency mask. Both are below the
1e-4 reference gate. PyMUST casts geometry/phase to float32 and uses small coordinate epsilon offsets; the oracle uses
exact direction cosines and direct complex128 exponentiation. The fixture uses zero attenuation: PyMUST's attenuation
conversion uses 8.69, whereas FastSIMUS and the oracle use the exact `20/log(10)` conversion.
