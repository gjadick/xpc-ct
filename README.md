# XPC-CT

An auto-differentiable Python framework for simulating and reconstructing propagation-based x-ray phase-contrast CT data.

This code base includes helpers for defining materials in terms of their refractive indices and attenuation coefficients, 
building simple phantoms, simulating parallel-beam PBI (various forward models) and CT scans, applying analytical and iterative 
phase-retrieval/material-decomposition methods, and reconstructing CT images with filtered back-projection.


## Overview

The typical workflow is:

1. Define a set of basis materials, such as water, tissue, adipose, bone, PMMA, or aluminum.
2. Build material maps describing the object.
3. Simulate propagation-based projection data with the forward model.
4. Optionally apply projection-domain phase retrieval or material decomposition.
5. Reconstruct CT images with iterative approach or FBP.

Examples are shown in `example.ipynb`, which demonstrates how to string modules and functions together.


## File summary

* `example.ipynb` — Example usage notebook.
* `forward.py` — JAX forward models for 1D propagation-based X-ray phase-contrast CT simulations.
* `materials.py` — Material definitions and helpers for computing refractive indices and physical constants.
* `nist_xcom.py` — NIST XCOM material and element definitions (used by `materials.py`).
* `phantoms.py` — Simple phantom-creation and resampling utilities.
* `phase_retrieval.py` — Paganin phase retrieval and linearized TIE material-decomposition helpers.
* `tomography.py` — CT rotation, projection, and reconstruction utilities.
* `metrics.py` — Simple image-quality metrics such as RMSE and SSIM.
* `plotting.py` — Preferred matplotlib plotting defaults and helper functions.


## Example

```python
import jax.numpy as jnp

from xpc_ct.materials import make_materials
from xpc_ct.forward import ProjApproxPBI_1D
from xpc_ct.tomography import get_recon

materials = make_materials(('water', 'bone'))  

fwd_model = ProjApproxPBI_1D(
    material_names=('water', 'bone'),
    energies=jnp.array([14.0]),
    thetas=jnp.linspace(0, jnp.pi, 180, endpoint=False),
    Nx=128,
    dx=1e-6,
    propdist=0.05,
)

# Must create `material_maps` phantom, shape ~ (n_materials, Nx, Nx), then run:
imgs = fwd_model.apply({}, material_maps)

```

## TODO

- Add 2D projections / 3D CT
- Add polychromatic acquisition bins
- Add delta/beta/other reconstruction options
- Memory management strategies
