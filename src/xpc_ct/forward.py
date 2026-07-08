"""
Forward models for 1D propagation-based X-ray phase-contrast CT.

This module contains auto-differentiable JAX forward models for parallel-beam
PB-XPCI measurements given an object defined by 2D "material maps". The material 
maps represent voxel-wise material volume fractions on a square `Nx` by `Nx` 
grid. For each projection angle, the object is rotated, converted to energy-
material-basis coefficients, propagated, and detected on a single-row detector.

Both projection approximation and multi-slice object modulation models are provided.

A `Material` class for calculating delta/beta can be found in `materials.py`.
A few preset materials are provided by name, for example 'water', 'bone', 'tissue'.
Any other material can be defined by its chemical composition. The NIST XCOM database
materials are listed in `nist_xcom.py`.

Alternatively, you can pass precomputed `delta_vals` and `beta_vals` directly.
This can eliminate the need for repeat delta/beta look-up table calls if running
the forward model several times in an iterative reconstruction.

Many forward model elements are precomputed to improve iterative recon speed.

All distances are in meters unless otherwise noted, and energies are in keV.

*** TODO ***
    - full 3D CT (2D projections at each angle). The current code only uses
      single-row (1D) projections to allow higher resolution without excessive
      memory requirements.
    - polychromatic scans. Current code allows for multiple monochromatic scans
      in a single forward run, but does not integrate them into one detected image.

"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.ndimage import map_coordinates
from flax import linen as nn

from .materials import basis_delta_beta_jnp, make_materials, wavelength_jnp


Array = Any
DTYPE = jnp.float32


class PBI_1D_Base(nn.Module):
    """
    Base class for 1D PB-XPCI forward models.

    Do not instantiate this class directly!! Use `ProjApproxPBI_1D` for the
    projection/thin-sample approximation or `MultiSlicePBI_1D` for multislice
    propagation through the object.

    Parameters
    ----------
    propdist:
        Free-space propagation distance from the object exit plane to the
        detector [m].
    I0_per_um2:
        Incident photons per square micron [um] per energy bin. This is 
        multiplied by the detector-pixel area to obtain per-pixel counts.
        Can be a scalar or array with shape `(n_energies)`.
    energies:
        Energy samples [keV], shape `(n_energies,)`.
    thetas:
        Projection angles [rad], shape `(n_angles,)`.
    delta_vals, beta_vals:
        Material refractive index components, shape `(n_energies, n_materials)`,
        where n = 1 - delta + i*beta. Can generate with `materials.basis_delta_beta_jnp`.
    material_names:
        Optional preset material names accepted by `materials.make_materials`.
        Used only when `delta_vals` and `beta_vals` are not supplied.
    material_E_min, material_E_max, material_dE:
        Energy-grid settings passed to `make_materials` when `material_names`
        are used.
    Nx, dx:
        Object matrix size and object pixel size [m]. The object is assumed to
        be square.
    dz:
        Multislice slice spacing [m]. Must be an integer multiple of `dx` and
        must divide `Nx * dx` evenly for `MultiSlicePBI_1D`.
    det_Nx, upx:
        Number of detector pixels and integer object-to-detector binning factor.
        The detector pixel size is `det_dx = upx * dx`.
    det_fwhm, det_psf:
        Optional detector point-spread function. Set `det_fwhm=None` to skip
        detector blurring.
    N_pad:
        Symmetric padding used during Fresnel propagation.
    cval:
        Constant fill value used when rotating material maps.
    dtype:
        Real dtype for energies, optical constants, and real-valued arrays.
    """

    # Acquisition settings.
    propdist: float = 0
    I0_per_um2: Array = 1.0
    energies: Array = field(default_factory=lambda: jnp.array([14.0], dtype=DTYPE))
    thetas: Array = field(default_factory=lambda: jnp.array([0.0], dtype=DTYPE))

    # Object set-up via delta/beta coefficients, shape ~ (n_energies, n_materials).
    delta_vals: Array | None = None
    beta_vals: Array | None = None

    # Alternative object set-up via preset material definitions, 
    # see materials.py for compatible names.
    material_names: tuple[str, ...] = ()
    material_E_min: float = 1.0
    material_E_max: float = 100.0
    material_dE: float = 1.0

    # Phantom/simulation grid.
    Nx: int = 64
    dx: float = 1e-6         # transverse plane (x, y) 
    dz: float | None = None  # propagation axis (z)

    # Detector.
    det_Nx: int | None = None
    upx: int = 1
    det_fwhm: float | None = None
    det_psf: str = 'lorentzian'
    psf_rel_thresh: float = 1e-3
    psf_max_kernel_width: float | None = 8.0

    # Propagation, ensure sufficient padding to avoid artifacts.
    N_pad: int = 8

    # Fill value for matrix rotations.
    cval: float = 0.0

    # Numeric dtype.
    dtype: Any = DTYPE

    @property
    def counts(self):
        """Incident photon counts per detector pixel for each energy bin."""
        det_dx_um = 1e6 * self.det_dx
        return self.I0_per_um2_jnp * det_dx_um**2
        
    @property
    def fov(self):
        """Object field of view [m]."""
        return self.dx * self.Nx

    @property
    def det_dx(self):
        """Detector pixel size [m]."""
        return self.upx * self.dx

    @property
    def det_fov(self):
        det_Nx = self.Nx // self.upx if self.det_Nx is None else self.det_Nx
        return self.det_dx * det_Nx

    @property
    def N_prop(self):
        """FFT grid length used for 1D Fresnel propagation."""
        return self.Nx + 2 * self.N_pad

    def setup(self):
        """Precompute material coefficients, detector parameters, and transfer functions."""
        self.energies_jnp = jnp.asarray(self.energies, dtype=self.dtype)
        self.thetas_jnp = jnp.asarray(self.thetas, dtype=self.dtype)

        self.basis_delta, self.basis_beta = self._setup_basis_coefficients()
        self.n_energies = int(self.basis_delta.shape[0])
        self.n_materials = int(self.basis_delta.shape[1])

        self.wavelens = wavelength_jnp(self.energies_jnp)
        self.wavenums = 2 * jnp.pi / self.wavelens
        
        self.I0_per_um2_jnp = _as_energy_array(
            self.I0_per_um2,
            name='I0_per_um2',
            n_energies=self.energies_jnp.shape[0],
            dtype=self.dtype,
        )

        if self.det_Nx is None:
            if self.Nx % self.upx != 0:
                raise ValueError('Require Nx % upx == 0 when det_Nx is inferred.')
            self.det_Nx_eff = self.Nx // self.upx
        else:
            self.det_Nx_eff = self.det_Nx
            
        self.det_pars = prepare_detector1D(
            Nx=self.Nx,
            dx=self.dx,
            det_Nx=self.det_Nx_eff,
            upx=self.upx,
        )

        self.psf_pars = prepare_psf1D(
            N=self.det_Nx_eff,
            dx=self.det_dx,
            fwhm=self.det_fwhm,
            psf=self.det_psf,
            rel_thresh=self.psf_rel_thresh,
            max_kernel_width=self.psf_max_kernel_width,
        )

        self.ms_pars = prepare_multislice_pars(
            Nx=self.Nx,
            dx=self.dx,
            dz=self.dx if self.dz is None else self.dz,
        )

        self.H_det = prepare_fresnel_transfer1D(
            N=self.N_prop,
            dx=self.dx,
            wavelens=self.wavelens,
            propdist=self.propdist,
        )

        self.H_slice = prepare_fresnel_transfer1D(
            N=self.N_prop,
            dx=self.dx,
            wavelens=self.wavelens,
            propdist=self.ms_pars['dz'],
        )
    
    def _setup_basis_coefficients(self):
        """Return `(delta_vals, beta_vals)` with shape `(nE, n_materials)`."""
        if (self.delta_vals is None) != (self.beta_vals is None):
            raise ValueError('delta_vals and beta_vals must be supplied together.')

        if self.delta_vals is None:
            if len(self.material_names) == 0:
                raise ValueError(
                    'Supply either delta_vals/beta_vals or material_names. '
                    'For arbitrary materials, precompute coefficients with '
                    'materials.basis_delta_beta_jnp(materials, energies).'
                )

            materials = make_materials(
                self.material_names,
                E_min=self.material_E_min,
                E_max=self.material_E_max,
                dE=self.material_dE,
                dtype=self.dtype,
            )
            delta_vals, beta_vals = basis_delta_beta_jnp(
                materials,
                self.energies_jnp,
                dtype=self.dtype,
            )
        else:
            delta_vals = self.delta_vals
            beta_vals = self.beta_vals

        delta_vals = _as_basis_coeff_array(
            delta_vals,
            name='delta_vals',
            n_energies=self.energies_jnp.shape[0],
            dtype=self.dtype,
        )
        beta_vals = _as_basis_coeff_array(
            beta_vals,
            name='beta_vals',
            n_energies=self.energies_jnp.shape[0],
            dtype=self.dtype,
        )

        if delta_vals.shape != beta_vals.shape:
            raise ValueError(
                'delta_vals and beta_vals must have matching shapes. '
                f'Got {delta_vals.shape} and {beta_vals.shape}.'
            )

        return delta_vals, beta_vals

    def make_delta_beta(self, basis_maps, energy_index):
        """
        Convert material-basis maps to delta and beta maps for one energy.

        Parameters
        ----------
        basis_maps:
            Array with shape `(n_materials, Nx, Nx)`.
        energy_index:
            Integer energy index.
        """
        delta_E = jnp.tensordot(self.basis_delta[energy_index], basis_maps, axes=(0, 0))
        beta_E = jnp.tensordot(self.basis_beta[energy_index], basis_maps, axes=(0, 0))
        return delta_E, beta_E

    def propagate1D(self, psi, H, pad_value):
        """Apply 1D Fresnel propagation to a complex wavefield."""
        if self.N_pad > 0:
            pad_value = jnp.asarray(pad_value, dtype=psi.dtype)
            psi = jnp.pad(
                psi,
                (self.N_pad, self.N_pad),
                mode='constant',
                constant_values=pad_value,
            )

        psi = jnp.fft.ifft(jnp.fft.fft(psi) * H)

        if self.N_pad > 0:
            psi = psi[self.N_pad:-self.N_pad]

        return psi

    def detect_field(self, field):
        """Convert a propagated wavefield to detector counts."""
        img = jnp.abs(field) ** 2
        img = block_average_detector1D(img, self.det_pars)
        img = apply_psf1D_precomp(img, self.psf_pars)
        return img

    def rotate_basis_maps(self, basis_maps, theta):
        """Rotate each material map by angle `theta`."""
        return jax.vmap(
            lambda arr: rotate_image(arr, theta, cval=self.cval),
            in_axes=0,
            out_axes=0,
        )(basis_maps)

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        """Forward model for one angle and one energy. Implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, material_maps) -> Array:
        """
        Simulate PB-XPCI measurements.

        Parameters
        ----------
        material_maps:
            Array of material volume fractions, shape (n_materials, Nx, Nx).

        Returns
        -------
        imgs:
            Expected photon counts with shape `(n_energies, n_angles, det_Nx)`.
        """
        basis_maps = _as_material_map_stack(material_maps)
        _check_material_maps(basis_maps, self.n_materials, self.Nx)

        energy_idxs = jnp.arange(self.n_energies)

        def angle_body(carry, theta):
            basis_theta = self.rotate_basis_maps(basis_maps, theta)

            def energy_body(carry, i):
                delta_theta, beta_theta = self.make_delta_beta(basis_theta, i)
                img = self.forward_one(
                    delta_theta=delta_theta,
                    beta_theta=beta_theta,
                    wavenum=self.wavenums[i],
                    count=self.counts[i],
                    H_det=self.H_det[i],
                    H_slice=self.H_slice[i],
                )
                return carry, img

            _, imgs_E = lax.scan(energy_body, None, energy_idxs)
            return carry, imgs_E

        _, imgs_theta_E = lax.scan(angle_body, None, self.thetas_jnp)
        return jnp.swapaxes(imgs_theta_E, 0, 1)


class ProjApproxPBI_1D(PBI_1D_Base):
    """Projection-approximation PB-XPCI forward model."""

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        """Simulate one projection using the thin-sample approximation."""
        del H_slice  # not used by the projection-approximation model

        delta_proj = jnp.sum(delta_theta, axis=0) * self.dx
        beta_proj = jnp.sum(beta_theta, axis=0) * self.dx

        transmission = jnp.exp(-1j * wavenum * delta_proj - wavenum * beta_proj)

        psi_in = jnp.sqrt(count) + 0j
        psi = psi_in * transmission
        psi = self.propagate1D(psi=psi, H=H_det, pad_value=psi_in)

        return self.detect_field(psi)


class MultiSlicePBI_1D(PBI_1D_Base):
    """Multislice PB-XPCI forward model."""

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        """Simulate one projection using multislice wave propagation."""
        delta_ms = coarsen_object_slices(delta_theta, self.ms_pars)
        beta_ms = coarsen_object_slices(beta_theta, self.ms_pars)
        dz = self.ms_pars['dz']

        psi_in = jnp.sqrt(count) + 0j
        psi = psi_in * jnp.ones((self.Nx,), dtype=jnp.result_type(psi_in, 1j))

        def transmit_slice(psi, slice_pars):
            delta_slice, beta_slice = slice_pars
            transmission = jnp.exp(
                -1j * wavenum * delta_slice * dz
                - wavenum * beta_slice * dz
            )
            return psi * transmission

        if delta_ms.shape[0] > 1:
            def slice_body(psi, slice_pars):
                psi = transmit_slice(psi, slice_pars)
                psi = self.propagate1D(psi=psi, H=H_slice, pad_value=psi_in)
                return psi, None

            psi, _ = lax.scan(slice_body, psi, (delta_ms[:-1], beta_ms[:-1]))

        psi = transmit_slice(psi, (delta_ms[-1], beta_ms[-1]))
        psi = self.propagate1D(psi=psi, H=H_det, pad_value=psi_in)

        return self.detect_field(psi)


# -----------------------------------------------------------------------------
# Geometry and detector helpers
# -----------------------------------------------------------------------------

def _as_energy_array(arr, name, n_energies, dtype=DTYPE):
    """Normalize scalar or per-energy inputs to shape `(n_energies,)`."""
    arr = jnp.asarray(arr, dtype=dtype)

    if arr.ndim == 0:
        return jnp.full((n_energies,), arr, dtype=dtype)

    arr = jnp.ravel(arr)

    if arr.shape[0] == 1:
        return jnp.full((n_energies,), arr[0], dtype=dtype)

    if arr.shape[0] != n_energies:
        raise ValueError(
            f'{name} must be scalar or have length n_energies={n_energies}. '
            f'Got shape {arr.shape}.'
        )

    return arr

def _as_basis_coeff_array(arr, name, n_energies, dtype=DTYPE):
    """Normalize material coefficient arrays to shape `(nE, n_materials)`."""
    arr = jnp.asarray(arr, dtype=dtype)

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(
            f'{name} must have shape (n_energies, n_materials). '
            f'Got shape {arr.shape}.'
        )

    if arr.shape[0] != n_energies:
        raise ValueError(
            f'{name} first dimension must match number of energies. '
            f'Got {arr.shape[0]} and {n_energies}.'
        )

    return arr


def _as_material_map_stack(material_maps):
    """Normalize material map inputs to shape `(n_materials, Nx, Nx)`."""
    if isinstance(material_maps, (tuple, list)):
        return jnp.stack([jnp.asarray(m) for m in material_maps], axis=0)

    arr = jnp.asarray(material_maps)

    if arr.ndim == 2:
        return arr[None, ...]

    return arr


def _check_material_maps(material_maps, n_materials, Nx):
    """Validate material map array shape."""
    if material_maps.ndim != 3:
        raise ValueError(
            'material_maps must have shape (n_materials, Nx, Nx).'
        )

    if material_maps.shape[0] != n_materials:
        raise ValueError(
            f'Expected {n_materials} material maps from optical coefficients, '
            f'but got {material_maps.shape[0]}.'
        )

    if material_maps.shape[1:] != (Nx, Nx):
        raise ValueError(
            f'Expected material map shape ({n_materials}, {Nx}, {Nx}), '
            f'but got {material_maps.shape}.'
        )


def make_rotation_locations_2d(shape, angle, dtype=DTYPE):
    """
    Return interpolation coordinates for rotating a 2D image about its center.

    Parameters
    ----------
    shape:
        `(Ny, Nx)` image shape.
    angle:
        Rotation angle [rad].
    dtype:
        Coordinate dtype.
    """
    Ny, Nx = shape
    cy = (Ny - 1) / 2
    cx = (Nx - 1) / 2

    y = jnp.arange(Ny, dtype=dtype) - cy
    x = jnp.arange(Nx, dtype=dtype) - cx
    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cos_t = jnp.cos(angle).astype(dtype)
    sin_t = jnp.sin(angle).astype(dtype)

    y_in = cos_t * yy - sin_t * xx + cy
    x_in = sin_t * yy + cos_t * xx + cx

    return jnp.stack([y_in, x_in], axis=0)


def rotate_image(image, angle, cval=0.0):
    """Rotate a 2D image using bilinear interpolation and constant fill."""
    image = jnp.asarray(image)
    locs = make_rotation_locations_2d(
        shape=image.shape,
        angle=angle,
        dtype=image.real.dtype,
    )

    return map_coordinates(
        image,
        locs,
        order=1,
        mode='constant',
        cval=cval,
    )


def prepare_detector1D(Nx, dx, det_Nx, upx):
    """
    Precompute centered crop/binning parameters for a 1D detector.

    `upx` is the number of object-grid pixels averaged into one detector
    pixel. The detector crop is centered in the propagated wavefield.
    """
    if upx < 1 or int(upx) != upx:
        raise ValueError('upx must be a positive integer.')

    if Nx < det_Nx * upx:
        raise ValueError('Require Nx >= det_Nx * upx.')

    N_crop = int(det_Nx * upx)
    i0 = int((Nx - N_crop) // 2)
    i1 = int(i0 + N_crop)

    return dict(
        Nx=int(Nx),
        dx=dx,
        det_Nx=int(det_Nx),
        upx=int(upx),
        det_dx=upx * dx,
        N_crop=N_crop,
        i0=i0,
        i1=i1,
    )


def block_average_detector1D(img, det_pars):
    """Center-crop and block-average a 1D intensity profile onto detector pixels."""
    upx = det_pars['upx']
    det_Nx = det_pars['det_Nx']
    i0 = det_pars['i0']
    i1 = det_pars['i1']

    img = img[..., i0:i1]

    if upx == 1:
        return img

    img = img.reshape(*img.shape[:-1], det_Nx, upx)
    return jnp.mean(img, axis=-1)


# -----------------------------------------------------------------------------
# Propagation and multislice helpers
# -----------------------------------------------------------------------------


def prepare_fresnel_transfer1D(N, dx, wavelens, propdist):
    """
    Precompute 1D Fresnel transfer functions.

    Parameters
    ----------
    N:
        FFT grid length.
    dx:
        Grid spacing [m].
    wavelens:
        Wavelengths [m], shape `(n_energies,)`.
    propdist:
        Propagation distance [m].

    Returns
    -------
    H:
        Complex transfer functions with shape `(n_energies, N)`.
    """
    wavelens = jnp.atleast_1d(jnp.asarray(wavelens))
    freqs = jnp.fft.fftfreq(N, d=dx)
    return jnp.exp(-1j * jnp.pi * wavelens[:, None] * propdist * freqs[None, :]**2)


def prepare_multislice_pars(Nx, dx, dz=None):
    """
    Precompute slice coarsening parameters for multislice propagation.

    `dz` must be an integer multiple of `dx` so slices can be formed by
    averaging adjacent object-grid rows.
    """
    if dz is None:
        dz = dx

    ratio = dz / dx
    slice_step = int(round(ratio))

    if not math.isclose(ratio, slice_step, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('Require dz to be an integer multiple of dx.')

    if Nx % slice_step != 0:
        raise ValueError('Require Nx % slice_step == 0.')

    Nz = Nx // slice_step

    return dict(
        slice_step=int(slice_step),
        Nz=int(Nz),
        dz=slice_step * dx,
    )


def coarsen_object_slices(arr, ms_pars):
    """Average adjacent object-grid rows into multislice slabs."""
    slice_step = ms_pars['slice_step']
    Nz = ms_pars['Nz']

    if slice_step == 1:
        return arr

    arr = arr.reshape(Nz, slice_step, arr.shape[-1])
    return jnp.mean(arr, axis=1)


# -----------------------------------------------------------------------------
# Detector PSF helpers
# -----------------------------------------------------------------------------


def lorentzian1D(x, fwhm, normalize=True):
    """Return a normalized 1D Lorentzian kernel sampled at `x`."""
    gamma = fwhm / 2
    kernel = gamma / (jnp.pi * (x**2 + gamma**2))

    if normalize:
        kernel = kernel / jnp.sum(kernel)

    return kernel


def gaussian1D(x, fwhm, normalize=True):
    """Return a normalized 1D Gaussian kernel sampled at `x`."""
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)

    if normalize:
        kernel = kernel / jnp.sum(kernel)

    return kernel


def get_psf_kernel_size(dx, fwhm, psf='lorentzian', rel_thresh=1e-3, max_kernel_width=8.0):
    """Choose a symmetric half-width size, in pixels, for a 1D detector PSF kernel."""
    if fwhm is None:
        return 0

    psf = psf.lower()

    if psf == 'lorentzian':
        gamma = fwhm / 2
        half_width = gamma * math.sqrt(1 / rel_thresh - 1)
    elif psf == 'gaussian':
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        half_width = sigma * math.sqrt(2 * math.log(1 / rel_thresh))
    else:
        raise ValueError(f'Unknown PSF: {psf}')

    if max_kernel_width is not None:
        half_width = min(half_width, max_kernel_width * fwhm)

    return max(1, math.ceil(half_width / dx))


def get_psf_kernel1D(dx, fwhm, psf='lorentzian', rel_thresh=1e-3, max_kernel_width=8.0):
    """Build a normalized 1D detector PSF kernel."""
    psf = psf.lower()
    n_half = get_psf_kernel_size(
        dx=dx,
        fwhm=fwhm,
        psf=psf,
        rel_thresh=rel_thresh,
        max_kernel_width=max_kernel_width,
    )
    x = jnp.arange(-n_half, n_half + 1) * dx

    if psf == 'lorentzian':
        return lorentzian1D(x, fwhm)
    if psf == 'gaussian':
        return gaussian1D(x, fwhm)

    raise ValueError(f'Unknown PSF: {psf}')


def prepare_psf1D(N, dx, fwhm=None, psf='lorentzian', rel_thresh=1e-3, max_kernel_width=8.0):
    """Precompute FFT parameters for optional 1D detector PSF convolution."""
    if fwhm is None:
        return None

    if fwhm <= 0:
        raise ValueError('fwhm must be positive or None.')

    kernel = get_psf_kernel1D(
        dx=dx,
        fwhm=fwhm,
        psf=psf,
        rel_thresh=rel_thresh,
        max_kernel_width=max_kernel_width,
    )

    K = kernel.shape[0]
    pad = (K - 1) // 2
    fft_N = N + 2 * pad + K - 1
    G = jnp.fft.rfft(kernel, n=fft_N)
    o = (K - 1) // 2

    return dict(
        kernel=kernel,
        G=G,
        K=K,
        pad=pad,
        fft_N=fft_N,
        o=o,
    )


def convolve1d_diy_precomp(img, psf_pars):
    """Convolve along the final axis using a precomputed 1D FFT kernel."""
    G = psf_pars['G']
    fft_N = psf_pars['fft_N']
    o = psf_pars['o']
    N = img.shape[-1]

    F = jnp.fft.rfft(img, n=fft_N, axis=-1)
    out = jnp.fft.irfft(F * G, n=fft_N, axis=-1)
    return out[..., o:o + N]


def apply_psf1D_precomp(img, psf_pars):
    """Apply optional 1D detector PSF convolution along the final axis."""
    if psf_pars is None:
        return img

    pad = psf_pars['pad']
    pad_width = [(0, 0)] * img.ndim
    pad_width[-1] = (pad, pad)

    img_pad = jnp.pad(img, pad_width, mode='edge')
    img_nonideal_pad = convolve1d_diy_precomp(img_pad, psf_pars)
    return img_nonideal_pad[..., pad:-pad]


# -----------------------------------------------------------------------------
# Noise helper
# -----------------------------------------------------------------------------


def add_poisson_noise(imgs, key, scale=1.0):
    """
    Add Poisson noise to expected-count data.

    Parameters
    ----------
    imgs:
        Expected counts.
    key:
        JAX PRNG key.
    scale:
        Optional scale factor. The returned array is
        `poisson(scale * imgs) / scale`.
    """
    return jax.random.poisson(key, scale * imgs, imgs.shape) / scale
