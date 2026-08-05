"""
Auto-differentiable spectral propagation-based x-ray phase-contrast CT forward models.

Both projection-approximation and multi-slice modulation models are provided through
`ProjApproxPBI` and `MultiSlicePBI`. 

There are two main geometries considered:

1. Single-row / 1D detector data
   Material maps have shape `(n_materials, Nz, Nx)`, representing the
   `(z, x)` CT slice, and each projection has shape `(det_Nx,)`. Fresnel
   propagation uses a 1D FFT along x. This is more efficient for demoing
   very large/high-res CT arrays, since it is just one slice, but it is
   not fully 3D.

2. Full 3D parallel-beam CT with a 2D detector panel
   Material maps have shape `(n_materials, Nz, Ny, Nx)`, with spatial axis
   order `(z, y, x)`, and each projection has shape `(det_Ny, det_Nx)`.
   The object rotates about the y axis, and Fresnel propagation uses a 2D FFT
   over the detector plane `(y, x)`.

Leaving `Ny=None` automatically selects single-row geometry for efficiency,
while supplying `Ny` uses full detector-panel geometry.

Dimension convention: z is the beam direction, x is detector-horizontal,
and y is detector-vertical. This also makes y the CT rotation axis, with rotation 
performed in each `(z, x)` plane. Projection or multislice stepping is along the 
propagation axis.

Each energy/wavelength is simulated independently as a monochromatic intensity image.
The monochromatic intensities are combined using `spectra_weights`, which are used as 
supplied and are not normalized. The weights can account for the shape of the spectra
as well as any polychromatic system effects, such as a detector efficiency function,
which should be pre-multiplied. 
There is an additional optional intensity scale `I0_per_um` (the intensity per square micron area).

All distances are in meters unless otherwise noted (e.g. `um`). Energies are in keV.
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax.scipy.ndimage import map_coordinates

from .materials import basis_delta_beta_jnp, make_materials, wavelength_jnp


Array = Any
DTYPE = jnp.float32


class PBI_Base(nn.Module):
    """
    Shared base class for single-row, full-panel, projection approximation,
    and multi-slice PB-XPCI forward models.

    Do not instantiate this class directly. Use `ProjApproxPBI` or
    `MultiSlicePBI`.

    Parameters
    ----------
    propdist:
        Free-space distance from the object exit plane to the detector [m].
    I0_per_um2:
        Incident photons per square micron for each spectrum. May be a scalar
        shared by all spectra or an array with shape `(n_spectra,)`. When each
        corresponding `spectra_weights` row sums to one, this is the total
        incident fluence integrated over the simulated energies.
    energies:
        Monochromatic energy samples [keV], shape `(n_energies,)`.
    spectra_weights:
        Photon weights with shape `(n_spectra, n_energies)`. A 1D array with
        shape `(n_energies,)` is interpreted as one spectrum. Weights are used
        as supplied and are not normalized. When `None`, an identity matrix is
        used so that each output corresponds to one monochromatic energy.
    thetas:
        Projection angles [rad], shape `(n_angles,)`.
    delta_vals, beta_vals:
        Optional precomputed material optical constants with shape
        `(n_energies, n_materials)`.
    material_names:
        Preset material names used when `delta_vals` and `beta_vals` are not
        supplied.
    Nx, Ny, Nz:
        Object shape along the x, y, and z axes. Full volumes use axis order
        `(Nz, Ny, Nx)`, while single-slice objects use `(Nz, Nx)`.

        - x: object and detector width.
        - y: object and detector height and the CT rotation axis.
        - z: object depth and the beam direction at zero angle.

        Leaving `Ny=None` selects single-row geometry. Supplying `Ny` selects
        panel geometry. When `Nz=None`, `Nz` defaults to `Nx`. Each CT slice is
        in the x-z plane for a fixed y value.
    dx, dy, dz:
        Object voxel spacing along x, y, and z [m]. `dy` and `dz` default to
        `dx`. In single-row geometry, `dy` specifies the physical detector-row
        spacing used for photon-count scaling. Rotation occurs in the physical
        x-z plane about y.
    multislice_thickness:
        Multislice slab thickness along the beam/z direction [m]. It defaults
        to `dz` and must be an integer multiple of `dz` that evenly divides
        `Nz * dz`.
    det_Nx, det_Ny:
        Detector dimensions along x and y. When omitted, they are inferred
        from the native propagated-field dimensions and detector binning.
    upx, upy:
        Positive integer detector up-sampling factors along x and y.
    fwhm:
        Optional detector PSF FWHM [m]. May be a scalar or one value per
        detector dimension. For panels, the PSF is applied separably along y
        and x. Individual dimensions may be disabled with `None`.
    psf:
        `'gaussian'` or `'lorentzian'`. The panel implementation applies the
        selected 1D profile separably along each detector axis.
    N_pad:
        Symmetric propagation padding in pixels. May be a scalar or one value
        per propagated-field dimension. Tuple order is x for a detector row
        and `(y, x)` for a detector panel.
    cval:
        Constant fill used outside the object during rotation.
    checkpoint_angles, checkpoint_slices:
        Apply JAX rematerialization to angle or multislice scan bodies to trade
        additional computation for lower reverse-mode autodiff memory.
    dtype:
        Real dtype for object coefficients and real-valued calculations.

    Returns
    -------
    imgs:
        Expected photon counts. The first dimension is always `n_spectra`:

        - Single-row: `(n_spectra, n_angles, det_Nx)`.
        - Full-panel: `(n_spectra, n_angles, det_Ny, det_Nx)`.
    """

    # Acquisition settings.
    propdist: float = 0.0
    I0_per_um2: Array = 1.0
    energies: Array = field(default_factory=lambda: jnp.array([14.0], dtype=DTYPE))
    spectra_weights: Array | None = None
    thetas: Array = field(default_factory=lambda: jnp.array([0.0], dtype=DTYPE))

    # Object setup via precomputed optical constants.
    delta_vals: Array | None = None
    beta_vals: Array | None = None

    # Alternative object setup via material definitions.
    material_names: tuple[str, ...] = ()
    material_E_min: float = 1.0
    material_E_max: float = 100.0
    material_dE: float = 1.0

    # Object grid. Axis order is (z, y, x) for volumes and (z, x) for slices.
    Nx: int = 64
    Ny: int | None = None
    Nz: int | None = None
    dx: float = 1e-6
    dy: float | None = None
    dz: float | None = None

    # Multislice slab thickness along the beam/z direction.
    multislice_thickness: float | None = None

    # Detector grid.
    det_Nx: int | None = None
    det_Ny: int | None = None
    upx: int = 1
    upy: int = 1

    # System point-spread-function (PSF) blur.
    fwhm: Array | None = None
    psf: str = 'lorentzian'
    psf_rel_thresh: float = 1e-3
    psf_max_kernel_width: float | None = 8.0

    # Fresnel padding.
    N_pad: int | tuple[int, ...] = 8

    # Rotation fill value.
    cval: float = 0.0

    # Optional autodiff memory controls.
    checkpoint_angles: bool = False
    checkpoint_slices: bool = False

    dtype: Any = DTYPE

    @property
    def field_ndim(self):
        """Number of dimensions in each propagated wavefield."""
        return 2 if self.Ny is not None else 1

    @property
    def Ny_eff(self):
        """Effective y dimension for panel models."""
        if self.field_ndim == 1:
            return None
        return self.Nx if self.Ny is None else self.Ny

    @property
    def Nz_eff(self):
        """Effective beam-direction object dimension."""
        return self.Nx if self.Nz is None else self.Nz

    @property
    def dy_eff(self):
        """Effective y spacing [m]."""
        return self.dx if self.dy is None else self.dy

    @property
    def dz_eff(self):
        """Effective z voxel spacing [m]."""
        return self.dx if self.dz is None else self.dz

    @property
    def fov_x(self):
        """Object x field of view [m]."""
        return self.dx * self.Nx

    @property
    def fov_y(self):
        """Object y field of view [m], or None for single-row models."""
        if self.field_ndim == 1:
            return None
        return self.dy_eff * self.Ny_eff

    @property
    def fov_z(self):
        """Object z field of view [m]."""
        return self.dz_eff * self.Nz_eff

    @property
    def det_dx(self):
        """Detector x pixel size [m]."""
        return self.upx * self.dx

    @property
    def det_dy(self):
        """Detector y pixel size [m]."""
        if self.field_ndim == 1:
            return self.dy_eff
        return self.upy * self.dy_eff

    @property
    def det_fov_x(self):
        """Detector x field of view [m]."""
        det_Nx = self.Nx // self.upx if self.det_Nx is None else self.det_Nx
        return self.det_dx * det_Nx

    @property
    def det_fov_y(self):
        """Detector y field of view [m], or None for single-row models."""
        if self.field_ndim == 1:
            return None
        det_Ny = self.Ny_eff // self.upy if self.det_Ny is None else self.det_Ny
        return self.det_dy * det_Ny

    @property
    def detector_pixel_area_um2(self):
        """Physical detector-pixel area in square microns."""
        return (1e6 * self.det_dx) * (1e6 * self.det_dy)

    @property
    def flat_field_signal(self):
        """No-object detector signal for each spectrum."""
        weights = _as_spectra_weights(
            self.spectra_weights,
            n_energies=jnp.atleast_1d(self.energies).size,
            dtype=self.dtype,
        )
        I0 = _as_spectrum_array(
            self.I0_per_um2,
            name='I0_per_um2',
            n_spectra=weights.shape[0],
            dtype=self.dtype,
        )
        return I0 * self.detector_pixel_area_um2 * jnp.sum(weights, axis=1)
        
        
    def setup(self):
        """Precompute geometry, spectrum, material, detector, and propagation terms."""
        self._validate_geometry()

        self.energies_jnp = jnp.atleast_1d(jnp.asarray(self.energies, dtype=self.dtype))
        self.thetas_jnp = jnp.atleast_1d(jnp.asarray(self.thetas, dtype=self.dtype))

        self.basis_delta, self.basis_beta = self._setup_basis_coefficients()
        self.n_energies = int(self.basis_delta.shape[0])
        self.n_materials = int(self.basis_delta.shape[1])

        self.spectra_weights_jnp = _as_spectra_weights(
            self.spectra_weights,
            n_energies=self.n_energies,
            dtype=self.dtype,
        )
        self.n_spectra = int(self.spectra_weights_jnp.shape[0])
        self.I0_per_um2_jnp = _as_spectrum_array(
            self.I0_per_um2,
            name='I0_per_um2',
            n_spectra=self.n_spectra,
            dtype=self.dtype,
        )

        self.wavelens = wavelength_jnp(self.energies_jnp)
        self.wavenums = 2 * jnp.pi / self.wavelens

        if self.field_ndim == 1:
            # Single-row geometry with object shape (z, x).
            self.object_shape = (int(self.Nz_eff), int(self.Nx))
            self.object_spacing = (float(self.dz_eff), float(self.dx))
            self.field_shape = (int(self.Nx),)
            self.field_spacing = (float(self.dx),)
            self.det_shape_eff = (
                _infer_detector_size(self.Nx, self.det_Nx, self.upx, 'x'),
            )
            self.det_binning = (int(self.upx),)
        else:
            # Full volume in (z, y, x); propagated panel field in (y, x).
            self.object_shape = (int(self.Nz_eff), int(self.Ny_eff), int(self.Nx))
            self.object_spacing = (
                float(self.dz_eff),
                float(self.dy_eff),
                float(self.dx),
            )
            self.field_shape = (int(self.Ny_eff), int(self.Nx))
            self.field_spacing = (float(self.dy_eff), float(self.dx))
            self.det_shape_eff = (
                _infer_detector_size(self.Ny_eff, self.det_Ny, self.upy, 'y'),
                _infer_detector_size(self.Nx, self.det_Nx, self.upx, 'x'),
            )
            self.det_binning = (int(self.upy), int(self.upx))

        self.det_Nx_eff = self.det_shape_eff[-1]
        self.det_Ny_eff = None if self.field_ndim == 1 else self.det_shape_eff[-2]

        self.pad = _as_int_tuple(self.N_pad, self.field_ndim, name='N_pad')
        self.prop_shape = tuple(
            N + 2 * pad_i
            for N, pad_i in zip(self.field_shape, self.pad)
        )

        self.det_pars = prepare_detector(
            field_shape=self.field_shape,
            field_spacing=self.field_spacing,
            det_shape=self.det_shape_eff,
            binning=self.det_binning,
        )

        fwhm_tuple = _as_optional_float_tuple(
            self.fwhm,
            self.field_ndim,
            name='fwhm',
        )
        self.psf_pars = prepare_psf(
            shape=self.det_shape_eff,
            spacing=self.det_pars['det_spacing'],
            fwhm=fwhm_tuple,
            psf=self.psf,
            rel_thresh=self.psf_rel_thresh,
            max_kernel_width=self.psf_max_kernel_width,
        )

        self.ms_pars = prepare_multislice_pars(
            Nz=self.Nz_eff,
            dz=self.dz_eff,
            multislice_thickness=self.multislice_thickness,
        )

        self.H_det = prepare_fresnel_transfer(
            shape=self.prop_shape,
            spacing=self.field_spacing,
            wavelens=self.wavelens,
            propdist=self.propdist,
        )

        self.H_slice = prepare_fresnel_transfer(
            shape=self.prop_shape,
            spacing=self.field_spacing,
            wavelens=self.wavelens,
            propdist=self.ms_pars['slice_thickness'],
        )

        self.spectrum_counts = (
            self.I0_per_um2_jnp
            * jnp.asarray(self.detector_pixel_area_um2, dtype=self.dtype)
        )

    def _validate_geometry(self):
        """Validate static geometry parameters before array construction."""
        if self.field_ndim not in (1, 2):
            raise ValueError('field_ndim must be 1 or 2.')

        for name, value in (
            ('Nx', self.Nx),
            ('Nz', self.Nz_eff),
        ):
            if int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer.')

        if self.field_ndim == 1 and self.Ny is not None:
            raise ValueError('Ny is only valid for a 2D detector-panel model.')

        if self.field_ndim == 2:
            if int(self.Ny_eff) != self.Ny_eff or self.Ny_eff < 1:
                raise ValueError('Ny must be a positive integer.')

        if self.dx <= 0:
            raise ValueError('dx must be positive.')

        if self.dz_eff <= 0:
            raise ValueError('dz must be positive.')

        if self.dy_eff <= 0:
            raise ValueError('dy must be positive.')

        if self.multislice_thickness is not None and self.multislice_thickness <= 0:
            raise ValueError('multislice_thickness must be positive or None.')

        for name, value in (
            ('upx', self.upx),
            ('upy', self.upy),
        ):
            if int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer.')

        if self.det_Nx is not None and (
            int(self.det_Nx) != self.det_Nx or self.det_Nx < 1
        ):
            raise ValueError('det_Nx must be a positive integer or None.')

        if self.det_Ny is not None and (
            int(self.det_Ny) != self.det_Ny or self.det_Ny < 1
        ):
            raise ValueError('det_Ny must be a positive integer or None.')

        if self.field_ndim == 1 and self.det_Ny is not None:
            raise ValueError('det_Ny is only valid for a 2D detector panel.')


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
        """Convert material-basis maps to delta and beta at one energy."""
        delta_E = jnp.tensordot(
            self.basis_delta[energy_index],
            basis_maps,
            axes=(0, 0),
        )
        beta_E = jnp.tensordot(
            self.basis_beta[energy_index],
            basis_maps,
            axes=(0, 0),
        )
        return delta_E, beta_E

    def propagate(self, psi, H, pad_value):
        """Apply rank-aware Fresnel propagation to a 1D or 2D wavefield."""
        if any(self.pad):
            pad_value = jnp.asarray(pad_value, dtype=psi.dtype)
            pad_width = tuple((pad_i, pad_i) for pad_i in self.pad)
            psi = jnp.pad(
                psi,
                pad_width,
                mode='constant',
                constant_values=pad_value,
            )

        if self.field_ndim == 1:
            psi = jnp.fft.ifft(jnp.fft.fft(psi) * H)
        else:
            psi = jnp.fft.ifftn(
                jnp.fft.fftn(psi, axes=(-2, -1)) * H,
                axes=(-2, -1),
            )

        if any(self.pad):
            crop = tuple(
                slice(pad_i, -pad_i if pad_i > 0 else None)
                for pad_i in self.pad
            )
            psi = psi[crop]

        return psi

    def detect_field(self, field):
        """Convert a propagated field to detector intensity."""
        img = jnp.abs(field) ** 2
        img = block_average_detector(img, self.det_pars)
        img = apply_psf_precomp(img, self.psf_pars)
        return img

    def rotate_basis_maps(self, basis_maps, theta):
        """Rotate every material map in the physical z-x plane about y."""
        if self.field_ndim == 1:
            return rotate_last_two_axes(
                basis_maps,
                angle=theta,
                spacing=(self.dz_eff, self.dx),
                cval=self.cval,
            )

        # For full volumes, basis_maps has shape (material, z, y, x).
        # Move y ahead of the rotated (z, x) plane, rotate slice-wise,
        # then move y back to recover (material, z, y, x).
        basis_maps_yfirst = jnp.moveaxis(basis_maps, -2, -3)
        basis_maps_yfirst = rotate_last_two_axes(
            basis_maps_yfirst,
            angle=theta,
            spacing=(self.dz_eff, self.dx),
            cval=self.cval,
        )
        return jnp.moveaxis(basis_maps_yfirst, -3, -2)

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        """Simulate one angle and energy. Implemented by subclasses."""
        raise NotImplementedError

    def _simulate_monochromatic(self, basis_maps, thetas_jnp):
        """Return unit-incident monochromatic intensities for all energies."""
        energy_idxs = jnp.arange(self.n_energies)
        unit_count = jnp.asarray(1.0, dtype=self.dtype)

        def angle_body(carry, theta):
            basis_theta = self.rotate_basis_maps(basis_maps, theta)

            def energy_body(energy_carry, i):
                delta_theta, beta_theta = self.make_delta_beta(basis_theta, i)
                img = self.forward_one(
                    delta_theta=delta_theta,
                    beta_theta=beta_theta,
                    wavenum=self.wavenums[i],
                    count=unit_count,
                    H_det=self.H_det[i],
                    H_slice=self.H_slice[i],
                )
                return energy_carry, img

            _, imgs_E = lax.scan(energy_body, None, energy_idxs)
            return carry, imgs_E

        scan_body = jax.checkpoint(angle_body) if self.checkpoint_angles else angle_body
        _, imgs_theta_E = lax.scan(scan_body, None, thetas_jnp)
        return jnp.swapaxes(imgs_theta_E, 0, 1)

    def __call__(self, material_maps, thetas=None) -> Array:
        """
        Simulate monochromatic or spectrally weighted PB-XPCI measurements.

        Parameters
        ----------
        material_maps:
            Single-row geometry: `(n_materials, Nz, Nx)` in `(z, x)` order.
            Full-panel geometry: `(n_materials, Nz, Ny, Nx)` in `(z, y, x)`
            order. A missing material axis is accepted for a single material.
        thetas:
            Optional angle subset. Supplying this enables exact angle batching
            without rebuilding energy-dependent transfer functions.

        Returns
        -------
        imgs:
            Expected photon counts with the first dimension equal to
            `n_spectra`:

            - Single-row: `(n_spectra, n_angles, det_Nx)`.
            - Full-panel: `(n_spectra, n_angles, det_Ny, det_Nx)`.
        """
        basis_maps = _as_material_map_stack(
            material_maps,
            object_ndim=self.field_ndim + 1,
        )
        _check_material_maps(
            basis_maps,
            n_materials=self.n_materials,
            object_shape=self.object_shape,
        )

        thetas_jnp = self.thetas_jnp if thetas is None else jnp.atleast_1d(
            jnp.asarray(thetas, dtype=self.dtype)
        )

        imgs_mono = self._simulate_monochromatic(basis_maps, thetas_jnp)

        # Incoherent spectral integration. imgs_mono has shape
        # (n_energies, n_angles, *det_shape), so contracting the energy axis
        # produces (n_spectra, n_angles, *det_shape).
        imgs_spectra = jnp.tensordot(
            self.spectra_weights_jnp,
            imgs_mono,
            axes=((1,), (0,)),
        )

        count_shape = (self.n_spectra,) + (1,) * (imgs_spectra.ndim - 1)
        return imgs_spectra * self.spectrum_counts.reshape(count_shape)


class _ProjApproxMixin:
    """Shared projection-approximation implementation."""

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        del H_slice

        delta_proj = jnp.sum(delta_theta, axis=0) * self.dz_eff
        beta_proj = jnp.sum(beta_theta, axis=0) * self.dz_eff

        transmission = jnp.exp(
            -1j * wavenum * delta_proj
            - wavenum * beta_proj
        )

        psi_in = jnp.sqrt(count) + 0j
        psi = psi_in * transmission
        psi = self.propagate(psi=psi, H=H_det, pad_value=psi_in)
        return self.detect_field(psi)


class _MultiSliceMixin:
    """Shared multislice implementation."""

    def forward_one(self, delta_theta, beta_theta, wavenum, count, H_det, H_slice):
        delta_ms = coarsen_object_slices(delta_theta, self.ms_pars)
        beta_ms = coarsen_object_slices(beta_theta, self.ms_pars)
        slab_thickness = self.ms_pars['slice_thickness']

        psi_in = jnp.sqrt(count) + 0j
        psi = psi_in * jnp.ones(
            self.field_shape,
            dtype=jnp.result_type(psi_in, 1j),
        )

        def transmit_slice(field, slice_pars):
            delta_slice, beta_slice = slice_pars
            transmission = jnp.exp(
                -1j * wavenum * delta_slice * slab_thickness
                - wavenum * beta_slice * slab_thickness
            )
            return field * transmission

        def slice_body(field, slice_pars):
            field = transmit_slice(field, slice_pars)
            field = self.propagate(
                psi=field,
                H=H_slice,
                pad_value=psi_in,
            )
            return field, None

        scan_body = jax.checkpoint(slice_body) if self.checkpoint_slices else slice_body

        if delta_ms.shape[0] > 1:
            psi, _ = lax.scan(
                scan_body,
                psi,
                (delta_ms[:-1], beta_ms[:-1]),
            )

        psi = transmit_slice(psi, (delta_ms[-1], beta_ms[-1]))
        psi = self.propagate(psi=psi, H=H_det, pad_value=psi_in)
        return self.detect_field(psi)


class ProjApproxPBI(_ProjApproxMixin, PBI_Base):
    """Projection-approximation PB-XPCI model for row or panel data."""


class MultiSlicePBI(_MultiSliceMixin, PBI_Base):
    """Multislice PB-XPCI model for row or panel data."""


# -----------------------------------------------------------------------------
# Input normalization and validation
# -----------------------------------------------------------------------------

def _as_spectrum_array(arr, name, n_spectra, dtype=DTYPE):
    """Normalize a scalar or per-spectrum input to `(n_spectra,)`."""
    arr = jnp.asarray(arr, dtype=dtype)

    if arr.ndim == 0:
        return jnp.full((n_spectra,), arr, dtype=dtype)

    arr = jnp.ravel(arr)

    if arr.shape[0] == 1:
        return jnp.full((n_spectra,), arr[0], dtype=dtype)

    if arr.shape[0] != n_spectra:
        raise ValueError(
            f'{name} must be scalar or have length n_spectra={n_spectra}. '
            f'Got shape {arr.shape}.'
        )

    return arr


def _as_spectra_weights(arr, n_energies, dtype=DTYPE):
    """
    Convert spectrum weights to shape `(n_spectra, n_energies)`.

    A 1D input defines one spectrum. `None` produces an identity matrix, so
    each output corresponds to one monochromatic energy. Weights are used as
    supplied and are not normalized.
    """
    if arr is None:
        return jnp.eye(n_energies, dtype=dtype)

    arr = jnp.asarray(arr, dtype=dtype)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(
            'spectra_weights must have shape (n_spectra, n_energies) or '
            f'(n_energies,). Got shape {arr.shape}.'
        )

    if arr.shape[1] != n_energies:
        raise ValueError(
            'spectra_weights second dimension must match n_energies. '
            f'Got {arr.shape[1]} and {n_energies}.'
        )

    return arr 


def _as_basis_coeff_array(arr, name, n_energies, dtype=DTYPE):
    """Normalize material coefficient arrays to `(nE, n_materials)`."""
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


def _as_material_map_stack(material_maps, object_ndim):
    """Normalize material maps to `(n_materials, *object_shape)`."""
    if isinstance(material_maps, (tuple, list)):
        arr = jnp.stack([jnp.asarray(m) for m in material_maps], axis=0)
    else:
        arr = jnp.asarray(material_maps)

    if arr.ndim == object_ndim:
        return arr[None, ...]

    return arr


def _check_material_maps(material_maps, n_materials, object_shape):
    """Validate material-map rank, material count, and spatial shape."""
    expected_ndim = len(object_shape) + 1

    if material_maps.ndim != expected_ndim:
        raise ValueError(
            'material_maps must have shape '
            f'(n_materials, {", ".join(map(str, object_shape))}). '
            f'Got shape {material_maps.shape}.'
        )

    if material_maps.shape[0] != n_materials:
        raise ValueError(
            f'Expected {n_materials} material maps from optical coefficients, '
            f'but got {material_maps.shape[0]}.'
        )

    if tuple(material_maps.shape[1:]) != tuple(object_shape):
        raise ValueError(
            f'Expected material map shape {(n_materials, *object_shape)}, '
            f'but got {material_maps.shape}.'
        )


def _as_int_tuple(value, ndim, name):
    """Broadcast an integer or validate an integer tuple of length `ndim`."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        values = (value,) * ndim

    if len(values) != ndim:
        raise ValueError(f'{name} must be scalar or have length {ndim}.')

    out = []
    for item in values:
        if int(item) != item or item < 0:
            raise ValueError(f'{name} entries must be nonnegative integers.')
        out.append(int(item))

    return tuple(out)


def _as_optional_float_tuple(value, ndim, name):
    """Broadcast a scalar/None or validate a per-axis optional float tuple."""
    if value is None:
        return (None,) * ndim

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        values = (value,) * ndim

    if len(values) != ndim:
        raise ValueError(f'{name} must be scalar or have length {ndim}.')

    out = []
    for item in values:
        if item is not None and item <= 0:
            raise ValueError(f'{name} entries must be positive or None.')
        out.append(None if item is None else float(item))

    return tuple(out)


def _infer_detector_size(native_size, requested_size, binning, axis_name):
    """Infer or validate one detector dimension."""
    if requested_size is None:
        if native_size % binning != 0:
            raise ValueError(
                f'Require native {axis_name} size % binning == 0 when the '
                f'detector {axis_name} size is inferred.'
            )
        return int(native_size // binning)

    if native_size < requested_size * binning:
        raise ValueError(
            f'Require native {axis_name} size >= detector size * binning. '
            f'Got {native_size} < {requested_size} * {binning}.'
        )

    return int(requested_size)


def normalize_data(data, sim):
    return data / sim.flat_field_signal[:,None,None,None]

    
# -----------------------------------------------------------------------------
# Rotation helpers
# -----------------------------------------------------------------------------

def make_rotation_locations_2d(shape, angle, dtype=DTYPE, spacing=(1.0, 1.0)):
    """
    Return interpolation coordinates for rotation in the physical z-x plane.

    `shape` is `(Nz, Nx)` and `spacing` is `(dz, dx)`. The y
    coordinate is unchanged because y is the CT rotation axis.
    """
    Nz, Nx = shape
    dz, dx = spacing

    cz = (Nz - 1) / 2
    cx = (Nx - 1) / 2

    z = (jnp.arange(Nz, dtype=dtype) - cz) * dz
    x = (jnp.arange(Nx, dtype=dtype) - cx) * dx
    zz, xx = jnp.meshgrid(z, x, indexing='ij')

    cos_t = jnp.cos(angle).astype(dtype)
    sin_t = jnp.sin(angle).astype(dtype)

    z_in = (cos_t * zz - sin_t * xx) / dz + cz
    x_in = (sin_t * zz + cos_t * xx) / dx + cx

    return jnp.stack([z_in, x_in], axis=0)


def rotate_image(image, angle, cval=0.0, spacing=(1.0, 1.0)):
    """Rotate a 2D image with bilinear interpolation and constant fill."""
    image = jnp.asarray(image)
    locs = make_rotation_locations_2d(
        shape=image.shape,
        angle=angle,
        spacing=spacing,
        dtype=image.real.dtype,
    )

    return map_coordinates(
        image,
        locs,
        order=1,
        mode='constant',
        cval=cval,
    )


def rotate_last_two_axes(arr, angle, spacing=(1.0, 1.0), cval=0.0):
    """Rotate every 2D plane across arbitrary leading dimensions."""
    arr = jnp.asarray(arr)

    if arr.ndim < 2:
        raise ValueError('arr must have at least two dimensions.')

    Nz, Nx = arr.shape[-2:]
    flat = arr.reshape((-1, Nz, Nx))

    rotated = jax.vmap(
        lambda image: rotate_image(
            image,
            angle=angle,
            cval=cval,
            spacing=spacing,
        ),
        in_axes=0,
        out_axes=0,
    )(flat)

    return rotated.reshape(arr.shape)


# -----------------------------------------------------------------------------
# Detector crop and block averaging
# -----------------------------------------------------------------------------

def prepare_detector(field_shape, field_spacing, det_shape, binning):
    """Precompute centered crop and block-averaging parameters."""
    field_shape = tuple(int(v) for v in field_shape)
    field_spacing = tuple(float(v) for v in field_spacing)
    det_shape = tuple(int(v) for v in det_shape)
    binning = tuple(int(v) for v in binning)

    ndim = len(field_shape)
    if not (
        len(field_spacing) == ndim
        and len(det_shape) == ndim
        and len(binning) == ndim
    ):
        raise ValueError('Detector geometry tuples must have matching lengths.')

    crop_shape = []
    crop_start = []
    crop_stop = []

    for native_N, det_N, up in zip(field_shape, det_shape, binning):
        if up < 1:
            raise ValueError('Detector binning factors must be positive integers.')

        N_crop = det_N * up
        if native_N < N_crop:
            raise ValueError('Native field must contain the requested detector crop.')

        i0 = (native_N - N_crop) // 2
        crop_shape.append(N_crop)
        crop_start.append(i0)
        crop_stop.append(i0 + N_crop)

    pars = dict(
        field_shape=field_shape,
        field_spacing=field_spacing,
        det_shape=det_shape,
        binning=binning,
        det_spacing=tuple(
            d * up
            for d, up in zip(field_spacing, binning)
        ),
        crop_shape=tuple(crop_shape),
        crop_start=tuple(crop_start),
        crop_stop=tuple(crop_stop),
    )

    return pars


def block_average_detector(img, det_pars):
    """Center-crop and block-average the final detector dimensions."""
    det_shape = det_pars['det_shape']
    binning = det_pars['binning']
    crop_start = det_pars['crop_start']
    crop_stop = det_pars['crop_stop']
    ndim = len(det_shape)

    crop = [slice(None)] * (img.ndim - ndim)
    crop.extend(
        slice(i0, i1)
        for i0, i1 in zip(crop_start, crop_stop)
    )
    img = img[tuple(crop)]

    for axis_local in reversed(range(ndim)):
        up = binning[axis_local]
        if up == 1:
            continue

        det_N = det_shape[axis_local]
        axis = img.ndim - ndim + axis_local
        new_shape = (
            img.shape[:axis]
            + (det_N, up)
            + img.shape[axis + 1:]
        )
        img = img.reshape(new_shape)
        img = jnp.mean(img, axis=axis + 1)

    return img


# -----------------------------------------------------------------------------
# Fresnel propagation and multislice helpers
# -----------------------------------------------------------------------------

def prepare_fresnel_transfer(shape, spacing, wavelens, propdist):
    """
    Precompute 1D or 2D Fresnel transfer functions.

    Returns an array with shape `(n_energies, *shape)`.
    """
    shape = tuple(int(v) for v in shape)
    spacing = tuple(float(v) for v in spacing)

    if len(shape) not in (1, 2):
        raise ValueError('Only 1D and 2D Fresnel propagation are supported.')

    if len(shape) != len(spacing):
        raise ValueError('shape and spacing must have matching lengths.')

    wavelens = jnp.atleast_1d(jnp.asarray(wavelens))
    f2 = jnp.zeros(shape, dtype=wavelens.dtype)

    for axis, (N, d) in enumerate(zip(shape, spacing)):
        freq = jnp.fft.fftfreq(N, d=d)
        reshape = [1] * len(shape)
        reshape[axis] = N
        f2 = f2 + freq.reshape(reshape) ** 2

    wavelength_shape = (wavelens.shape[0],) + (1,) * len(shape)
    return jnp.exp(
        -1j
        * jnp.pi
        * wavelens.reshape(wavelength_shape)
        * propdist
        * f2[None, ...]
    )


def prepare_multislice_pars(
    Nz=None,
    dz=None,
    multislice_thickness=None,
):
    """
    Precompute slab coarsening along the beam/z direction.

    Parameters
    ----------
    Nz:
        Number of native object samples along z.
    dz:
        Native object voxel spacing along z [m].
    multislice_thickness:
        Desired multislice slab thickness along z [m]. It defaults to `dz`
        and must be an integer multiple of `dz` that evenly divides
        `Nz * dz`.
    """
    if Nz is None or dz is None:
        raise ValueError('Nz and dz are required.')

    if dz <= 0:
        raise ValueError('dz must be positive.')

    if multislice_thickness is None:
        multislice_thickness = dz

    if multislice_thickness <= 0:
        raise ValueError('multislice_thickness must be positive.')

    ratio = multislice_thickness / dz
    slice_step = int(round(ratio))

    if not math.isclose(ratio, slice_step, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            'Require multislice_thickness to be an integer multiple of dz.'
        )

    if Nz % slice_step != 0:
        raise ValueError('Require Nz % slice_step == 0.')

    N_slices = Nz // slice_step
    thickness_eff = slice_step * dz

    return dict(
        slice_step=int(slice_step),
        N_slices=int(N_slices),
        voxel_dz=dz,
        multislice_thickness=thickness_eff,
        slice_thickness=thickness_eff,
    )


def coarsen_object_slices(arr, ms_pars):
    """Average adjacent beam/z samples into multislice slabs."""
    slice_step = ms_pars['slice_step']
    N_slices = ms_pars['N_slices']

    # The beam direction is always the first spatial axis z:
    #   (Nz, Nx) for single-row models
    #   (Nz, Ny, Nx) for full-panel models.
    if slice_step == 1:
        return arr

    arr = arr.reshape(
        N_slices,
        slice_step,
        *arr.shape[1:],
    )
    return jnp.mean(arr, axis=1)


# -----------------------------------------------------------------------------
# Detector PSF helpers
# -----------------------------------------------------------------------------

def lorentzian1D(x, fwhm, normalize=True):
    """Return a sampled 1D Lorentzian kernel."""
    gamma = fwhm / 2
    kernel = gamma / (jnp.pi * (x**2 + gamma**2))

    if normalize:
        kernel = kernel / jnp.sum(kernel)

    return kernel


def gaussian1D(x, fwhm, normalize=True):
    """Return a sampled 1D Gaussian kernel."""
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)

    if normalize:
        kernel = kernel / jnp.sum(kernel)

    return kernel


def get_psf_kernel_size(
    dx,
    fwhm,
    psf='lorentzian',
    rel_thresh=1e-3,
    max_kernel_width=8.0,
):
    """Choose a symmetric PSF half-width in pixels."""
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


def get_psf_kernel1D(
    dx,
    fwhm,
    psf='lorentzian',
    rel_thresh=1e-3,
    max_kernel_width=8.0,
):
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


def prepare_psf1D(
    N,
    dx,
    fwhm=None,
    psf='lorentzian',
    rel_thresh=1e-3,
    max_kernel_width=8.0,
):
    """Precompute one axis of separable detector PSF convolution."""
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
    offset = (K - 1) // 2

    return dict(
        kernel=kernel,
        G=G,
        K=K,
        pad=pad,
        fft_N=fft_N,
        o=offset,
    )


def prepare_psf(
    shape,
    spacing,
    fwhm,
    psf='lorentzian',
    rel_thresh=1e-3,
    max_kernel_width=8.0,
):
    """Precompute optional separable PSF convolution for every detector axis."""
    return tuple(
        prepare_psf1D(
            N=N,
            dx=d,
            fwhm=fwhm_i,
            psf=psf,
            rel_thresh=rel_thresh,
            max_kernel_width=max_kernel_width,
        )
        for N, d, fwhm_i in zip(shape, spacing, fwhm)
    )


def convolve1d_diy_precomp(img, psf_pars, axis=-1):
    """Convolve one array axis using a precomputed real FFT kernel."""
    if psf_pars is None:
        return img

    axis = axis % img.ndim
    img_moved = jnp.moveaxis(img, axis, -1)

    pad = psf_pars['pad']
    pad_width = [(0, 0)] * img_moved.ndim
    pad_width[-1] = (pad, pad)
    img_pad = jnp.pad(img_moved, pad_width, mode='edge')

    G = psf_pars['G']
    fft_N = psf_pars['fft_N']
    offset = psf_pars['o']
    N_pad = img_pad.shape[-1]

    F = jnp.fft.rfft(img_pad, n=fft_N, axis=-1)
    out = jnp.fft.irfft(F * G, n=fft_N, axis=-1)
    out = out[..., offset:offset + N_pad]

    if pad > 0:
        out = out[..., pad:-pad]

    return jnp.moveaxis(out, -1, axis)


def apply_psf_precomp(img, psf_pars):
    """Apply separable PSF convolution over the final detector dimensions."""
    ndim = len(psf_pars)

    for axis_local, axis_pars in enumerate(psf_pars):
        if axis_pars is None:
            continue
        axis = img.ndim - ndim + axis_local
        img = convolve1d_diy_precomp(img, axis_pars, axis=axis)

    return img


# -----------------------------------------------------------------------------
# Noise helper
# -----------------------------------------------------------------------------

def add_poisson_noise(imgs, key, scale=1.0):
    """
    Add Poisson noise to expected-count data.

    The returned array is `poisson(scale * imgs) / scale`.
    """
    return jax.random.poisson(key, scale * imgs, imgs.shape) / scale