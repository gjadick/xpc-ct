"""
Material definitions and X-ray optics helpers.

Defines the `Material` class wrapper for PB-XPCI simulations,
which conveniently converts chemical composition definitions to 
complex refractive indices components or attenuation coefficients
as a function of energy.

Materials are defined by density and chemical composition via either:

    1. an xraydb-compatible chemical formula, e.g. 'H2O', 'Al', 'C5H8O2'
    2. a NIST-style mass-fraction composition string, e.g. 'H(0.111898)O(0.888102)'
    3. a mass-fraction dictionary, e.g. {'H': 0.111898, 'O': 0.888102}

Energies are in keV. Densities are in g/cm^3 (matching xraydb).
Lengths are in meters unless otherwise noted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import jax.numpy as jnp
import xraydb

from .nist_xcom import get_nist_xcom_entry


# Physical constants
H = 6.62607015e-34        # Planck constant [J s]
C = 299792458.0           # speed of light [m / s]
J_EV = 1.602176634e-19    # joules per eV


def wavelength(E_keV):
    """Convert photon energy [keV] to wavelength [m]."""
    E_keV = np.asarray(E_keV)
    return 1e-3 * H * C / (E_keV * J_EV)


def wavelength_jnp(E_keV):
    """JAX version of wavelength()."""
    E_keV = jnp.asarray(E_keV)
    return 1e-3 * H * C / (E_keV * J_EV)


def wavenumber(E_keV):
    """Convert photon energy [keV] to wavenumber [1/m]."""
    return 2 * np.pi / wavelength(E_keV)


def wavenumber_jnp(E_keV):
    """JAX version of wavenumber()."""
    return 2 * jnp.pi / wavelength_jnp(E_keV)


def energy_from_wavelength(wavelen):
    """Convert wavelength [m] to photon energy [keV]."""
    wavelen = np.asarray(wavelen)
    return 1e-3 * H * C / (wavelen * J_EV)


@dataclass
class Material:
    """
    X-ray material class with interpolated delta/beta lookup.

    Parameters
    ----------
    name:
        Material name used for labels and plotting.
    density:
        Material density [g/cm^3].
    formula:
        xraydb-compatible chemical formula. Example: 'H2O', 'Al', 'C5H8O2'.
    mass_fractions:
        Dictionary of elemental mass fractions. Values are normalized internally.
        Example: {'H': 0.111898, 'O': 0.888102}.
    E_min, E_max, dE:
        Energy grid [keV] used to precompute delta and beta.
    dtype:
        JAX dtype for precomputed JAX lookup arrays.

    Notes
    -----
    xraydb uses energy in eV and density in g/cm^3. This class exposes energy
    in keV for consistency with the rest of the PB-XPCI code.
    """

    name: str
    density: float
    formula: str | None = None
    mass_fractions: dict[str, float] | None = None
    E_min: float = 1.0
    E_max: float = 100.0
    dE: float = 1.0
    dtype: Any = jnp.float32

    energy_grid: np.ndarray = field(init=False, repr=False)
    delta_grid: np.ndarray = field(init=False, repr=False)
    beta_grid: np.ndarray = field(init=False, repr=False)

    energy_grid_jnp: jnp.ndarray = field(init=False, repr=False)
    delta_grid_jnp: jnp.ndarray = field(init=False, repr=False)
    beta_grid_jnp: jnp.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Validate inputs and precompute numpy/JAX delta-beta lookup grids."""
        if self.formula is None and self.mass_fractions is None:
            raise ValueError('Either formula or mass_fractions must be provided.')

        if self.formula is None:
            self.formula = mass_fractions_to_formula(self.mass_fractions)

        self.energy_grid = np.arange(self.E_min, self.E_max + self.dE, self.dE)
        self.delta_grid, self.beta_grid = self._delta_beta_xraydb(self.energy_grid)

        self.energy_grid_jnp = jnp.asarray(self.energy_grid, dtype=self.dtype)
        self.delta_grid_jnp = jnp.asarray(self.delta_grid, dtype=self.dtype)
        self.beta_grid_jnp = jnp.asarray(self.beta_grid, dtype=self.dtype)

    def _check_energy_bounds(self, E_keV):
        """Raise an error if requested energies are outside the precomputed grid."""
        E = np.asarray(E_keV)

        if np.any(E < self.E_min) or np.any(E > self.E_max):
            raise ValueError(
                f'Requested energy outside lookup range for {self.name}. '
                f'Valid range is [{self.E_min}, {self.E_max}] keV.'
            )

    def _delta_beta_xraydb(self, E_keV):
        """
        Compute delta and beta directly using xraydb.

        Parameters
        ----------
        E_keV:
            Energy or energies [keV].

        Returns
        -------
        delta, beta:
            numpy arrays containing refractive-index decrement and absorption
            component.
        """
        E_eV = 1e3 * np.asarray(E_keV, dtype=float)
        delta, beta, _ = xraydb.xray_delta_beta(self.formula, self.density, E_eV)
        return np.asarray(delta), np.asarray(beta)

    def delta_beta(self, E_keV):
        """
        Return interpolated delta and beta at energy E_keV.

        Parameters
        ----------
        E_keV:
            Scalar or array of energies [keV].

        Returns
        -------
        delta, beta:
            numpy arrays, or numpy scalars for scalar input.
        """
        self._check_energy_bounds(E_keV)

        delta = np.interp(E_keV, self.energy_grid, self.delta_grid)
        beta = np.interp(E_keV, self.energy_grid, self.beta_grid)

        return delta, beta

    def delta_beta_jnp(self, E_keV):
        """
        JAX version of delta_beta().
        """
        E_keV = jnp.asarray(E_keV, dtype=self.dtype)

        delta = jnp.interp(E_keV, self.energy_grid_jnp, self.delta_grid_jnp)
        beta = jnp.interp(E_keV, self.energy_grid_jnp, self.beta_grid_jnp)

        return delta, beta

    def mu(self, E_keV):
        """
        Return linear attenuation coefficient mu = 2 k beta [1/m].
        """
        _, beta = self.delta_beta(E_keV)
        return 2 * wavenumber(E_keV) * beta

    def mu_jnp(self, E_keV):
        """
        JAX version of mu().
        """
        _, beta = self.delta_beta_jnp(E_keV)
        return 2 * wavenumber_jnp(E_keV) * beta


def parse_mass_fraction_string(comp):
    """
    Parse a NIST-style elemental mass-fraction string.

    Parameters
    ----------
    comp:
        String like 'H(0.111898)O(0.888102)' or 'Ca(24.43)C(14.64)O(58.49)'.

    Returns
    -------
    mass_fractions:
        Dictionary mapping element symbols to normalized mass fractions.

    Notes
    -----
    The numbers do not need to sum to 1 or 100. They are normalized internally.
    """
    pattern = r'([A-Z][a-z]?)\(([0-9.eE+-]+)\)'
    matches = re.findall(pattern, comp)

    if not matches:
        raise ValueError(f'Could not parse mass-fraction string: {comp}')

    elems = [m[0] for m in matches]
    weights = np.asarray([float(m[1]) for m in matches], dtype=float)

    if np.any(weights < 0):
        raise ValueError('Mass fractions must be nonnegative.')

    if weights.sum() == 0:
        raise ValueError('Mass fractions must not sum to zero.')

    weights = weights / weights.sum()

    return dict(zip(elems, weights))


def mass_fractions_to_formula(mass_fractions, scale=100.0, fmt='.8g'):
    """
    Convert elemental mass fractions into an xraydb-compatible formula.

    Parameters
    ----------
    mass_fractions:
        Dictionary mapping element symbols to mass fractions.
    scale:
        Arbitrary scaling factor for atomic ratios. The absolute scale cancels.
    fmt:
        Format string for atomic coefficients.

    Returns
    -------
    formula:
        Formula string compatible with xraydb.xray_delta_beta().

    Example
    -------
    {'H': 0.111898, 'O': 0.888102} becomes something like
    'H66.6667O33.3333'.
    """
    if mass_fractions is None:
        raise ValueError('mass_fractions cannot be None.')

    elems = list(mass_fractions.keys())
    weights = np.asarray([mass_fractions[el] for el in elems], dtype=float)

    if np.any(weights < 0):
        raise ValueError('Mass fractions must be nonnegative.')

    if weights.sum() == 0:
        raise ValueError('Mass fractions must not sum to zero.')

    weights = weights / weights.sum()

    atomic_masses = np.asarray([xraydb.atomic_mass(el) for el in elems])
    mol = weights / atomic_masses
    mol_frac = mol / mol.sum()
    coeff = scale * mol_frac

    return ''.join(f'{el}{format(c, fmt)}' for el, c in zip(elems, coeff))


def material_from_mass_fraction_string(name, comp, density, **kwargs):
    """
    Construct a Material from a NIST-style mass-fraction string.

    Parameters
    ----------
    name:
        Material name.
    comp:
        Composition string like 'H(0.111898)O(0.888102)'.
    density:
        Density [g/cm^3].
    kwargs:
        Additional Material arguments, e.g. E_min, E_max, dE, dtype.
    """
    mass_fractions = parse_mass_fraction_string(comp)
    return Material(
        name=name,
        density=density,
        mass_fractions=mass_fractions,
        **kwargs,
    )


def material_from_nist_xcom(key, name=None, **kwargs):
    """
    Construct a Material from a NIST XCOM material or element entry.
    """
    entry = get_nist_xcom_entry(key)

    params = {
        'name': entry['name'] if name is None else name,
        'density': entry['density'],
    }

    if 'matcomp' in entry:
        params['mass_fractions'] = parse_mass_fraction_string(entry['matcomp'])
    elif 'symbol' in entry:
        params['formula'] = entry['symbol']
    else:
        raise ValueError(f'NIST XCOM entry {key!r} has no matcomp or symbol.')

    params.update(kwargs)

    return Material(**params)


def basis_delta_beta(materials, E_keV):
    """
    Precompute delta and beta for a material basis using numpy.

    Parameters
    ----------
    materials:
        Sequence of Material objects.
    E_keV:
        Scalar or array of energies [keV].

    Returns
    -------
    delta, beta:
        Arrays with shape (n_energies, n_materials).
    """
    E_keV = np.atleast_1d(np.asarray(E_keV, dtype=float))

    deltas = []
    betas = []

    for material in materials:
        delta, beta = material.delta_beta(E_keV)
        deltas.append(delta)
        betas.append(beta)

    delta = np.stack(deltas, axis=1)
    beta = np.stack(betas, axis=1)

    return delta, beta


def basis_delta_beta_jnp(materials, E_keV, dtype=jnp.float32):
    """
    Precompute delta and beta for a material basis using JAX.

    Parameters
    ----------
    materials:
        Sequence of Material objects.
    E_keV:
        Scalar or array of energies [keV].
    dtype:
        JAX dtype for returned arrays.

    Returns
    -------
    delta, beta:
        JAX arrays with shape (n_energies, n_materials).

    Notes
    -----
    Call before running forward model or recon to avoid repeated per-material 
    lookup inside the forward pass.
    """
    E_keV = jnp.atleast_1d(jnp.asarray(E_keV, dtype=dtype))

    deltas = []
    betas = []

    for material in materials:
        delta, beta = material.delta_beta_jnp(E_keV)
        deltas.append(delta)
        betas.append(beta)

    delta = jnp.stack(deltas, axis=1)
    beta = jnp.stack(betas, axis=1)

    return delta, beta


def basis_mu(materials, E_keV):
    """
    Precompute linear attenuation coefficients for a material basis using numpy.

    Returns
    -------
    mu:
        Array with shape (n_energies, n_materials), units [1/m].
    """
    _, beta = basis_delta_beta(materials, E_keV)
    k = np.atleast_1d(wavenumber(E_keV))[:, None]
    return 2 * k * beta


def basis_mu_jnp(materials, E_keV, dtype=jnp.float32):
    """
    Precompute linear attenuation coefficients for a material basis using JAX.

    Returns
    -------
    mu:
        JAX array with shape (n_energies, n_materials), units [1/m].
    """
    E_keV = jnp.atleast_1d(jnp.asarray(E_keV, dtype=dtype))
    _, beta = basis_delta_beta_jnp(materials, E_keV, dtype=dtype)
    k = wavenumber_jnp(E_keV)[:, None]
    return 2 * k * beta


# Small preset library.
MATERIAL_PRESETS = {
    'air': {
        'density': 0.0012,
        'mass_fraction_string': 'C(0.000124)N(0.755268)O(0.231781)Ar(0.012827)',
    },
    'water': {
        'density': 1.0,
        'mass_fraction_string': 'H(0.111898)O(0.888102)',
    },
    'adipose': {
        'density': 0.95,
        'mass_fraction_string': (
            'H(0.114000)C(0.598000)N(0.007000)O(0.278000)'
            'Na(0.001000)S(0.001000)Cl(0.001000)'
        ),
    },
    'tissue': {
        'density': 1.06,
        'mass_fraction_string': (
            'H(0.102000)C(0.143000)N(0.034000)O(0.708000)'
            'Na(0.002000)P(0.003000)S(0.003000)Cl(0.002000)K(0.003000)'
        ),
    },
    'bone': {
        'density': 1.92,
        'mass_fraction_string': (
            'H(0.034000)C(0.155000)N(0.042000)O(0.435000)'
            'Na(0.001000)Mg(0.002000)P(0.103000)S(0.003000)Ca(0.225000)'
        ),
    },
    'pmma': {
        'density': 1.19,
        'mass_fraction_string': 'H(0.080541)C(0.599846)O(0.319613)',
    },
    'aluminum': {
        'density': 2.70,
        'formula': 'Al',
    },
}


def make_material(name, **kwargs):
    """
    Construct a preset Material by name.

    Parameters
    ----------
    name:
        Preset material name. Currently supports:
        'air', 'water', 'adipose', 'tissue', 'pmma', 'aluminum'.
    kwargs:
        Optional overrides passed to Material, e.g. E_min, E_max, dE, dtype,
        or density.

    Returns
    -------
    material:
        Material object.
    """
    key = name.lower()

    if key not in MATERIAL_PRESETS:
        valid = ', '.join(MATERIAL_PRESETS)
        raise ValueError(f'Unknown material {name!r}. Valid options are: {valid}.')

    params = dict(MATERIAL_PRESETS[key])
    params.update(kwargs)

    comp = params.pop('mass_fraction_string', None)

    if comp is not None:
        return material_from_mass_fraction_string(
            name=key,
            comp=comp,
            **params,
        )

    return Material(name=key, **params)


def make_materials(names, **kwargs):
    """
    Construct multiple preset materials.

    Parameters
    ----------
    names:
        Iterable of material names.
    kwargs:
        Optional arguments passed to every make_material() call.

    Returns
    -------
    materials:
        Tuple of Material objects.
    """
    return tuple(make_material(name, **kwargs) for name in names)