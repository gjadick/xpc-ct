import numpy as np
import jax.numpy as jnp

from .materials import make_material, basis_delta_beta, basis_mu

_Water = make_material('water')  # default material


def paganin_pr(
    data,
    energy,
    propdist,
    dx,
    material=_Water,
    I0=1.0
):
    """
    Paganin single-material phase retrieval.

    Parameters
    ----------
    data : array
        Image data in counts/normalized counts, shape (N_angles, Ny, Nx).

    energy : float
        Energy in keV.

    propdist : float
        Propagation distance in meters.

    dx : float
        Pixel size in meters.

    material : Material
        Assumed single material for the object (default is water).

    I0 : float
        Data normalization factor (default is 1.0, assumes already normalized).

    Returns
    -------
    T : jnp.ndarray
        Retrieved projected thickness, shape (N_angles, Ny, Nx).
    """

    data = np.asarray(data, dtype=np.float64) / I0
    
    if data.ndim != 3:
        raise ValueError(f'data must have shape (N_angles, Ny, Nx), got shape {data.shape}.')
    
    N_angles, Ny, Nx = data.shape

    delta, _ = material.delta_beta(energy)
    mu = material.mu(energy)

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    f2 = fx[:, None]**2 + fy[None, :]**2

    H = mu / (
        mu + 4 * np.pi**2 * propdist * delta * f2
    )

    img_filt = np.fft.ifftn(
        np.fft.fftn(data, axes=(1, 2)) * H[None, :, :],
        axes=(1, 2),
    )

    T = -np.log(np.real(img_filt)) / mu

    return jnp.asarray(T, dtype=jnp.float32)


def schaff_md(
    data,
    energies,
    propdist,
    dx,
    basis_materials,
    I0=1.0,
    rcond=0.0
):
    """
    Linearized TIE/PBI material-decomposition solver (Schaff et al. 2020)

    Parameters
    ----------
    data : array
        Image data in counts/normalized counts, shape (N_bins, N_angles, Ny, Nx).

    energies : array
        Energies in keV, shape (N_bins,).

    propdist : float
        Propagation distance in meters.

    dx : float
        Pixel size in meters.

    basis_materials : sequence
        Assumed basis materials. 

    I0 : float
        Data normalization factor (default is 1.0, assumes already normalized).

    Returns
    -------
    Ts : jnp.ndarray
        Material thickness maps, shape (N_materials, N_angles, Ny, Nx).
    """

    data = np.asarray(data, dtype=np.float64) / I0
    energies = np.asarray(energies, dtype=np.float64)

    if data.ndim != 4:
        raise ValueError(f'data must have shape (N_bins, N_angles, Ny, Nx), got shape {data.shape}.')

    N_bins, N_angles, Ny, Nx = data.shape
    n_materials = len(basis_materials)

    delta, _ = basis_delta_beta(basis_materials, energies)
    mu = basis_mu(basis_materials, energies)

    # Spatial-frequency coordinates.
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    f2 = fx[:, None]**2 + fy[None, :]**2

    # Linearized PBI operator:
    # A_xycm = mu_cm + 4 pi^2 z delta_cm |f_xy|^2
    A = (
        mu[None, None, :, :]
        + 4 * np.pi**2 * propdist * delta[None, None, :, :] * f2[:, :, None, None]
    )

    # Right-hand side in Fourier domain.
    b = np.fft.fftn(-np.log(data), axes=(2, 3))

    # Frequency-wise normal equations.
    ATA = np.einsum('xycm,xycn->xymn', A, A)
    ATb = np.einsum('xycm,caxy->axym', A, b)

    # Solve for material thickness Fourier coefficients.
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    xF = np.einsum(
        'xymc,caxy->axym',
        A_pinv,
        b,
    )

    Ts = np.real(np.fft.ifftn(xF, axes=(1, 2)))
    Ts = np.moveaxis(Ts, -1, 0)

    return jnp.asarray(Ts, dtype=jnp.float32)


def beltran_pr(
    data,
    energy,
    propdist,
    dx,
    total_thickness,
    materials,
    I0=1.0
):
    """
    Beltran generalization of Paganin's method for multi-material phase retrieval from a single image. Requires knowledge of total thickness of the object.

    Parameters
    ----------
    data : array
        Propagation-based image data, shape (N_angles, Ny, Nx).

    energy : float
        Energy in keV.

    propdist : float
        Propagation distance in meters.

    dx : float
        Pixel size in meters.

    total_thickness : array
        Total projected object thickness in meters. Must broadcast to the
        shape of `data`.

    materials : sequence
        Material objects. The first material is the encasing material and
        all remaining materials are treated as embedded materials.

    I0 : float or array
        Data normalization factor.

    Returns
    -------
    Ts : jnp.ndarray
        Retrieved material thickness maps, shape
        (N_materials, N_angles, Ny, Nx). The first map is the residual
        encasing-material thickness.
    """

    data = np.asarray(data, dtype=np.float64) / I0

    if data.ndim != 3:
        raise ValueError(
            f'data must have shape (N_angles, Ny, Nx), got shape {data.shape}.'
        )

    if len(materials) < 2:
        raise ValueError('materials must contain an encasing material and at least one embedded material.')

    try:
        A = np.broadcast_to(
            np.asarray(total_thickness, dtype=np.float64),
            data.shape,
        )
    except ValueError as exc:
        raise ValueError(
            f'total_thickness must broadcast to data shape {data.shape}.'
        ) from exc

    _, Ny, Nx = data.shape

    delta, _ = basis_delta_beta(materials, energy)
    mu = basis_mu(materials, energy)

    delta = delta[0]
    mu = mu[0]

    ddelta = delta[1:] - delta[0]
    dmu = mu[1:] - mu[0]

    if np.any(np.isclose(dmu, 0.0)):
        raise ValueError('Embedded and encasing materials must have different attenuation coefficients.')

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    f2 = fx[:, None]**2 + fy[None, :]**2

    H = 1 / (1 + (4 * np.pi**2 * propdist * f2[:, :, None] * ddelta) / dmu)

    rhs = data * np.exp(mu[0] * A)

    q = np.fft.ifftn(
        np.fft.fftn(rhs, axes=(1, 2))[..., None] * H[None, ...],
        axes=(1, 2),
    )

    q = np.maximum(np.real(q), np.finfo(np.float64).tiny)
    T_embedded = -np.log(q) / dmu
    T_embedded = np.moveaxis(T_embedded, -1, 0)

    T_encasing = A - np.sum(T_embedded, axis=0)
    Ts = np.concatenate([T_encasing[None, ...], T_embedded], axis=0)

    return jnp.asarray(Ts, dtype=jnp.float32)