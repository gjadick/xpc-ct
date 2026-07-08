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
        Image data in counts/normalized counts, shape (N_angles, Nx, Ny).

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
        Retrieved projected thickness, shape (N_angles, Nx, Ny).
    """

    data = np.asarray(data, dtype=np.float64) / I0
    
    if data.ndim != 3:
        raise ValueError(f'data must have shape (N_angles, Nx, Ny), got shape {data.shape}.')
    
    N_angles, Nx, Ny = data.shape

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


def tie_matdecomp(
    data,
    energies,
    propdist,
    dx,
    basis_materials,
    I0=1.0
):
    """
    Linearized TIE/PBI material-decomposition solver (Schaff et al. 2020)

    Parameters
    ----------
    data : array
        Image data in counts/normalized counts, shape (N_bins, N_angles, Nx, Ny).

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
        Material thickness maps, shape (N_materials, N_angles, Nx, Ny).
    """

    data = np.asarray(data, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)

    if data.ndim != 4:
        raise ValueError(f'data must have shape (N_bins, N_angles, Nx, Ny), got shape {data.shape}.')

    N_bins, N_angles, Nx, Ny = data.shape
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
    xF = np.einsum(
        'xymn,axyn->axym',
        np.linalg.pinv(ATA),
        ATb,
    )

    Ts = np.real(np.fft.ifftn(xF, axes=(1, 2)))
    Ts = np.moveaxis(Ts, -1, 0)

    return jnp.asarray(Ts, dtype=jnp.float32)


