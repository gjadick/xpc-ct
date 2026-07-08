"""
2D CT simulation helpers + FBP reconstruction.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial

from .materials import make_material

_Water = make_material('water')  
_Air = make_material('air')  

DTYPE = jnp.float32


def mu_to_HU(mu, energy=10.0):
    mu_water = _Water.mu(energy)
    mu_air = _Air.mu(energy)
    return 1000 * (mu - mu_water) / (mu_water - mu_air)


def make_rotation_locations_2d(shape, angle, dtype=DTYPE):
    Ny, Nx = shape

    cy = (Ny - 1) / 2
    cx = (Nx - 1) / 2

    y = jnp.arange(Ny, dtype=dtype) - cy
    x = jnp.arange(Nx, dtype=dtype) - cx

    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cos_t = jnp.cos(angle).astype(dtype)
    sin_t = jnp.sin(angle).astype(dtype)

    y_in = cos_t * yy - sin_t * xx
    x_in = sin_t * yy + cos_t * xx

    y_in = y_in + cy
    x_in = x_in + cx

    return jnp.stack([y_in, x_in], axis=0)


def rotate_image(image, angle, cval=0.0):
    locs = make_rotation_locations_2d(
        shape = image.shape,
        angle = angle,
        dtype = image.real.dtype,
    )

    return map_coordinates(
        image,
        locs,
        order = 1,
        mode = 'constant',
        cval = cval,
    )


def make_rot_locs(thetas, Nx, dtype=DTYPE):
    c = (Nx - 1) / 2

    y, x = jnp.meshgrid(
        jnp.arange(Nx, dtype=dtype) - c,
        jnp.arange(Nx, dtype=dtype) - c,
        indexing='ij',
    )

    def one(theta):
        theta = theta.astype(dtype)
        ct = jnp.cos(theta)
        st = jnp.sin(theta)

        yy = ct*y + st*x + c
        xx = -st*y + ct*x + c

        return jnp.stack([yy, xx])

    return jax.vmap(one)(thetas.astype(dtype))


def rotate_locs(img, locs):
    return map_coordinates(
        img,
        locs,
        order=1,
        mode='constant',
        cval=0.0,
    )


def select_energy_array(arr, i, nE):
    if arr.ndim == 2:
        return arr

    elif arr.ndim == 3:
        assert arr.shape[0] == nE
        return arr[i]

    else:
        raise ValueError('Expected arr to have shape (Nx, Nx) or (nE, Nx, Nx).')


def get_recon_coords(N_matrix, FOV):
    x = (FOV / N_matrix) * jnp.arange((1 - N_matrix) / 2, N_matrix / 2, 1, dtype=DTYPE)
    X, Y = jnp.meshgrid(x, x, indexing='xy')
    return X, Y


def make_ramp_filter_fourier(
    N_channels,
    s,
    cutoff_frac=1.0,
    window=None,
):
    """
    Windowed Fourier-domain ramp filter for parallel-beam FBP.

    cutoff_frac:
        Fraction of Nyquist frequency to keep [range 0.0 to 1.0].
        1.0 = full ramp
        decreasing to 0.0 = smoother, more low-pass
    """
    freq = jnp.fft.rfftfreq(N_channels, d=s)
    f_nyq = 1 / (2 * s)
    f_cut = cutoff_frac * f_nyq

    ramp = jnp.abs(freq)

    x = jnp.clip(freq / f_cut, 0.0, 1.0)

    if window == 'hann':
        win = 0.5 * (1.0 + jnp.cos(jnp.pi * x))
        win = jnp.where(freq <= f_cut, win, 0.0)

    elif window == 'hamming':
        win = 0.54 + 0.46 * jnp.cos(jnp.pi * x)
        win = jnp.where(freq <= f_cut, win, 0.0)

    elif window == 'cosine':
        win = jnp.cos(0.5 * jnp.pi * x)
        win = jnp.where(freq <= f_cut, win, 0.0)

    elif window == 'hard':
        win = jnp.where(freq <= f_cut, 1.0, 0.0)

    elif window is None:
        win = jnp.ones_like(freq)

    else:
        raise ValueError(f'Unknown window: {window}')

    H = ramp * win

    return H.astype(DTYPE)
    

@partial(jax.jit, static_argnames=())
def pre_process_fft(sino_log, H):
    N = sino_log.shape[1]
    N_fft = 2 * (H.shape[0] - 1)

    sino_f = jnp.fft.rfft(sino_log, n=N_fft, axis=1)
    sino_filtered = jnp.fft.irfft(sino_f * H[None, :], n=N_fft, axis=1)

    return sino_filtered[:, :N].astype(DTYPE)

    
@partial(jax.jit, static_argnames=())
def do_fbp_vmap(sino, X, Y, dchannel, thetas):
    """
    Vectorized parallel-beam filtered backprojection.
    """
    N_proj, N_channels = sino.shape

    def backproject_one_proj(proj_row, theta_i):
        channel_targets = X * jnp.cos(theta_i) - Y * jnp.sin(theta_i)
        i_channel_matrix = channel_targets / dchannel + (N_channels - 1) / 2

        fbp_i = map_coordinates(
            proj_row,
            i_channel_matrix[None, ...],
            order=1,
            mode='constant',
            cval=0.0,
        )

        return jnp.nan_to_num(fbp_i)

    all_proj = jax.vmap(backproject_one_proj, in_axes=(0, 0), out_axes=0)(
        sino,
        thetas,
    )

    matrix = jnp.sum(all_proj, axis=0) * jnp.mean(jnp.diff(thetas))  # assumes equally spaced views
    return matrix.astype(DTYPE)


def get_recon(sino_log, N_matrix, FOV, s, thetas, ramp_window=None, ramp_cutoff=1.0):
    X, Y = get_recon_coords(N_matrix, FOV)
    H = make_ramp_filter_fourier(2 * sino_log.shape[1], s, window=ramp_window, cutoff_frac=ramp_cutoff)  # pad 2x

    sino_filtered = pre_process_fft(sino_log, H)
    recon = do_fbp_vmap(sino_filtered, X, Y, s, thetas)

    return recon