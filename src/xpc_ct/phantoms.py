import numpy as np
import jax.numpy as jnp
from scipy.ndimage import zoom, gaussian_filter

def resample_smooth_2D(phantom, dx, dx0, k_blur=1, thresh=0.35):
    """
    Smoothly upsample a 2D phantom, represented by discrete material maps.
    """
    
    Ny0, Nx0 = phantom.shape
    Nx = int(round(Nx0 * dx0 / dx))
    Ny = int(round(Ny0 * dx0 / dx))
    
    best_score = np.full([Ny, Nx], -np.inf, dtype=np.float32)
    label_img = np.zeros([Ny, Nx], dtype=np.uint8)  # the upsampled phantom!
    
    for m in np.unique(phantom[phantom > 0]):
    
        sigma = k_blur * dx0 / dx
        pad = int(np.ceil(k_blur))
    
        mask = phantom == m
        up_mask = zoom(mask, (Nx/Nx0, Ny/Ny0), order=0)
        
        blur = gaussian_filter(up_mask.astype(np.float32), sigma=sigma)  # temp convert to float
        score = blur - thresh
    
        # update the phantom
        score_view = best_score 
        label_view = label_img  
    
        update = score > score_view
        update &= score > 0
    
        score_view[update] = score[update]
        label_view[update] = m
    
    return label_img


def make_circle_cluster_phantom(
    Nx: int = 1024,
    Ny: int | None = None,
    dx: float = 1.0,
    p_main: float = 0.4,
    p_ring: float = 0.25,
    p_smalls = (0.01, 0.02, 0.03, 0.04, 0.05),
    dtype=np.uint8,
):
    """
    Generate a test phantom.
    """
    if Ny is None:
        Ny = Nx

    r_main = p_main * Nx
    r_ring = p_ring * Nx
    r_smalls = np.array(p_smalls) * Nx

    n_small = len(r_smalls)

    y = (np.arange(Ny) - (Ny - 1) / 2) * dx
    x = (np.arange(Nx) - (Nx - 1) / 2) * dx
    yy, xx = np.meshgrid(y, x, indexing='ij')

    phantom = np.zeros((Ny, Nx))

    main_mask = xx**2 + yy**2 <= r_main**2
    phantom = np.where(main_mask, 1.0, phantom)

    angles = np.linspace(0.0, 2 * np.pi, n_small, endpoint=False)

    for theta, r_small in zip(angles, r_smalls):
        xc = r_ring * np.cos(theta)
        yc = r_ring * np.sin(theta)
        small_mask = (xx - xc)**2 + (yy - yc)**2 <= r_small**2
        phantom = np.where(small_mask, 2.0, phantom)

    return phantom.astype(dtype)



def rotated_coords(X, Y, x0=0.0, y0=0.0, theta=0.0):
    ct = jnp.cos(theta)
    st = jnp.sin(theta)

    Xc = X - x0
    Yc = Y - y0

    Xr =  ct * Xc + st * Yc
    Yr = -st * Xc + ct * Yc

    return Xr, Yr


def ellipsoid_thickness(X, Y, x0=0.0, y0=0.0, a=0.2, b=0.1, theta=0.0, tmax=1.0):
    Xr, Yr = rotated_coords(X, Y, x0=x0, y0=y0, theta=theta)

    rho2 = (Xr / a)**2 + (Yr / b)**2
    inside = rho2 <= 1.0

    return tmax * jnp.where(inside, jnp.sqrt(jnp.clip(1.0 - rho2, 0.0)), 0.0)


def add_ellipsoids(X, Y, ellipsoids):
    tmap = jnp.zeros_like(X)

    for e in ellipsoids:
        tmap = tmap + ellipsoid_thickness(X, Y, **e)

    return tmap


def cylinder_tmap(X, Y, tmax=1.0, x0=0.0):
    FOV = jnp.max(X) - jnp.min(X)
    radius = 0.4 * FOV

    rho2 = ((X - x0) / radius)**2
    inside = rho2 <= 1.0

    return tmax * jnp.where(inside, jnp.sqrt(jnp.clip(1.0 - rho2, 0.0)), 0.0)


def ellipsoids_tmap(X, Y, tmax=1.0, N=4, y=0.0, v=2.0, h=0.5, a_min=0.5, a_max=1):
    FOV = jnp.max(X) - jnp.min(X)
    
    pars = {
        'y0': -y * FOV,
        'theta': 0.0, 
    }
    scales = np.linspace(a_min, a_max, N)[::-1]
    
    x0s = FOV * np.linspace(-h, h, N) / 2.0
    a0s = (0.5 * FOV / (2 * N)) * scales
    ts =  tmax * scales

    ellipsoids = [
        dict(x0=x, a=a, b=v*a, tmax=t, **pars) for (x, a, t) in zip(x0s, a0s, ts)
    ]

    tmap = add_ellipsoids(X, Y, ellipsoids)
    cyl_mask = cylinder_tmap(X, Y, tmax=1.0) > 0

    return jnp.where(cyl_mask, tmap, 0.0)