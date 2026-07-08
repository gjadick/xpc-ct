import numpy as np
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
    r_main: float = 400.0,
    r_ring: float = 250.0,
    r_smalls = (10, 20, 30, 40, 50),
    dtype=np.uint8,
):
    """
    Generate a test phantom.
    """
    if Ny is None:
        Ny = Nx

    r_smalls = np.asarray(r_smalls)
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