import numpy as np
from scipy.special import ndtr

def load_spectrum(filename, Emin=0, norm=True, dtype=np.float32):
    """
    Read spectrum from CSV file to a numpy array. 
    Format should be two columns: energies, weights
    """
    with open(filename, 'r') as f:
        spectrum = np.array([line.split(',') for line in f.readlines()[1:]], dtype=dtype).T
        
    spectrum = spectrum[:, spectrum[0] > Emin]

    if norm:
        spectrum[1] /= np.sum(spectrum[1])
        
    return spectrum


def resample_spectrum(spectrum, dx=1.0, norm=True):
    """
    Re-sample an energy spectrum with new energy bin width `dx`.    
    """
    x, y = spectrum
    x2 = np.arange(x.min(), x.max()+dx, dx)
    y2 = np.interp(x2, x, y) 
    if norm:
        y2 /= dx
    return np.array([x2, y2]) 


def effective_energies(energies, spectra_weights):
    """
    Calculate the effective energy of each spectrum.
    """
    energies = np.asarray(energies)
    spectra_weights = np.asarray(spectra_weights)
    return np.sum(spectra_weights * energies, axis=1) / np.sum(spectra_weights, axis=1)
    

def split_spectrum_gaussian(spectrum, Et, gap_keV=0.0, fwhm_keV=3.0):
    """
    Softly split energy spectrum into low- and high-energy bins with optional dead band
    centered at a threshold energy, assuming Gaussian detector response.
    
    Inputs
    ------
    spectrum ~ [energies, IE] : array-like, shape (2, N,)
        energies = Energy bin centers [keV], assumed ascending.
        IE = Incident photon counts per energy bin.
    Et : float or array-like
        Threshold energy [keV] defining the center of the exclusion band between
        the low- and high-energy bins.
    gap_keV : float, optional
        Width of the exclusion band [keV] centered at `E_thresh`.
    fwhm_keV : float, optional
        Detector energy resolution full width at half maximum [keV], used to
        define the Gaussian soft roll-off at the threshold boundaries.
    
    Outputs
    -------
    s_low : jax.Array, shape (N,)
        Low-energy portion of the spectrum after soft thresholding.
    s_high : jax.Array, shape (N,)
        High-energy portion of the spectrum after soft thresholding.
    """
    energies, IE = spectrum
    
    Et = np.atleast_1d(Et)
    energies = np.asarray(energies)
    IE = np.asarray(IE)
    sigma = np.asarray(fwhm_keV / 2.355)

    E_lo = Et[..., None] - 0.5 * gap_keV
    E_hi = Et[..., None] + 0.5 * gap_keV
    z_lo = (E_lo - energies[None, :]) / sigma
    z_hi = (E_hi - energies[None, :]) / sigma
    c_lo = ndtr(z_lo)   # P(meas < E_lo | true E)
    c_hi = ndtr(z_hi)   # P(meas < E_hi | true E)

    w_low = c_lo
    w_high = 1.0 - c_hi

    s_low = w_low * IE[None, :]
    s_high = w_high * IE[None, :]

    return np.array([
        [energies, s_low[0]],
        [energies, s_high[0]]
    ])


def split_spectrum(energies, Et, gap_keV=0.0, fwhm_keV=0.0):
    counts = jnp.ones(energies.size)
    return split_spectrum_gaussian(energies, counts, Et, gap_keV, fwhm_keV)


def split_spectrum_multi(energies, thresholds, gap=0.0, fwhm=0.0):
    thresholds = sorted(jnp.atleast_1d(thresholds))
    etas = []
    envelope = jnp.ones(energies.size)
    for Et in thresholds:
        eta_lo, eta_hi = split_spectrum(energies, Et, gap, fwhm)
        etas.append(eta_lo * envelope)
        envelope = eta_hi
    etas.append(envelope)
    return jnp.array(etas)

    