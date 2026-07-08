import numpy as np
from skimage.metrics import structural_similarity 

def calc_ssim(x, y, data_range=1.0, mask=None):
    if mask is not None:
        x = x[mask]
        y = y[mask]
    return structural_similarity(x, y, data_range=data_range)
    
def calc_rmse(x, y, mask=None):
    if mask is not None:
        x = x[mask]
        y = y[mask]
    return np.sqrt(np.mean((x-y)**2))