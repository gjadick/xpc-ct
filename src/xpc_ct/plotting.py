import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from cycler import cycler

    
custom_colors = ['#0072B2', '#009E73', '#F0E442', '#D55E00', '#CC79A7']   # color-blind friendly

plt.rcParams.update({
    'figure.dpi': 150,  #300,
    'font.size':10,
    'axes.titlesize':10,
    'axes.labelsize':8,
    'axes.linewidth': .5,
    'axes.prop_cycle': cycler('color', custom_colors),
    'xtick.top': True, 
    'ytick.right': True, 
    'xtick.direction': 'in', 
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'legend.fontsize': 8,
    'lines.linewidth':1,
    'image.cmap':'gray',
    # 'text.usetex': True,
    # 'font.family': 'serif'
})

def add_scalebar(ax, width_px, dx_per_px, real_units, color='k', loc='lower left', sv=1):
    """
    Add a scale bar to a matplotlib image axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw the scale bar.

    width_px : float
        Width of the scale bar in image pixels.

    dx_per_px : float
        Physical size represented by one image pixel, in units of `real_units`.

    real_units : str
        Unit label for the physical scale bar, e.g. 'um', 'mm', or 'cm'.

    color : str, optional
        Scale bar and text color. Default is 'k' for black.

    loc : str, optional
        Location of the scale bar. Default is 'lower left'.

    sv : float, optional
        Vertical thickness of the scale bar in pixels. Default is 1.
    """
    label = f'{dx_per_px * width_px:.0f} {real_units}'
    scalebar = AnchoredSizeBar(
        ax.transData,
        width_px,
        label,
        loc,
        size_vertical=sv,
        frameon=False,
        color=color,
    )
    ax.add_artist(scalebar)
    
