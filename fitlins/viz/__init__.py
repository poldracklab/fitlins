from matplotlib import pyplot as plt
from .corr import plot_corr_matrix
from .contrasts import plot_contrast_matrix


def plot_and_save(fname, plotter, *args, **kwargs):
    if (kwargs.get('axes'), kwargs.get('ax')) == (None, None):
        fig = plt.figure()
        axes = plt.gca()
        if 'axes' in kwargs:
            kwargs['axes'] = axes
        else:
            kwargs['ax'] = axes
    plotter(*args, **kwargs)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
