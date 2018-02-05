from matplotlib import pyplot as plt
from .corr import plot_corr_matrix


def plot_and_save(fname, plotter, *args, **kwargs):
    fig = plt.figure()
    plotter(*args, **kwargs)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
