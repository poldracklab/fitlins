#!/usr/bin/env python
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'


def plot_contrast_matrix(contrast_matrix, ornt='vertical', ax=None):
    """ Plot correlation matrix

    Parameters
    ----------
    mat : DataFrame
        Design matrix with columns consisting of explanatory variables followed
        by confounds
    n_evs : int
        Number of explanatory variables to separate from confounds
    partial : {'upper', 'lower', None}, optional
        Plot matrix as upper triangular (default), lower triangular or full

    Returns
    -------
    ax : Axes
        Axes containing plot
    """

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if ornt == 'horizontal':
        contrast_matrix = contrast_matrix.T

    vmax = np.abs(contrast_matrix.values).max()

    # Use a red/blue (+1/-1) diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(contrast_matrix, vmin=-vmax, vmax=vmax, square=True,
                linewidths=0.5, cmap=cmap,
                cbar_kws={'shrink': 0.5, 'orientation': ornt,
                          'ticks': np.linspace(-vmax, vmax, 5)},
                ax=ax)

    # Variables along top and left
    ax.xaxis.tick_top()
    xtl = ax.get_xticklabels()
    ax.set_xticklabels(xtl, rotation=90)

    return ax
