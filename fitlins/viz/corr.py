#!/usr/bin/env python
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'


def plot_corr_matrix(mat, n_evs, partial='upper', ax=None):
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
        ax = plt.figure().subplot(1, 1, 1)

    # For a heatmap mask, 0 = show, 1 = hide
    mask = None if partial is None else np.ones_like(mat, dtype=bool)
    if partial == 'upper':
        mask[np.triu_indices_from(mask)] = False
    elif partial == 'lower':
        mask[np.tril_indices_from(mask)] = False
    elif partial is not None:
        raise ValueError("partial must be 'upper' or 'lower' or None")

    # Use a red/blue (+1/-1) diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(mat, vmin=-1, vmax=1, mask=mask, square=True,
                linewidths=0.5, cmap=cmap, cbar_kws={'shrink': 0.5}, ax=ax)

    # Separate EV-EV, EV-Confound, Confound-Confound correlation sections with
    # black line
    if partial == 'upper':
        ax.hlines([n_evs], n_evs, len(mat))
        ax.vlines([n_evs], 0, n_evs)
    elif partial == 'lower':
        ax.hlines([n_evs], 0, n_evs)
        ax.vlines([n_evs], n_evs, len(mat))
    else:
        ax.hlines([n_evs], 0, len(mat))
        ax.vlines([n_evs], 0, len(mat))

    # Label with variable names
    if partial == 'upper':
        # Upper: variables along top, and following the diagonal to the left
        ax.xaxis.tick_top()
        xtl = ax.get_xticklabels()
        ax.set_xticklabels(xtl, rotation=90)
        ax.set_yticklabels([])
        for i, var in enumerate(mat.columns):
            ax.text(i - 0.2, i + 0.6, var, ha="right", va="center")
    elif partial == 'lower':
        # Lower: Variables along left and following the diagonal to the top
        ax.set_xticklabels([])
        for i, var in enumerate(mat.columns):
            ax.text(i + 0.6, i - 0.2, var, ha="center", va="bottom", rotation='90')
    else:
        # Full: Variables along top and left
        ax.xaxis.tick_top()
        xtl = ax.get_xticklabels()
        ax.set_xticklabels(xtl, rotation=90)

    return ax
