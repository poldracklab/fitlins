import os

import pandas as pd
import nibabel as nb
from nilearn import plotting as nlp
import nistats as nis
from nipype.utils.filemanip import fname_presuffix, split_filename

from ..viz import plot_and_save, plot_corr_matrix, plot_contrast_matrix


def visualization(data, image_type, visualize):

    def _load_data(fname):
        _, _, ext = split_filename(fname)
        if ext == '.tsv':
            return pd.read_table(fname, index_col=0)
        elif ext in ('.nii', '.nii.gz', '.gii'):
            return nb.load(fname)
        raise ValueError("Unknown file type!")

    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_style('white')
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['image.interpolation'] = 'nearest'

    data = _load_data(data)
    out_name = fname_presuffix(data,
                               suffix='.' + image_type,
                               newpath=os.getcwd(),
                               use_ext=False)
    visualize(data, out_name)
    return out_name


def DesignPlot(data, image_type):
    def _visualize(data, out_name):
        from matplotlib import pyplot as plt
        plt.set_cmap('viridis')
        plot_and_save(out_name, nis.reporting.plot_design_matrix, data)
    return visualization(data, image_type, _visualize)


def DesignCorrelationPlot(data, image_type, contrast_info):
    def _visualize(data, out_name):
        contrast_matrix = pd.DataFrame({c['name']: c['weights'][0]
                                        for c in contrast_info})
        all_cols = list(data.columns)
        evs = set(contrast_matrix.index)
        if set(contrast_matrix.index) != all_cols[:len(evs)]:
            ev_cols = [col for col in all_cols if col in evs]
            confound_cols = [col for col in all_cols if col not in evs]
            data = data[ev_cols + confound_cols]
        plot_and_save(out_name, plot_corr_matrix,
                      data.drop(columns='constant').corr(),
                      len(evs))
    return visualization(data, image_type, _visualize)


def ContrastMatrixPlot(data, image_type, contrast_info, orientation):
    def _visualize(data, out_name):
        contrast_matrix = pd.DataFrame({c['name']: c['weights'][0]
                                        for c in contrast_info},
                                       index=data.columns)
        contrast_matrix.fillna(value=0, inplace=True)
        if 'constant' in contrast_matrix.index:
            contrast_matrix = contrast_matrix.drop(index='constant')
        plot_and_save(out_name, plot_contrast_matrix, contrast_matrix,
                      ornt=self.inputs.orientation)
    return visualization(data, image_type, _visualize)


def GlassBrainPlot(data, image_type, threshold, vmax, colormap):
    def _visualize(data, out_name):
        import numpy as np
        if vmax is None:
            abs_data = np.abs(data.get_fdata())
            pctile99 = np.percentile(abs_data, 99.99)
            if abs_data.max() - pctile99 > 10:
                vmax = pctile99
        nlp.plot_glass_brain(data, colorbar=True, plot_abs=False,
                             display_mode='lyrz', axes=None,
                             vmax=vmax, threshold=threshold,
                             cmap=colormap,
                             output_file=out_name)
    return visualization(data, image_type, _visualize)
