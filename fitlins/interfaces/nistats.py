import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nb
import nistats as nis
import nistats.reporting
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    OutputMultiPath, File,
    traits, isdefined
    )

from ..viz import plot_and_save, plot_corr_matrix, plot_contrast_matrix


def build_contrast_matrix(contrast_spec, design_matrix,
                          identity=None):
    """Construct contrast matrix and return contrast type

    Parameters
    ----------
    contrast_spec : DataFrame
        Weight matrix with contrasts as rows and regressors as columns
        May have 'type' column indicating T/F test
    design_matrix : DataFrame
        GLM design matrix with regressors as columns and TR time as rows
    identity : list of strings
        Names of explanatory variables to ensure "identity" contrasts are
        provided.

    Returns
    -------
    contrast_matrix : DataFrame
        Weight matrix with contrasts as columns and regressors as rows.
        Regressors match columns (including order) of design matrix.
        Identity contrasts are included.
    contrast_types : Series
        Series of 'T'/'F' indicating the type of test for each column in
        the returned contrast matrix.
        Identity contrasts use T-tests.
    """
    # The basic spec is just a transposed matrix with an optional 'type'
    # column
    # We'll re-transpose and expand this matrix
    init_matrix = contrast_spec.drop('type', axis='columns').T
    init_types = contrast_spec['type'] if 'type' in contrast_spec \
        else pd.Series()

    if identity is None:
        identity = []
    all_cols = init_matrix.columns.tolist()
    all_cols.extend(set(identity) - set(all_cols))

    contrast_matrix = pd.DataFrame(index=design_matrix.columns,
                                   columns=all_cols, data=0)
    contrast_matrix.loc[tuple(init_matrix.axes)] = init_matrix

    contrast_types = pd.Series(index=all_cols, data='T')
    contrast_types[init_types.index] = init_types

    if identity:
        contrast_matrix.loc[identity, identity] = np.eye(len(identity))
        contrast_types[identity] = 'T'

    return contrast_matrix, contrast_types


class FirstLevelModelInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    session_info = traits.Dict()
    contrast_info = File(exists=True)


class FirstLevelModelOutputSpec(TraitedSpec):
    estimate_maps = OutputMultiPath(File())
    contrast_maps = OutputMultiPath(File())
    design_matrix = File()
    design_matrix_plot = File()
    correlation_matrix_plot = File()
    contrast_matrix_plot = File()


class FirstLevelModel(SimpleInterface):
    input_spec = FirstLevelModelInputSpec
    output_spec = FirstLevelModelOutputSpec

    def _run_interface(self, runtime):
        info = self.inputs.session_info

        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        events = pd.read_hdf(info['events'], key='events')
        confounds = pd.read_hdf(info['confounds'], key='confounds')
        if isdefined(self.inputs.contrast_info):
            contrast_spec = pd.read_hdf(self.inputs.contrast_info,
                                        key='contrasts')
        else:
            contrast_spec = pd.DataFrame()

        mat = dm.make_design_matrix(
            frame_times=np.arange(vols) * info['repetition_time'],
            paradigm=events.rename(columns={'condition': 'trial_type',
                                            'amplitude': 'modulation'}),
            add_regs=confounds,
            add_reg_names=confounds.columns.tolist(),
            drift_model=None if 'Cosine00' in confounds.columns else 'cosine',
            )

        exp_vars = events['condition'].unique().tolist()

        contrast_matrix, contrast_types = build_contrast_matrix(contrast_spec,
                                                                mat, exp_vars)

        plt.set_cmap('viridis')
        plot_and_save('design.svg', nis.reporting.plot_design_matrix, mat)
        self._results['design_matrix_plot'] = os.path.join(runtime.cwd,
                                                           'design.svg')

        plot_and_save('correlation.svg', plot_corr_matrix,
                      mat.drop(columns='constant').corr(), len(exp_vars))
        self._results['correlation_matrix_plot'] = os.path.join(
            runtime.cwd, 'correlation.svg')

        plot_and_save('contrast.svg', plot_contrast_matrix,
                      contrast_matrix.drop(['constant'], 'index'),
                      ornt='horizontal')
        self._results['contrast_matrix_plot'] = os.path.join(
            runtime.cwd, 'contrast.svg')

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        flm = level1.FirstLevelModel(mask=mask_file)
        flm.fit(img, design_matrices=mat)

        estimate_maps = []
        contrast_maps = []
        stat_fmt = os.path.join(runtime.cwd, '{}.nii.gz').format
        for contrast, ctype in zip(contrast_matrix, contrast_types):
            stat = flm.compute_contrast(contrast_matrix[contrast].values,
                                        {'T': 't', 'F': 'F'}[ctype])
            fname = stat_fmt(contrast)
            stat.to_filename(fname)
            if contrast in exp_vars:
                estimate_maps.append(fname)
            else:
                contrast_maps.append(fname)
        self._results['estimate_maps'] = estimate_maps
        self._results['contrast_maps'] = contrast_maps

        return runtime
