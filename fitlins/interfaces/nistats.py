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

from ..viz import plot_and_save, plot_corr_matrix


class FirstLevelModelInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    session_info = traits.Dict()
    contrast_info = traits.Dict()


class FirstLevelModelOutputSpec(TraitedSpec):
    estimate_maps = OutputMultiPath(File())
    contrast_maps = OutputMultiPath(File())
    design_matrix = File()
    design_matrix_plot = File()
    correlation_matrix_plot = File()
    contrast_matrix_plot = File()


class FirstLevelModel(SimpleInterface):
    def _run_interface(self, runtime):
        info = self.inputs.session_info

        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        events = pd.read_hdf(info['events'])
        confounds = pd.read_hdf(info['confounds'])

        mat = dm.make_design_matrix(
            frame_times=np.arange(vols) * info['repetition_time'],
            paradigm=events,
            add_regs=confounds,
            add_reg_names=confounds.columns,
            drift_model=None if 'Cosine00' in confounds.columns else 'cosine',
            )

        plt.set_cmap('viridis')
        plot_and_save('design.svg', nis.reporting.plot_design_matrix, mat)
        self._results['design_matrix_plot'] = os.path.join(runtime.cwd,
                                                           'design.svg')

        plot_and_save('correlation.svg', plot_corr_matrix,
                      mat.drop(columns='constant').corr(),
                      len(events.columns))
        self._results['correlation_matrix_plot'] = os.path.join(
            runtime.cwd, 'correlation.svg')

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        flm = level1.FirstLevelModel(mask=mask_file)
        flm.fit(img, design_matrices=mat)

        return runtime
