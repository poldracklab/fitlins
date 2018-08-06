import pandas as pd
import nibabel as nb
from nilearn import plotting as nlp
import nistats as nis
import nistats.reporting  # noqa: F401

from nipype.interfaces.base import (
    SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
    )
from nipype.utils.filemanip import fname_presuffix, split_filename

from ..viz import plot_and_save, plot_corr_matrix, plot_contrast_matrix


class VisualizationInputSpec(BaseInterfaceInputSpec):
    data = File(mandatory=True, desc='Data file to visualize')
    image_type = traits.Enum('svg', 'png', mandatory=True)


class VisualizationOutputSpec(TraitedSpec):
    figure = File(desc='Visualization')


class Visualization(SimpleInterface):
    input_spec = VisualizationInputSpec
    output_spec = VisualizationOutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set_style('white')
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['image.interpolation'] = 'nearest'

        data = self._load_data(self.inputs.data)
        out_name = fname_presuffix(self.inputs.data,
                                   suffix='.' + self.inputs.image_type,
                                   newpath=runtime.cwd,
                                   use_ext=False)
        self._visualize(data, out_name)
        self._results['figure'] = out_name
        return runtime

    def _load_data(self, fname):
        _, _, ext = split_filename(fname)
        if ext == '.tsv':
            return pd.read_table(fname, index_col=0)
        elif ext in ('.nii', '.nii.gz', '.gii'):
            return nb.load(fname)
        raise ValueError("Unknown file type!")


class DesignPlot(Visualization):
    def _visualize(self, data, out_name):
        from matplotlib import pyplot as plt
        plt.set_cmap('viridis')
        plot_and_save(out_name, nis.reporting.plot_design_matrix, data)


class DesignCorrelationPlotInputSpec(VisualizationInputSpec):
    explanatory_variables = traits.Range(
        low=0, desc='Number of explanatory variables')


class DesignCorrelationPlot(Visualization):
    input_spec = DesignCorrelationPlotInputSpec

    def _visualize(self, data, out_name):
        plot_and_save(out_name, plot_corr_matrix,
                      data.drop(columns='constant').corr(),
                      self.inputs.explanatory_variables)


class ContrastMatrixPlotInputSpec(VisualizationInputSpec):
    orientation = traits.Enum('horizontal', 'vertical', usedefault=True,
                              desc='Display orientation of contrast matrix')


class ContrastMatrixPlot(Visualization):
    input_spec = ContrastMatrixPlotInputSpec

    def _visualize(self, data, out_name):
        if 'constant' in data.index:
            data = data.drop(index='constant')
        plot_and_save(out_name, plot_contrast_matrix, data,
                      ornt=self.inputs.orientation)


class GlassBrainPlot(Visualization):
    def _visualize(self, data, out_name):
        nlp.plot_glass_brain(data, colorbar=True, plot_abs=False,
                             display_mode='lyrz', axes=None,
                             output_file=out_name)
