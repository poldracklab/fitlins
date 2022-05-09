import csv
from glob import glob

from nipype.interfaces.fsl.utils import Text2Vest
from nipype.interfaces.fsl.model import FILMGLS, ContrastMgr
import pandas as pd

from .nistats import FirstLevelModel, _flatten, prepare_contrasts


class FirstLevelModel(FirstLevelModel):

    def _run_interface(self, runtime):
        spec = self.inputs.spec
        fsl_txt = self.inputs.design_matrix.replace('.tsv', '_fsl.txt')
        fsl_mat = self.inputs.design_matrix.replace('.tsv', '_fsl.mat')
        fsl_tcon_txt = self.inputs.design_matrix.replace('.tsv', '_tcon.txt')
        fsl_tcon_mat = self.inputs.design_matrix.replace('.tsv', '_tcon.mat')
        fsl_fcon_txt = self.inputs.design_matrix.replace('.tsv', '_fcon.txt')
        fsl_fcon_mat = self.inputs.design_matrix.replace('.tsv', '_fcon.mat')
        # convert matrix to fsl matrix
        mat = pd.read_csv(self.inputs.design_matrix, delimiter='\t', index_col=0)
        mat.to_csv(fsl_txt, sep=" ", header=False, index=False)
        _ = Text2Vest(in_file=fsl_txt, out_file=fsl_mat).run()

        contrast_metadata = []
        t_contrasts = []
        f_contrasts = []
        for name, weights, cont_ents, contrast_test in prepare_contrasts(
            spec['contrasts'], mat.columns
        ):
            
            contrast_metadata.append(
                {
                    "name": spec['name'],
                    "level": spec['level'],
                    "stat": contrast_test,
                    **cont_ents,
                }
            )
            if contrast_test == "t":
                t_contrasts.append(weights[0])
            elif contrast_test == "F":
                # TODO: DO NOT KNOW HOW TO HANDLE
                f_contrasts.append(weights)

        # TODO: determine reasonable threshold/mask?
        model = FILMGLS(
            in_file=self.inputs.bold_file,
            design_file=fsl_mat,
        )

        if t_contrasts:
            with open(fsl_tcon_txt, 'w+') as f_tcon:
                wr = csv.writer(f_tcon, delimiter=" ")
                wr.writerows(t_contrasts)
            
            _ = Text2Vest(in_file=fsl_tcon_txt, out_file=fsl_tcon_mat).run()

            model.inputs.tcon_file = fsl_tcon_mat
        
        # TODO: DO NOT KNOW HOW TO HANDLE
        if f_contrasts:
            with open(fsl_fcon_txt, 'w+') as f_fcon:
                wr = csv.writer(f_tcon, delimiter=" ")
                wr.writerows(f_contrasts)
            
            _ = Text2Vest(in_file=fsl_fcon_txt, out_file=fsl_fcon_mat).run()

            model.inputs.fcon_file = fsl_fcon_mat

        model_fit = model.run()

        out_ents = spec['entities'].copy()
        model_maps = [model_fit.outputs.sigmasquareds]

        model_metadata = [{'stat': 'r_square', **out_ents}]

        self._results['effect_maps'] = sorted(glob(model_fit.outputs.results_dir + '/cope*'))
        self._results['variance_maps'] = sorted(glob(model_fit.outputs.results_dir + '/varcope*'))
        self._results['stat_maps'] = sorted(glob(model_fit.outputs.results_dir + '/tstat*')) # TODO: handle F-stats
        self._results['zscore_maps'] = sorted(glob(model_fit.outputs.results_dir + '/zstat*')) # TODO: handle F-stats 
        self._results['pvalue_maps'] = sorted(glob(model_fit.outputs.results_dir + '/cope*')) # TODO: CALCULATE P-VALUES
        self._results['contrast_metadata'] = contrast_metadata
        self._results['model_maps'] = model_maps
        self._results['model_metadata'] = model_metadata

        return runtime