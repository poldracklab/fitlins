import os
import numpy as np
import pandas as pd

from nipype.interfaces.base import LibraryBaseInterface, SimpleInterface, isdefined

from .abstract import (
    DesignMatrixInterface, FirstLevelEstimatorInterface, SecondLevelEstimatorInterface)


class NistatsBaseInterface(LibraryBaseInterface):
    _pkg = 'nistats'


def prepare_contrasts(contrasts, all_regressors):
    """ Make mutable copy of contrast list, and
    generate contrast design_matrix from dictionary weight mapping
    """
    if not isdefined(contrasts):
        return []

    out_contrasts = []
    for contrast in contrasts:
        # Are any necessary values missing for contrast estimation?
        missing = any([[n for n, v in row.items()
                        if v != 0 and n not in all_regressors]
                       for row in contrast['weights']])
        if not missing:
            # Fill in zeros
            weights = np.array([
                [row[col] if col in row else 0 for col in all_regressors]
                for row in contrast['weights']
                ])[0]

            out_contrasts.append(
                (contrast['name'], weights, contrast['type']))

    return out_contrasts


class DesignMatrix(NistatsBaseInterface, DesignMatrixInterface, SimpleInterface):

    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import design_matrix as dm
        info = self.inputs.session_info
        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        drop_missing = bool(self.inputs.drop_missing)

        if info['sparse'] not in (None, 'None'):
            sparse = pd.read_hdf(info['sparse'], key='sparse').rename(
                columns={'condition': 'trial_type',
                         'amplitude': 'modulation'})
            sparse = sparse.dropna(subset=['modulation'])  # Drop NAs
        else:
            sparse = None

        if info['dense'] not in (None, 'None'):
            dense = pd.read_hdf(info['dense'], key='dense')

            missing_columns = dense.isna().all()
            if drop_missing:
                # Remove columns with NaNs
                dense = dense[dense.columns[missing_columns == False]]
            elif missing_columns.any():
                    missing_names = ', '.join(
                        dense.columns[missing_columns].tolist())
                    raise RuntimeError(
                        f'The following columns are empty: {missing_names}. '
                        'Use --drop-missing to drop before model fitting.')

            column_names = dense.columns.tolist()
            drift_model = None if (('cosine00' in column_names) |
                                   ('cosine_00' in column_names)) else 'cosine'

            if dense.empty:
                dense = None
                column_names = None
        else:
            dense = None
            column_names = None
            drift_model = 'cosine'

        mat = dm.make_first_level_design_matrix(
            frame_times=np.arange(vols) * info['repetition_time'],
            events=sparse,
            add_regs=dense,
            add_reg_names=column_names,
            drift_model=drift_model,
        )

        mat.to_csv('design.tsv', sep='\t')
        self._results['design_matrix'] = os.path.join(runtime.cwd,
                                                      'design.tsv')
        return runtime


class FirstLevelModel(NistatsBaseInterface, FirstLevelEstimatorInterface, SimpleInterface):
    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import first_level_model as level1
        mat = pd.read_csv(self.inputs.design_matrix, delimiter='\t', index_col=0)
        img = nb.load(self.inputs.bold_file)

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None
        flm = level1.FirstLevelModel(
            mask_img=mask_file, smoothing_fwhm=smoothing_fwhm)
        flm.fit(img, design_matrices=mat)

        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        contrast_metadata = []
        out_ents = self.inputs.contrast_info[0]['entities']
        fname_fmt = os.path.join(runtime.cwd, '{}_{}.nii.gz').format
        for name, weights, contrast_type in prepare_contrasts(
                self.inputs.contrast_info, mat.columns.tolist()):
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents}
                )
            maps = flm.compute_contrast(
                weights, contrast_type, output_type='all')

            for map_type, map_list in (('effect_size', effect_maps),
                                       ('effect_variance', variance_maps),
                                       ('z_score', zscore_maps),
                                       ('p_value', pvalue_maps),
                                       ('stat', stat_maps)):

                fname = fname_fmt(name, map_type)
                maps[map_type].to_filename(fname)
                map_list.append(fname)

        self._results['effect_maps'] = effect_maps
        self._results['variance_maps'] = variance_maps
        self._results['stat_maps'] = stat_maps
        self._results['zscore_maps'] = zscore_maps
        self._results['pvalue_maps'] = pvalue_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime


def _flatten(x):
    return [elem for sublist in x for elem in sublist]


def _match(query, metadata):
    for key, val in query.items():
        if metadata.get(key) != val:
            return False
    return True


class SecondLevelModel(NistatsBaseInterface, SecondLevelEstimatorInterface, SimpleInterface):
    def _run_interface(self, runtime):
        from nistats import second_level_model as level2
        from nistats.contrasts import fixed_effects_img
        import nibabel as nb

        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None

        # Only initate for level higher than subject
        if self.inputs.level != 'Subject':
            model = level2.SecondLevelModel(smoothing_fwhm=smoothing_fwhm)

        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        contrast_metadata = []
        out_ents = self.inputs.contrast_info[0]['entities']  # Same for all
        fname_fmt = os.path.join(runtime.cwd, '{}_{}.nii.gz').format

        # Only keep files which match all entities for contrast
        stat_metadata = _flatten(self.inputs.stat_metadata)
        input_effects = _flatten(self.inputs.effect_maps)
        # XXX nistats should begin supporting mixed effects models soon
        input_variances = _flatten(self.inputs.variance_maps)
        # input_variances = [None] * len(input_effects)

        filtered_effects = []
        filtered_variances = []
        names = []
        for m, eff, var in zip(stat_metadata, input_effects, input_variances):
            if _match(out_ents, m):
                filtered_effects.append(eff)
                filtered_variances.append(var)
                names.append(m['contrast'])

        for name, weights, contrast_type in prepare_contrasts(self.inputs.contrast_info, names):
            # Need to add F-test support for intercept (more than one column)
            # Currently only taking 0th column as intercept (t-test)
            effects = (np.array(filtered_effects)[weights != 0]).tolist()
            variances = (np.array(filtered_variances)[weights != 0]).tolist()

            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents})

            # For now hard-coding to do FEMA at the subject level
            # Pass-through happens automatically as it can handle 1 input
            if self.inputs.level == 'Subject':
                # Smoothing not supported
                fe_res = fixed_effects_img(
                    effects, variances)
                maps = {
                    'effect_size': fe_res[0],
                    'effect_variance': fe_res[1],
                    'stat': fe_res[2]
                }
            else:
                if len(effects) < 2:
                    raise RuntimeError(
                        "At least two inputs are required for a 't' for 'F' "
                        "second level contrast")

                design_matrix = pd.DataFrame(
                    {'intercept': weights[weights != 0]})

                model.fit(effects, design_matrix=design_matrix)

                maps = model.compute_contrast(
                    second_level_stat_type=contrast_type,
                    output_type='all')

            for map_type, map_list in (('effect_size', effect_maps),
                                       ('effect_variance', variance_maps),
                                       ('z_score', zscore_maps),
                                       ('p_value', pvalue_maps),
                                       ('stat', stat_maps)):
                map = maps.get(map_type)
                if map:
                    fname = fname_fmt(name, map_type)
                    map.to_filename(fname)
                    map_list.append(fname)

        self._results['effect_maps'] = effect_maps
        self._results['variance_maps'] = variance_maps
        self._results['stat_maps'] = stat_maps
        self._results['contrast_metadata'] = contrast_metadata

        # These are "optional" as fixed effects do not support these (yet?)
        if zscore_maps:
            self._results['zscore_maps'] = zscore_maps
        if pvalue_maps:
            self._results['pvalue_maps'] = pvalue_maps

        return runtime
