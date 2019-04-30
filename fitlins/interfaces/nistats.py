import os
import numpy as np
import pandas as pd

from nipype.interfaces.base import (
    LibraryBaseInterface, SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined
    )


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
        # Fill in zeros
        weights = np.array([
            [row[col] if col in row else 0 for col in all_regressors]
            for row in contrast['weights']
            ])
        out_contrasts.append(
            (contrast['name'], weights, contrast['type']))

    return out_contrasts


class FirstLevelModelInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True)
    session_info = traits.Dict()
    contrast_info = traits.List(traits.Dict)
    smoothing_fwhm = traits.Float(desc='Full-width half max (FWHM) in mm for smoothing in mask')


class FirstLevelModelOutputSpec(TraitedSpec):
    effect_maps = traits.List(File)
    variance_maps = traits.List(File)
    stat_maps = traits.List(File)
    zscore_maps = traits.List(File)
    pvalue_maps = traits.List(File)
    contrast_metadata = traits.List(traits.Dict)
    design_matrix = File()


class FirstLevelModel(NistatsBaseInterface, SimpleInterface):
    input_spec = FirstLevelModelInputSpec
    output_spec = FirstLevelModelOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import design_matrix as dm
        from nistats import first_level_model as level1
        info = self.inputs.session_info
        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        if info['sparse'] not in (None, 'None'):
            sparse = pd.read_hdf(info['sparse'], key='sparse').rename(
                columns={'condition': 'trial_type',
                         'amplitude': 'modulation'})
            sparse = sparse.dropna(subset=['modulation'])  # Drop NAs
        else:
            sparse = None

        if info['dense'] not in (None, 'None'):
            dense = pd.read_hdf(info['dense'], key='dense')
            column_names = dense.columns.tolist()
            drift_model = None if 'cosine_00' in column_names else 'cosine'
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

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None
        flm = level1.FirstLevelModel(
            mask=mask_file, smoothing_fwhm=smoothing_fwhm)
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
            maps = flm.compute_contrast(weights, contrast_type, output_type='all')
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents}
                )

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


class SecondLevelModelInputSpec(BaseInterfaceInputSpec):
    effect_maps = traits.List(traits.List(File(exists=True)), mandatory=True)
    variance_maps = traits.List(traits.List(File(exists=True)))
    stat_metadata = traits.List(traits.List(traits.Dict), mandatory=True)
    contrast_info = traits.List(traits.Dict, mandatory=True)
    smoothing_fwhm = traits.Float(desc='Full-width half max (FWHM) in mm for smoothing in mask')


class SecondLevelModelOutputSpec(TraitedSpec):
    effect_maps = traits.List(File)
    variance_maps = traits.List(File)
    stat_maps = traits.List(File)
    zscore_maps = traits.List(File)
    pvalue_maps = traits.List(File)
    contrast_metadata = traits.List(traits.Dict)
    contrast_matrix = File()


def _flatten(x):
    return [elem for sublist in x for elem in sublist]


def _match(query, metadata):
    for key, val in query.items():
        if metadata.get(key) != val:
            return False
    return True


class SecondLevelModel(NistatsBaseInterface, SimpleInterface):
    input_spec = SecondLevelModelInputSpec
    output_spec = SecondLevelModelOutputSpec

    def _run_interface(self, runtime):
        from nistats import second_level_model as level2
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None
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
        # input_variances = _flatten(self.inputs.variance_maps)
        input_variances = [None] * len(input_effects)

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
            weights = weights[0]
            effects = (np.array(filtered_effects)[weights != 0]).tolist()
            _variances = (np.array(filtered_variances)[weights != 0]).tolist()
            design_matrix = pd.DataFrame({'intercept': weights[weights != 0]})

            model.fit(effects, design_matrix=design_matrix)

            maps = model.compute_contrast(second_level_stat_type=contrast_type,
                                          output_type='all')
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents})

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
