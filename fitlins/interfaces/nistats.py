import os
from functools import reduce
import numpy as np
import pandas as pd
import nibabel as nb
from nistats import design_matrix as dm
from nistats import first_level_model as level1
from nistats import second_level_model as level2

from nipype.interfaces.base import (
    LibraryBaseInterface, SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    OutputMultiObject, File, traits, isdefined
    )

from ..utils import dict_intersection


class NistatsBaseInterface(LibraryBaseInterface):
    _pkg = 'nistats'


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
    contrast_maps = traits.List(File)
    contrast_metadata = traits.List(traits.Dict)
    design_matrix = File()
    contrast_matrix = File()


class FirstLevelModel(NistatsBaseInterface, SimpleInterface):
    input_spec = FirstLevelModelInputSpec
    output_spec = FirstLevelModelOutputSpec

    def _run_interface(self, runtime):
        info = self.inputs.session_info

        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        events = pd.read_hdf(info['events'], key='events')

        if info['confounds'] is not None and info['confounds'] != 'None':
            confounds = pd.read_hdf(info['confounds'], key='confounds')
            confound_names = confounds.columns.tolist()
            drift_model = None if 'Cosine00' in confound_names else 'cosine'
        else:
            confounds = None
            confound_names = None
            drift_model = 'cosine'

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
            add_reg_names=confound_names,
            drift_model=drift_model,
            )

        # Assume that explanatory variables == HRF-convolved variables
        exp_vars = events['condition'].unique().tolist()

        contrast_matrix, contrast_types = build_contrast_matrix(contrast_spec,
                                                                mat, exp_vars)

        mat.to_csv('design.tsv', sep='\t')
        self._results['design_matrix'] = os.path.join(runtime.cwd,
                                                      'design.tsv')

        contrast_matrix.to_csv('contrasts.tsv', sep='\t')
        self._results['contrast_matrix'] = os.path.join(
            runtime.cwd, 'contrasts.tsv')

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        flm = level1.FirstLevelModel(mask=mask_file)
        flm.fit(img, design_matrices=mat)

        contrast_maps = []
        contrast_metadata = []
        stat_fmt = os.path.join(runtime.cwd, '{}.nii.gz').format
        for contrast, ctype in zip(contrast_matrix, contrast_types):
            es = flm.compute_contrast(contrast_matrix[contrast].values,
                                      {'T': 't', 'F': 'F'}[ctype],
                                      output_type='effect_size')
            es_fname = stat_fmt(contrast)
            es.to_filename(es_fname)

            contrast_maps.append(es_fname)
            contrast_metadata.append({'contrast': contrast,
                                      'type': 'effect'})
        self._results['contrast_maps'] = contrast_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime


class SecondLevelModelInputSpec(BaseInterfaceInputSpec):
    stat_files = traits.List(traits.List(File(exists=True)), mandatory=True)
    stat_metadata = traits.List(traits.List(traits.Dict))
    contrast_info = File(exists=True)
    contrast_indices = traits.List(traits.Dict)


class SecondLevelModelOutputSpec(TraitedSpec):
    contrast_maps = traits.List(File)
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
        model = level2.SecondLevelModel()
        files = []
        # Super inefficient... think more about this later
        for idx in self.inputs.contrast_indices:
            for fname, metadata in zip(_flatten(self.inputs.stat_files),
                                       _flatten(self.inputs.stat_metadata)):
                if _match(idx, metadata):
                    files.append(fname)
                    break
            else:
                raise ValueError

        out_ents = reduce(dict_intersection, self.inputs.contrast_indices)
        in_ents = [{key: val for key, val in index.items() if key not in out_ents}
                   for index in self.inputs.contrast_indices]

        contrast_spec = pd.read_hdf(self.inputs.contrast_info,
                                    key='contrasts')

        contrast_matrix = contrast_spec.drop(columns=['type']).T
        contrast_types = contrast_spec['type']

        contrast_matrix.index = ['_'.join('{}-{}'.format(key, val)
                                          for key, val in ents.items())
                                 for ents in in_ents]
        contrast_matrix.to_csv('contrasts.tsv', sep='\t')
        self._results['contrast_matrix'] = os.path.join(
            runtime.cwd, 'contrasts.tsv')

        out_ents['type'] = 'stat'

        contrast_maps = []
        contrast_metadata = []
        stat_fmt = os.path.join(runtime.cwd, '{}.nii.gz').format
        for contrast, ctype in zip(contrast_matrix, contrast_types):
            intercept = contrast_matrix[contrast]
            data = np.array(files)[intercept != 0].tolist()
            intercept = intercept[intercept != 0]

            model.fit(data, design_matrix=pd.DataFrame({'intercept': intercept}))
            stat_type = {'T': 't', 'F': 'F'}[ctype]

            stat = model.compute_contrast(second_level_stat_type=stat_type)
            stat_fname = stat_fmt(contrast)
            stat.to_filename(stat_fname)

            contrast_maps.append(stat_fname)
            metadata = out_ents.copy()
            metadata['contrast'] = contrast
            contrast_metadata.append(metadata)

        self._results['contrast_maps'] = contrast_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime
