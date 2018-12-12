import os
import numpy as np
import pandas as pd
import nibabel as nb
from nistats import design_matrix as dm
from nistats import first_level_model as level1
from nistats import second_level_model as level2

from nipype.interfaces.base import (
    LibraryBaseInterface, SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined
    )


class NistatsBaseInterface(LibraryBaseInterface):
    _pkg = 'nistats'


def prepare_contrasts(contrasts, all_regressors):
    """ Make mutable copy of contrast list, and generate contrast design_matrix
    from dictionary weight mapping
    """
    if not isdefined(contrasts):
        return []
    else:
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


class FirstLevelModelOutputSpec(TraitedSpec):
    contrast_maps = traits.List(File)
    contrast_metadata = traits.List(traits.Dict)
    design_matrix = File()


class FirstLevelModel(NistatsBaseInterface, SimpleInterface):
    input_spec = FirstLevelModelInputSpec
    output_spec = FirstLevelModelOutputSpec

    def _run_interface(self, runtime):
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
        flm = level1.FirstLevelModel(mask=mask_file)
        flm.fit(img, design_matrices=mat)

        contrast_maps = []
        contrast_metadata = []
        for name, weights, type in prepare_contrasts(
                self.inputs.contrast_info, mat.columns.tolist()):
            es = flm.compute_contrast(weights,
                                      type,
                                      output_type='effect_size')
            es_fname = os.path.join(
                runtime.cwd, '{}.nii.gz').format(name)
            es.to_filename(es_fname)

            contrast_maps.append(es_fname)
            contrast_metadata.append({'contrast': name,
                                      'type': 'effect'})

        self._results['contrast_maps'] = contrast_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime


class SecondLevelModelInputSpec(BaseInterfaceInputSpec):
    stat_files = traits.List(traits.List(File(exists=True)), mandatory=True)
    stat_metadata = traits.List(traits.List(traits.Dict), mandatory=True)
    contrast_info = traits.List(traits.Dict, mandatory=True)


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

        # Flatten and join files and metadata into single list of tuples
        inputs = [(self.inputs.stat_files[i][j], item)
                  for i, sublist in enumerate(self.inputs.stat_metadata)
                  for j, item in enumerate(sublist)]
        contrast_maps = []
        contrast_metadata = []

        ents = self.inputs.contrast_info[0]['entities']
        files = []
        names = []
        for file, md in inputs:
            # If file matches all contrast entities
            if not sum([1 for e, v in ents.items() if md[e] != v]):
                files.append(file)
                names.append(md['contrast'])
        files = np.array(files)

        for name, weights, type in prepare_contrasts(
          self.inputs.contrast_info, names):
            # Need to add F-test support for intercept (more than one column)
            # Currently only taking 0th column as intercept (t-test)
            dm = weights[0]

            # Filter input files [intercept != 0]
            model.fit(files[dm != 0].tolist(),
                      design_matrix=pd.DataFrame({'intercept': dm[dm != 0]}))

            stat = model.compute_contrast(
                second_level_stat_type=type)
            stat_fname = os.path.join(
                runtime.cwd, '{}.nii.gz').format(name)
            stat.to_filename(stat_fname)
            contrast_maps.append(stat_fname)

            metadata = {
                'type': 'stat',
                'contrast': name
                }
            contrast_metadata.append(metadata)

        self._results['contrast_maps'] = contrast_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime
