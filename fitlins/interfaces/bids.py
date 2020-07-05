import os
import re
from pathlib import Path
from gzip import GzipFile
import json
import shutil
import numpy as np
import nibabel as nb

from nipype import logging
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    InputMultiPath, OutputMultiPath, File, Directory,
    traits, isdefined
    )
from nipype.interfaces.io import IOBase

from ..utils import snake_to_camel

iflogger = logging.getLogger('nipype.interface')

ENTITY_WHITELIST = {'task', 'run', 'session', 'subject', 'space',
                    'acquisition', 'reconstruction', 'echo'}


def bids_split_filename(fname):
    """Split a filename into parts: path, base filename, and extension

    Respects multi-part file types used in BIDS standard and draft extensions

    Largely copied from nipype.utils.filemanip.split_filename

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        path of fname
    fname : str
        basename of filename, without extension
    ext : str
        file extension of fname
    """
    special_extensions = [
        ".surf.gii", ".func.gii",
        ".dtseries.nii", ".dscalar.nii",
        ".nii.gz", ".tsv.gz",
        ]

    pth = os.path.dirname(fname)
    fname = os.path.basename(fname)

    for special_ext in special_extensions:
        if fname.lower().endswith(special_ext.lower()):
            ext_len = len(special_ext)
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    else:
        fname, ext = os.path.splitext(fname)

    return pth, fname, ext


def _ensure_model(model):
    model = getattr(model, 'filename', model)

    if isinstance(model, str):
        if os.path.exists(model):
            with open(model) as fobj:
                model = json.load(fobj)
        else:
            model = json.loads(model)
    return model


class ModelSpecLoaderInputSpec(BaseInterfaceInputSpec):
    database_path = Directory(exists=False,
                              desc='Path to bids database')
    model = traits.Either('default', InputMultiPath(File(exists=True)),
                          desc='Model filename')
    selectors = traits.Dict(desc='Limit models to those with matching inputs')


class ModelSpecLoaderOutputSpec(TraitedSpec):
    model_spec = OutputMultiPath(traits.Dict(),
                                 desc='Model specification(s) as Python dictionaries')


class ModelSpecLoader(SimpleInterface):
    """
    Load BIDS Stats Models specifications from a BIDS directory
    """
    input_spec = ModelSpecLoaderInputSpec
    output_spec = ModelSpecLoaderOutputSpec

    def _run_interface(self, runtime):
        import bids
        from bids.analysis import auto_model
        models = self.inputs.model
        if not isinstance(models, list):
            database_path = self.inputs.database_path
            layout = bids.BIDSLayout.load(database_path=database_path)

            if not isdefined(models):
                # model is not yet standardized, so validate=False
                # Ignore all subject directories and .git/ and .datalad/ directories
                small_layout = bids.BIDSLayout(
                    layout.root, derivatives=[d.root for d in layout.derivatives.values()],
                    validate=False,
                    ignore=[re.compile(r'sub-'),
                            re.compile(r'\.(git|datalad)')])
                # PyBIDS can double up, so find unique models
                models = list(set(small_layout.get(suffix='smdl', return_type='file')))
                if not models:
                    raise ValueError("No models found")
            elif models == 'default':
                models = auto_model(layout)

        models = [_ensure_model(m) for m in models]

        if self.inputs.selectors:
            # This is almost certainly incorrect
            models = [model for model in models
                      if all(val in model['input'].get(key, [val])
                             for key, val in self.inputs.selectors.items())]

        self._results['model_spec'] = models

        return runtime


IMPUTATION_SNIPPET = """\
<div class="warning">
    The following confounds had NaN values for the first volume: {}.
    The mean of non-zero values for the remaining entries was imputed.
    If another strategy is desired, it must be explicitly specified in
    the model.
</div>
"""


class LoadBIDSModelInputSpec(BaseInterfaceInputSpec):
    database_path = Directory(exists=True,
                              mandatory=True,
                              desc='Path to bids database directory.')
    model = traits.Dict(desc='Model specification', mandatory=True)
    selectors = traits.Dict(desc='Limit collected sessions', usedefault=True)


class LoadBIDSModelOutputSpec(TraitedSpec):
    design_info = traits.List(traits.Dict,
                              desc='Descriptions of design matrices with sparse events, '
                                   'dense regressors and TR')
    contrast_info = traits.List(traits.List(traits.List(traits.Dict)),
                                desc='A list of contrast specifications at each unit of analysis')
    entities = traits.List(traits.List(traits.Dict),
                           desc='A list of applicable entities at each unit of analysis')
    warnings = traits.List(File, desc='HTML warning snippet for reporting issues')


class LoadBIDSModel(SimpleInterface):
    """
    Read a BIDS dataset and model and produce configurations that may be
    adapted to various model-fitting packages.

    Outputs
    -------
    design_info : list of dictionaries
        At the first level, a dictionary per-run containing the following keys:
            'sparse' : HDF5 file containing sparse representation of events
                       (onset, duration, amplitude)
            'dense'  : HDF5 file containing dense representation of events
                       regressors
            'repetition_time'   : float (in seconds)

    entities : list of list of dictionaries
        The entities list contains a list for each level of analysis.
        At each level, the list contains a dictionary of entities that apply to each
        unit of analysis. For example, if the first level is "Run" and there are 20
        runs in the dataset, the first entry will be a list of 20 dictionaries, each
        uniquely identifying a run.

    contrast_info : list of lists of files
        A list of contrast specifications at each unit of analysis; hence a list of
        dictionaries for each entity dictionary.
        A contrast specification is a list of contrast dictionaries
        Each dictionary has form:
          {
            'entities': dict,
            'name': str,
            'type': 't' or 'F',
            'weights': dict
          }

        The entities dictionary is a subset of the corresponding dictionary in the entities
        output.
        The weights dictionary is a mapping from design matrix column names to floats.

    warnings : list of files
        Files containing HTML snippets with any warnings produced while processing the first
        level.
    """
    input_spec = LoadBIDSModelInputSpec
    output_spec = LoadBIDSModelOutputSpec

    def _run_interface(self, runtime):
        from bids.analysis import Analysis
        from bids.layout import BIDSLayout

        layout = BIDSLayout.load(database_path=self.inputs.database_path)

        selectors = self.inputs.selectors

        analysis = Analysis(model=self.inputs.model, layout=layout)
        analysis.setup(drop_na=False, **selectors)
        self._load_level1(runtime, analysis)
        self._load_higher_level(runtime, analysis)

        return runtime

    def _load_level1(self, runtime, analysis):
        step = analysis.steps[0]
        step_subdir = Path(runtime.cwd) / step.level
        step_subdir.mkdir(parents=True, exist_ok=True)

        entities = []
        design_info = []
        contrast_info = []
        warnings = []
        for coll in step.get_collections():
            sparse = dense = None
            try:
                sparse = coll.to_df(include_dense=False)
            except ValueError:
                pass
            try:
                dense = coll.to_df(include_sparse=False, sampling_rate="TR")
            except ValueError:
                pass
            ents = coll.entities.copy()

            info = {}

            # Metadata is now included in entities
            TR = ents.pop('RepetitionTime', None)  # Required field in seconds
            if TR is None:  # But is unreliable (for now?)
                preproc_files = analysis.layout.get(
                    extension=['.nii', '.nii.gz'], desc='preproc', **ents)
                if len(preproc_files) != 1:
                    raise ValueError('Too many BOLD files found')

                fname = preproc_files[0].path
                TR = analysis.layout.get_metadata(fname)['RepetitionTime']

            # Ignore metadata entities
            entity_whitelist = analysis.layout.get_entities(metadata=False)
            ents = {key: ents[key] for key in ents if key in entity_whitelist}

            # ents is now pretty populous
            ents.pop('suffix', None)
            ents.pop('datatype', None)

            if step.level in ('session', 'subject', 'dataset'):
                ents.pop('run', None)
            if step.level in ('subject', 'dataset'):
                ents.pop('session', None)
            if step.level == 'dataset':
                ents.pop('subject', None)
            space = ents.get('space')
            if space is None:
                spaces = analysis.layout.get_spaces(
                    suffix='bold',
                    extension=['.nii', '.nii.gz', '.dtseries.nii', '.func.gii'])
                if spaces:
                    spaces = sorted(spaces)
                    space = spaces[0]
                    if len(spaces) > 1:
                        iflogger.warning(
                            'No space was provided, but multiple spaces were detected: %s. '
                            'Selecting the first (ordered lexicographically): %s'
                            % (', '.join(spaces), space))
                ents['space'] = space

            ent_string = '_'.join('{}-{}'.format(key, val)
                                  for key, val in ents.items())

            sparse_file = None
            if sparse is not None:
                sparse_file = step_subdir / '{}_sparse.h5'.format(ent_string)
                sparse.to_hdf(sparse_file, key='sparse')

            imputed = []
            if dense is not None:
                # Note that FMRIPREP includes CosineXX columns to accompany
                # t/aCompCor
                # We may want to add criteria to include HPF columns that are not
                # explicitly listed in the model
                names = [var for var in step.model['x'] if var in dense.columns]
                names.extend(col for col in dense.columns if col.startswith('non_steady_state'))
                dense = dense[names]

                # These confounds are defined pairwise with the current volume
                # and its predecessor, and thus may be undefined (have value
                # NaN) at the first volume.
                # In these cases, we impute the mean non-zero value, for the
                # expected NaN only.
                # Any other NaNs must be handled by an explicit transform in
                # the BIDS model.
                for imputable in ('framewise_displacement',
                                  'std_dvars', 'dvars'):
                    if imputable in dense.columns:
                        vals = dense[imputable].values
                        if not np.isnan(vals[0]):
                            continue

                        # Impute the mean non-zero, non-NaN value
                        dense[imputable][0] = np.nanmean(vals[vals != 0])
                        imputed.append(imputable)

                dense_file = step_subdir / '{}_dense.h5'.format(ent_string)
                dense.to_hdf(dense_file, key='dense')

            else:
                dense_file = None

            info['sparse'] = str(sparse_file) if sparse_file else None
            info['dense'] = str(dense_file) if dense_file else None
            info['repetition_time'] = TR

            contrasts = [dict(c._asdict()) for c in step.get_contrasts(coll)]
            for con in contrasts:
                con['weights'] = con['weights'].to_dict('records')
                # Ugly hack. This should be taken care of on the pybids side.
                con['entities'] = {k: v for k, v in con['entities'].items()
                                   if k in ENTITY_WHITELIST}
                if step.level in ('session', 'subject', 'dataset'):
                    con['entities'].pop('run', None)
                if step.level in ('subject', 'dataset'):
                    con['entities'].pop('session', None)
                if step.level == 'dataset':
                    con['entities'].pop('subject', None)

            warning_file = step_subdir / '{}_warning.html'.format(ent_string)
            with warning_file.open('w') as fobj:
                if imputed:
                    fobj.write(IMPUTATION_SNIPPET.format(', '.join(imputed)))

            entities.append(ents)
            design_info.append(info)
            contrast_info.append(contrasts)
            warnings.append(str(warning_file))

        self._results['design_info'] = design_info
        self._results['warnings'] = warnings
        self._results.setdefault('entities', []).append(entities)
        self._results.setdefault('contrast_info', []).append(contrast_info)

    def _load_higher_level(self, runtime, analysis):
        for step in analysis.steps[1:]:
            contrast_info = []
            for coll in step.get_collections():
                contrasts = [dict(c._asdict()) for c in step.get_contrasts(coll)]
                if all(c['weights'].empty for c in contrasts):
                    continue

                for contrast in contrasts:
                    contrast['weights'] = contrast['weights'].to_dict('records')
                    # Ugly hack. This should be taken care of on the pybids side.
                    contrast['entities'] = {k: v
                                            for k, v in contrast['entities'].items()
                                            if k in ENTITY_WHITELIST}
                    if step.level in ('session', 'subject', 'dataset'):
                        contrast['entities'].pop('run', None)
                    if step.level in ('subject', 'dataset'):
                        contrast['entities'].pop('session', None)
                    if step.level == 'dataset':
                        contrast['entities'].pop('subject', None)
                contrast_info.append(contrasts)

            self._results['contrast_info'].append(contrast_info)


class BIDSSelectInputSpec(BaseInterfaceInputSpec):
    database_path = Directory(exists=True,
                              mandatory=True,
                              desc='Path to bids database.')
    entities = InputMultiPath(traits.Dict(), mandatory=True)
    selectors = traits.Dict(desc='Additional selectors to be applied',
                            usedefault=True)


class BIDSSelectOutputSpec(TraitedSpec):
    bold_files = OutputMultiPath(File)
    mask_files = OutputMultiPath(traits.Either(File, None))
    entities = OutputMultiPath(traits.Dict)


class BIDSSelect(SimpleInterface):
    input_spec = BIDSSelectInputSpec
    output_spec = BIDSSelectOutputSpec

    def _run_interface(self, runtime):
        from bids.layout import BIDSLayout

        layout = BIDSLayout.load(database_path=self.inputs.database_path)

        bold_files = []
        mask_files = []
        entities = []
        for ents in self.inputs.entities:
            selectors = {'desc': 'preproc', **ents, **self.inputs.selectors}
            bold_file = layout.get(**selectors)

            if len(bold_file) == 0:
                raise FileNotFoundError(
                    "Could not find BOLD file in {} with entities {}"
                    "".format(layout.root, selectors))
            elif len(bold_file) > 1:
                raise ValueError(
                    "Non-unique BOLD file in {} with entities {}.\n"
                    "Matches:\n\t{}"
                    "".format(layout.root, selectors,
                              "\n\t".join(
                                  '{} ({})'.format(
                                      f.path,
                                      layout.files[f.path].entities)
                                  for f in bold_file)))

            # Select exactly matching mask file (may be over-cautious)
            bold_ents = layout.parse_file_entities(bold_file[0].path)
            bold_ents['suffix'] = 'mask'
            bold_ents['desc'] = 'brain'
            bold_ents['extension'] = ['.nii', '.nii.gz']
            mask_file = layout.get(**bold_ents)
            bold_ents.pop('suffix')
            bold_ents.pop('desc')

            bold_files.append(bold_file[0].path)
            mask_files.append(mask_file[0].path if mask_file else None)
            entities.append(bold_ents)

        self._results['bold_files'] = bold_files
        self._results['mask_files'] = mask_files
        self._results['entities'] = entities

        return runtime


def _copy_or_convert(in_file, out_file):
    in_ext = bids_split_filename(in_file)[2]
    out_ext = bids_split_filename(out_file)[2]

    # Copy if filename matches
    if in_ext == out_ext:
        copyfile(in_file, out_file, copy=True, use_hardlink=True)
        return

    # gzip/gunzip if it's easy
    if in_ext == out_ext + '.gz' or in_ext + '.gz' == out_ext:
        read_open = GzipFile if in_ext.endswith('.gz') else open
        write_open = GzipFile if out_ext.endswith('.gz') else open
        with read_open(in_file, mode='rb') as in_fobj:
            with write_open(out_file, mode='wb') as out_fobj:
                shutil.copyfileobj(in_fobj, out_fobj)
        return

    # Let nibabel take a shot
    try:
        nb.save(nb.load(in_file), out_file)
    except Exception:
        pass
    else:
        return

    raise RuntimeError("Cannot convert {} to {}".format(in_ext, out_ext))


class BIDSDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = Directory(
        mandatory=True,
        desc='Path to BIDS (or derivatives) root directory')
    in_file = InputMultiPath(File(exists=True), mandatory=True)
    entities = InputMultiPath(traits.Dict, usedefault=True,
                              desc='Per-file entities to include in filename')
    fixed_entities = traits.Dict(usedefault=True,
                                 desc='Entities to include in all filenames')
    path_patterns = InputMultiPath(
        traits.Str, desc='BIDS path patterns describing format of file names')


class BIDSDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File, desc='output file')


class BIDSDataSink(IOBase):
    input_spec = BIDSDataSinkInputSpec
    output_spec = BIDSDataSinkOutputSpec

    _always_run = True
    _extension_map = {".nii": ".nii.gz"}

    def _list_outputs(self):
        from bids.layout import BIDSLayout
        base_dir = self.inputs.base_directory

        os.makedirs(base_dir, exist_ok=True)

        layout = BIDSLayout(base_dir, validate=False)
        path_patterns = self.inputs.path_patterns
        if not isdefined(path_patterns):
            path_patterns = None

        out_files = []
        for entities, in_file in zip(self.inputs.entities,
                                     self.inputs.in_file):
            ents = {**self.inputs.fixed_entities}
            ents.update(entities)
            ext = bids_split_filename(in_file)[2]
            ents['extension'] = self._extension_map.get(ext, ext)

            ents = {k: snake_to_camel(str(v)) for k, v in ents.items()}

            out_fname = os.path.join(
                base_dir, layout.build_path(
                    ents, path_patterns, validate=False))
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)

            _copy_or_convert(in_file, out_fname)
            out_files.append(out_fname)

        return {'out_file': out_files}
