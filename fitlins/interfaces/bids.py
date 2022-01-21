import json
import os
import re
import shutil

import numpy as np
import nibabel as nb

from itertools import chain
from pathlib import Path
from gzip import GzipFile

from nipype import logging
from nipype.utils.filemanip import copyfile
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    SimpleInterface,
    InputMultiPath,
    OutputMultiPath,
    File,
    Directory,
    traits,
    isdefined,
)
from nipype.interfaces.io import IOBase

from ..utils import snake_to_camel, to_alphanum

iflogger = logging.getLogger('nipype.interface')

ENTITY_WHITELIST = {
    'task',
    'run',
    'session',
    'subject',
    'space',
    'acquisition',
    'reconstruction',
    'echo',
}


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
        ".surf.gii",
        ".func.gii",
        ".dtseries.nii",
        ".dscalar.nii",
        ".nii.gz",
        ".tsv.gz",
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
    database_path = Directory(exists=False, desc='Path to bids database')
    model = traits.Either('default', InputMultiPath(File(exists=True)), desc='Model filename')
    selectors = traits.Dict(desc='Limit models to those with matching inputs')


class ModelSpecLoaderOutputSpec(TraitedSpec):
    model_spec = OutputMultiPath(
        traits.Dict(), desc='Model specification(s) as Python dictionaries'
    )


class ModelSpecLoader(SimpleInterface):
    """
    Load BIDS Stats Models specifications from a BIDS directory
    """

    input_spec = ModelSpecLoaderInputSpec
    output_spec = ModelSpecLoaderOutputSpec

    def _run_interface(self, runtime):
        import bids
        from bids.modeling import auto_model

        models = self.inputs.model
        if not isinstance(models, list):
            database_path = self.inputs.database_path
            layout = bids.BIDSLayout.load(database_path=database_path)

            if not isdefined(models):
                # model is not yet standardized, so validate=False
                # Ignore all subject directories and .git/ and .datalad/ directories
                indexer = bids.BIDSLayoutIndexer(
                    ignore=[re.compile(r'sub-'), re.compile(r'\.(git|datalad)')]
                )
                small_layout = bids.BIDSLayout(
                    layout.root,
                    derivatives=[d.root for d in layout.derivatives.values()],
                    validate=False,
                    indexer=indexer,
                )
                # PyBIDS can double up, so find unique models
                models = list(set(small_layout.get(suffix='smdl', return_type='file')))
                if not models:
                    raise ValueError("No models found")
            elif models == 'default':
                models = auto_model(layout)

        models = [_ensure_model(m) for m in models]

        if self.inputs.selectors:
            # This is almost certainly incorrect
            models = [
                model
                for model in models
                if all(
                    val in model['Input'].get(key, [val])
                    for key, val in self.inputs.selectors.items()
                )
            ]

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
    database_path = Directory(exists=True, mandatory=True, desc='Path to bids database directory.')
    model = traits.Dict(desc='Model specification', mandatory=True)
    selectors = traits.Dict(desc='Limit collected sessions', usedefault=True)


class LoadBIDSModelOutputSpec(TraitedSpec):
    design_info = traits.List(
        traits.Dict,
        desc='Descriptions of design matrices with sparse events, ' 'dense regressors and TR',
    )
    warnings = traits.List(File, desc='HTML warning snippet for reporting issues')
    all_specs = traits.Dict(desc='A collection of all specs built from the statsmodel')


class LoadBIDSModel(SimpleInterface):
    """
    Read a BIDS dataset and model and produce configurations that may be
    adapted to various model-fitting packages.

    Outputs
    -------
    design_info : list of list of dictionaries
        At the first level, a dictionary per-run containing the following keys:
            'sparse' : HDF5 file containing sparse representation of events
                       (onset, duration, amplitude)
            'dense'  : HDF5 file containing dense representation of events
                       regressors
            'repetition_time'   : float (in seconds)

    all_specs : dictionary of list of dictionaries
        The collection of specs from each level. Each dict at individual levels
        contains the following keys:
            'contrasts'  : a list of ContrastInfo objects each unit of analysis.
                A contrast specifiction is a list of contrast
                dictionaries. Each dict has form:
                  {
                    'name': str,
                    'conditions': list,
                    'weights: list,
                    test: str,
                    'entities': dict,
                  }
            'entities'  : The entities list contains a list for each level of analysis.
                At each level, the list contains a dictionary of entities that apply to
                each unit of analysis. For example, if the level is "Run" and there are
                20 runs in the dataset, the first entry will be a list of 20 dictionaries,
                each uniquely identifying a run.
            'level'  : The current level of the analysis [run, subject, dataset...]
            'X'  : The design matrix
            'model'  : The model part from the BIDS-StatsModels specification.
            'metadata' (only higher-levels): a parallel DataFrame with the same number of
                rows as X that contains all known metadata variabes that vary on a row-by-row
                basis but aren't actually predictiors

    warnings : list of files
        Files containing HTML snippets with any warnings produced while processing the first
        level.
    """

    input_spec = LoadBIDSModelInputSpec
    output_spec = LoadBIDSModelOutputSpec

    def _run_interface(self, runtime):
        from bids.modeling import BIDSStatsModelsGraph
        from bids.layout import BIDSLayout

        layout = BIDSLayout.load(database_path=self.inputs.database_path)
        selectors = self.inputs.selectors

        graph = BIDSStatsModelsGraph(layout, self.inputs.model)
        graph.load_collections(**selectors)

        self._results['all_specs'] = self._load_graph(runtime, graph)

        return runtime

    def _load_graph(self, runtime, graph, node=None, inputs=None, **filters):
        if node is None:
            node = graph.root_node

        specs = node.run(inputs, group_by=node.group_by, **filters)
        outputs = list(chain(*[s.contrasts for s in specs]))

        base_entities = graph.model["input"]

        if node.level == 'run':
            self._load_run_level(runtime, graph, specs)

        all_specs = {
            node.name: [
                {
                    'contrasts': [c._asdict() for c in spec.contrasts],
                    'entities': {**base_entities, **spec.entities},
                    'level': spec.node.level,
                    'X': spec.X,
                    'name': spec.node.name,
                    'model': spec.node.model,
                    # Metadata is only used in higher level models; save space
                    'metadata': spec.metadata if spec.node.level != "run" else None,
                }
                for spec in specs
            ]
        }

        for child in node.children:
            all_specs.update(
                self._load_graph(runtime, graph, child.destination, outputs, **child.filter)
            )

        return all_specs

    def _load_run_level(self, runtime, graph, specs):
        design_info = []
        warnings = []

        step_subdir = Path(runtime.cwd) / "run"
        step_subdir.mkdir(parents=True, exist_ok=True)

        for spec in specs:
            info = {}
            if "RepetitionTime" not in spec.metadata:
                # This shouldn't happen, so raise a (hopefully informative)
                # exception if I'm wrong
                fname = graph.layout.get(**spec.entities, suffix='bold')[0].path
                raise ValueError(
                    f"Preprocessed file {fname} does not have an " "associated RepetitionTime"
                )

            info["repetition_time"] = spec.metadata['RepetitionTime'][0]

            ent_string = '_'.join(f"{key}-{val}" for key, val in spec.entities.items())

            # These confounds are defined pairwise with the current volume and its
            # predecessor, and thus may be undefined (have value NaN) at the first volume.
            # In these cases, we impute the mean non-zero value, for the expected NaN only.
            # For derivatives, an initial "derivative" of 0 is reasonable.
            # Any other NaNs must be handled by an explicit transform in the BIDS model.
            initial_na = spec.data.columns[np.isnan(spec.data.values[0])]
            imputed = []
            for col in initial_na:
                if col in ('framewise_displacement', 'std_dvars', 'dvars'):
                    imputed.append(col)
                    vals = spec.data[col].values
                    spec.data[col][0] = np.nanmean(vals[vals != 0])
                elif "derivative1" in col:
                    imputed.append(col)
                    spec.data[col][0] = 0

            info["dense"] = str(step_subdir / '{}_dense.h5'.format(ent_string))
            spec.data.to_hdf(info["dense"], key='dense')

            warning_file = step_subdir / '{}_warning.html'.format(ent_string)
            with warning_file.open('w') as fobj:
                if imputed:
                    fobj.write(IMPUTATION_SNIPPET.format(', '.join(imputed)))

            design_info.append(info)
            warnings.append(str(warning_file))

        self._results['warnings'] = warnings
        self._results['design_info'] = design_info


class BIDSSelectInputSpec(BaseInterfaceInputSpec):
    database_path = Directory(exists=True, mandatory=True, desc='Path to bids database.')
    entities = InputMultiPath(traits.Dict(), mandatory=True)
    selectors = traits.Dict(desc='Additional selectors to be applied', usedefault=True)


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
                    "".format(layout.root, selectors)
                )
            elif len(bold_file) > 1:
                raise ValueError(
                    "Non-unique BOLD file in {} with entities {}.\n"
                    "Matches:\n\t{}"
                    "".format(
                        layout.root,
                        selectors,
                        "\n\t".join(
                            '{} ({})'.format(f.path, layout.files[f.path].entities)
                            for f in bold_file
                        ),
                    )
                )

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
    base_directory = Directory(mandatory=True, desc='Path to BIDS (or derivatives) root directory')
    in_file = InputMultiPath(File(exists=True), mandatory=True)
    entities = InputMultiPath(
        traits.Dict, usedefault=True, desc='Per-file entities to include in filename'
    )
    fixed_entities = traits.Dict(usedefault=True, desc='Entities to include in all filenames')
    path_patterns = InputMultiPath(
        traits.Str, desc='BIDS path patterns describing format of file names'
    )


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
        for entities, in_file in zip(self.inputs.entities, self.inputs.in_file):
            ents = {**self.inputs.fixed_entities}
            ents.update(entities)
            ext = bids_split_filename(in_file)[2]
            ents['extension'] = self._extension_map.get(ext, ext)

            # In some instances, name/contrast could have the following
            # format (eg: gain.Range, gain.EqualIndifference).
            # This prevents issues when creating/searching files for the report
            for k, v in ents.items():
                if k in ("name", "contrast", "stat"):
                    ents.update({k: to_alphanum(str(v))})

            out_fname = os.path.join(
                base_dir, layout.build_path(ents, path_patterns, validate=False)
            )
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)

            _copy_or_convert(in_file, out_fname)
            out_files.append(out_fname)

        return {'out_file': out_files}
