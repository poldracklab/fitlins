import os
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    InputMultiPath, OutputMultiPath, File, Directory,
    traits, isdefined
    )
from bids import grabbids as gb, analysis as ba


class LoadLevel1BIDSModelInputSpec(BaseInterfaceInputSpec):
    bids_dirs = InputMultiPath(Directory(exists=True),
                               mandatory=True,
                               desc='BIDS dataset root directories')
    model = File(exists=True, desc='Model filename')
    selectors = traits.Dict(desc='Limit collected sessions')
    include_pattern = InputMultiPath(
        traits.Str, xor=['exclude_pattern'],
        desc='Patterns to select sub-directories of BIDS root')
    exclude_pattern = InputMultiPath(
        traits.Str, xor=['include_pattern'],
        desc='Patterns to ignore sub-directories of BIDS root')


class LoadLevel1BIDSModelOutputSpec(TraitedSpec):
    session_info = traits.List(traits.Dict())
    contrast_info = traits.List(File())


class LoadLevel1BIDSModel(SimpleInterface):
    input_spec = LoadLevel1BIDSModelInputSpec
    output_spec = LoadLevel1BIDSModelOutputSpec

    def _run_interface(self, runtime):
        include = self.inputs.include_pattern
        exclude = self.inputs.include_pattern
        if not isdefined(include):
            include = None
        if not isdefined(exclude):
            exclude = None
        layout = gb.BIDSLayout(self.inputs.bids_dirs, include=include,
                               exclude=exclude)
        model_fname = self.inputs.model
        if not isdefined(model_fname):
            models = layout.get(type='model')
            if len(models) == 1:
                model_fname = models[0].filename
            elif models:
                raise ValueError("Ambiguous model")
            else:
                raise ValueError("No models found")

        selectors = (self.inputs.selectors
                     if isdefined(self.inputs.selectors) else {})

        analysis = ba.Analysis(model=model_fname, layout=layout)
        selectors.update(analysis.model['input'])
        analysis.setup(**selectors)
        block = analysis.blocks[0]

        session_info = []
        contrast_info = []
        for paradigm, _, ents in block.get_design_matrix(
                block.model['HRF_variables'], mode='sparse'):
            info = {'entities': ents}

            bold_files = layout.get(type='bold',
                                    extensions=['.nii', '.nii.gz'],
                                    **ents)
            if len(bold_files) != 1:
                raise ValueError('Too many BOLD files found')

            fname = bold_files[0].filename

            # Required field in seconds
            TR = layout.get_metadata(fname)['RepetitionTime']

            _, confounds, _ = block.get_design_matrix(mode='dense',
                                                      sampling_rate=1/TR,
                                                      **ents)[0]

            # Note that FMRIPREP includes CosineXX columns to accompany
            # t/aCompCor
            # We may want to add criteria to include HPF columns that are not
            # explicitly listed in the model
            names = [col for col in confounds.columns
                     if col.startswith('NonSteadyStateOutlier') or
                     col in block.model['variables']]

            ent_string = '_'.join('{}-{}'.format(key, val)
                                  for key, val in ents.items())
            events_file = os.path.join(runtime.cwd,
                                       '{}_events.h5'.format(ent_string))
            confounds_file = os.path.join(runtime.cwd,
                                          '{}_confounds.h5'.format(ent_string))
            paradigm.to_hdf(events_file, key='events')
            confounds[names].fillna(0).to_hdf(confounds_file, key='confounds')
            info['events'] = events_file
            info['confounds'] = confounds_file
            info['repetition_time'] = TR

            # Transpose so each contrast gets a row of data instead of column
            contrasts = block.get_contrasts([contrast['name']
                                             for contrast in block.contrasts],
                                            **ents)[0][0].T
            # Add test indicator column
            contrasts['type'] = [contrast['type']
                                 for contrast in block.contrasts]

            contrasts_file = os.path.join(runtime.cwd,
                                          '{}_contrasts.h5'.format(ent_string))
            contrasts.to_hdf(contrasts_file, key='contrasts')

            session_info.append(info)
            contrast_info.append(contrasts_file)

        runtime.analysis = analysis

        self._results['session_info'] = session_info
        self._results['contrast_info'] = contrast_info
        return runtime


class BIDSSelectInputSpec(BaseInterfaceInputSpec):
    bids_dirs = InputMultiPath(Directory(exists=True),
                               mandatory=True,
                               desc='BIDS dataset root directories')
    session_info = traits.List(traits.Dict(), mandatory=True)
    selectors = traits.Dict(desc='Additional selectors to be applied',
                            usedefault=True)


class BIDSSelectOutputSpec(TraitedSpec):
    bold_files = OutputMultiPath(File)
    mask_files = OutputMultiPath(traits.Either(File, None))


class BIDSSelect(SimpleInterface):
    input_spec = BIDSSelectInputSpec
    output_spec = BIDSSelectOutputSpec

    def _run_interface(self, runtime):
        layout = gb.BIDSLayout(self.inputs.bids_dirs)
        bold_files = []
        mask_files = []
        for info in self.inputs.session_info:
            ents = info['entities']
            selectors = {**self.inputs.selectors, **ents}
            bold_file = layout.get(extensions=['.nii', '.nii.gz'], **selectors)

            if len(bold_file) == 0:
                raise FileNotFoundError(
                    "Could not find BOLD file in {} with entities {}"
                    "".format(self.inputs.bids_dirs, selectors))
            elif len(bold_file) > 1:
                raise ValueError(
                    "Non-unique BOLD file in {} with entities {}.\n"
                    "Matches:\n\t{}"
                    "".format(self.inputs.bids_dirs, selectors,
                              "\n\t".join('{} ({})'.format(f.filename, layout.files[f.filename].entities)
                                          for f in bold_file)))

            # Select exactly matching mask file (may be over-cautious)
            bold_ents = layout.parse_entities(bold_file[0].filename)
            bold_ents['type'] = 'brainmask'
            mask_file = layout.get(extensions=['.nii', '.nii.gz'], **bold_ents)

            bold_files.append(bold_file[0].filename)
            mask_files.append(mask_file[0].filename if mask_file else None)

        self._results['bold_files'] = bold_files
        self._results['mask_files'] = mask_files

        return runtime
