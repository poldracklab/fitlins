import os
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, SimpleInterface,
    InputMultiPath, File, Directory,
    traits, isdefined
    )
from bids import grabbids as gb, analysis as ba


class LoadLevel1BIDSModelInputSpec(BaseInterfaceInputSpec):
    bids_dirs = InputMultiPath(Directory(exists=True),
                               mandatory=True,
                               desc='BIDS dataset root directories')
    model = File(exists=True, desc='Model filename')
    selectors = traits.Dict(desc='Limit collected sessions')


class LoadLevel1BIDSModelOutputSpec(TraitedSpec):
    session_info = traits.List(traits.Dict())
    contrast_info = traits.List(File())


class LoadLevel1BIDSModel(SimpleInterface):
    input_spec = LoadLevel1BIDSModelInputSpec
    output_spec = LoadLevel1BIDSModelOutputSpec

    def _run_interface(self, runtime):
        layout = gb.BIDSLayout(self.inputs.bids_dirs)
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

        self._results['session_info'] = session_info
        self._results['contrast_info'] = contrast_info
        return runtime
