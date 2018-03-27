import nibabel as nb
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
    contrast_info = traits.List(traits.Any())


class LoadLevel1BIDSModel(SimpleInterface):
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

        selectors = self.inputs.selectors if isdefined(self.inputs.selectors) else {}

        analysis = ba.Analysis(model=model_fname, layout=layout)
        selectors.update(analysis.model['input'])
        analysis.setup(**selectors)
        block = analysis.blocks[0]

        session_info = []
        for paradigm, _, ents in block.get_design_matrix(block.model['HRF_variables'],
                                                         mode='sparse'):
            info = {'entities': ents}

            bold_files = layout.get(type='bold',
                                    extensions=['.nii', '.nii.gz'],
                                    **ents)
            if len(bold_files) != 1:
                raise ValueError('Too many BOLD files found')

            fname = bold_files[0].filename

            # Required field in seconds
            TR = layout.get_metadata(fname)['RepetitionTime']
            # vols = nb.load(fname).shape[3]

            _, confounds, _ = block.get_design_matrix(mode='dense',
                                                      sampling_rate=1/TR,
                                                      **ents)[0]

            info['events'] = paradigm
            info['confounds'] = confounds

            session_info.append(info)

        self._results['session_info'] = session_info
        return runtime
