from nipype.pipeline import engine as pe
from ..interfaces.bids import LoadLevel1BIDSModel, BIDSSelect, BIDSDataSink
from ..interfaces.nistats import FirstLevelModel


def init_fitlins_wf(bids_dir, preproc_dir, base_dir, name='fitlins_wf'):
    wf = pe.Workflow(name=name, base_dir=base_dir)

    loader = pe.Node(
        LoadLevel1BIDSModel(bids_dirs=[bids_dir, preproc_dir]),
        name='loader')
    getter = pe.Node(
        BIDSSelect(bids_dirs=preproc_dir,
                   selectors={'type': 'preproc',
                              'space': 'MNI152NLin2009cAsym'}),
        name='getter')
    flm = pe.MapNode(
        FirstLevelModel(),
        iterfield=['session_info', 'contrast_info', 'bold_file', 'mask_file'],
        name='flm')

    sink_dms = pe.MapNode(
        BIDSDataSink(
            base_directory='/work/derivatives/fitlins',
            fixed_entities={'type': 'design'},
            path_patterns='sub-{subject}/[ses-{session}/]sub-{subject}_'
                          '[ses-{session}_]task-{task}_bold_{type<design>}.svg'
            ),
        iterfield=['entities', 'in_file'],
        name='sink_dms')

    def _get_ents(session_infos):
        return [info['entities'] for info in session_infos]

    wf.connect([
        (loader, getter,  [('session_info', 'session_info')]),
        (loader, flm, [('session_info', 'session_info'),
                       ('contrast_info', 'contrast_info')]),
        (getter, flm, [('bold_files', 'bold_file'),
                       ('mask_files', 'mask_file')]),
        (loader, sink_dms, [(('session_info', _get_ents), 'entities')]),
        (flm, sink_dms, [('design_matrix_plot', 'in_file')]),
        ])

    return wf
