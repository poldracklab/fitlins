from nipype.pipeline import engine as pe
from ..interfaces.bids import LoadLevel1BIDSModel, BIDSSelect, BIDSDataSink
from ..interfaces.nistats import FirstLevelModel


def init_fitlins_wf(bids_dir, preproc_dir, out_dir,
                    base_dir=None, name='fitlins_wf'):
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

    image_pattern = 'sub-{subject}/[ses-{session}/]sub-{subject}_' \
        '[ses-{session}_]task-{task}_bold_{type<design|corr|contrasts>}.svg'
    ds_design = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'type': 'design'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        name='ds_design')

    ds_corr = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'type': 'corr'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        name='ds_corr')

    ds_contrasts = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     fixed_entities={'type': 'contrasts'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        name='ds_contrasts')

    wf.connect([
        (loader, getter,  [('entities', 'entities')]),
        (loader, flm, [('session_info', 'session_info'),
                       ('contrast_info', 'contrast_info')]),
        (getter, flm, [('bold_files', 'bold_file'),
                       ('mask_files', 'mask_file')]),
        (loader, ds_design, [('entities', 'entities')]),
        (loader, ds_corr, [('entities', 'entities')]),
        (loader, ds_contrasts, [('entities', 'entities')]),
        (flm, ds_design, [('design_matrix_plot', 'in_file')]),
        (flm, ds_corr, [('correlation_matrix_plot', 'in_file')]),
        (flm, ds_contrasts, [('contrast_matrix_plot', 'in_file')]),
        ])

    return wf
