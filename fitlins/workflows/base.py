from pathlib import Path
from nipype.pipeline import engine as pe
from ..interfaces.bids import (
    ModelSpecLoader, LoadLevel1BIDSModel, BIDSSelect, BIDSDataSink)
from ..interfaces.nistats import FirstLevelModel


def init_fitlins_wf(bids_dir, preproc_dir, out_dir, space, exclude_pattern=None,
                    include_pattern=None, model=None, participants=None,
                    base_dir=None, name='fitlins_wf'):
    wf = pe.Workflow(name=name, base_dir=base_dir)

    specs = ModelSpecLoader(bids_dir=bids_dir)
    if model is not None:
        specs.inputs.model = model

    all_models = specs.run().outputs.model_spec
    if not all_models:
        raise RuntimeError("Unable to find or construct models")

    selectors = {'subject': participants} if participants is not None else {}

    loader = pe.Node(
        LoadLevel1BIDSModel(bids_dir=bids_dir,
                            selectors=selectors),
        name='loader')

    if preproc_dir is not None:
        loader.inputs.preproc_dir = preproc_dir
    if exclude_pattern is not None:
        loader.inputs.exclude_pattern = exclude_pattern
    if include_pattern is not None:
        loader.inputs.include_pattern = include_pattern

    if isinstance(all_models, list):
        loader.iterables = ('model', all_models)
    else:
        loader.inputs.model = all_models

    getter = pe.Node(
        BIDSSelect(bids_dir=bids_dir,
                   selectors={'type': 'preproc', 'space': space}),
        name='getter')

    if preproc_dir is not None:
        getter.inputs.preproc_dir = preproc_dir

    flm = pe.MapNode(
        FirstLevelModel(),
        iterfield=['session_info', 'contrast_info', 'bold_file', 'mask_file'],
        name='flm')

    reportlet_dir = Path(base_dir) / 'reportlets' / 'fitlins'
    reportlet_dir.mkdir(parents=True, exist_ok=True)


    snippet_pattern = '[sub-{subject}/][ses-{session}/][sub-{subject}_]' \
        '[ses-{session}_]task-{task}_[run-{run}_]snippet.html'
    ds_model_warnings = pe.MapNode(
        BIDSDataSink(base_directory=str(reportlet_dir),
                     path_patterns=snippet_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_model_warning')

    contrast_pattern = '[sub-{subject}/][ses-{session}/][sub-{subject}_]' \
        '[ses-{session}_]task-{task}_[run-{run}_]bold[_space-{space}]_' \
        'contrast-{contrast}_{type<effect>}.nii.gz'
    ds_estimate_maps = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_pattern),
        iterfield=['fixed_entities', 'entities', 'in_file'],
        run_without_submitting=True,
        name='ds_estimate_maps')
    ds_contrast_maps = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_pattern),
        iterfield=['fixed_entities', 'entities', 'in_file'],
        run_without_submitting=True,
        name='ds_contrast_maps')

    contrast_plot_pattern = '[sub-{subject}/][ses-{session}/][sub-{subject}_]'\
        '[ses-{session}_]task-{task}_[run-{run}_]bold[_space-{space}]_' \
        'contrast-{contrast}_ortho.png'
    ds_estimate_plots = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_plot_pattern),
        iterfield=['fixed_entities', 'entities', 'in_file'],
        run_without_submitting=True,
        name='ds_estimate_plots')
    ds_contrast_plots = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_plot_pattern),
        iterfield=['fixed_entities', 'entities', 'in_file'],
        run_without_submitting=True,
        name='ds_contrast_plots')

    image_pattern = 'sub-{subject}/[ses-{session}/]sub-{subject}_' \
        '[ses-{session}_]task-{task}_[run-{run}_]bold_' \
        '{type<design|corr|contrasts>}.svg'
    ds_design = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'type': 'design'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_design')

    ds_corr = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'type': 'corr'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_corr')

    ds_contrasts = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     fixed_entities={'type': 'contrasts'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_contrasts')

    wf.connect([
        (loader, getter,  [('entities', 'entities')]),
        (loader, flm, [('session_info', 'session_info'),
                       ('contrast_info', 'contrast_info')]),
        (loader, ds_model_warnings, [('entities', 'entities'),
                                     ('warnings', 'in_file')]),
        (getter, flm, [('bold_files', 'bold_file'),
                       ('mask_files', 'mask_file')]),
        (getter, ds_estimate_maps, [('entities', 'fixed_entities')]),
        (getter, ds_contrast_maps, [('entities', 'fixed_entities')]),
        (flm, ds_estimate_maps, [('estimate_maps', 'in_file'),
                                 ('estimate_metadata', 'entities')]),
        (flm, ds_contrast_maps, [('contrast_maps', 'in_file'),
                                 ('contrast_metadata', 'entities')]),
        (getter, ds_estimate_plots, [('entities', 'fixed_entities')]),
        (getter, ds_contrast_plots, [('entities', 'fixed_entities')]),
        (flm, ds_estimate_plots, [('estimate_map_plots', 'in_file'),
                                  ('estimate_metadata', 'entities')]),
        (flm, ds_contrast_plots, [('contrast_map_plots', 'in_file'),
                                  ('contrast_metadata', 'entities')]),
        (loader, ds_design, [('entities', 'entities')]),
        (loader, ds_corr, [('entities', 'entities')]),
        (loader, ds_contrasts, [('entities', 'entities')]),
        (flm, ds_design, [('design_matrix_plot', 'in_file')]),
        (flm, ds_corr, [('correlation_matrix_plot', 'in_file')]),
        (flm, ds_contrasts, [('contrast_matrix_plot', 'in_file')]),
        ])

    return wf
