from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..interfaces.bids import (
    ModelSpecLoader, LoadBIDSModel, BIDSSelect, BIDSDataSink)
from ..interfaces.nistats import FirstLevelModel, SecondLevelModel
from ..interfaces.visualizations import (
    DesignPlot, DesignCorrelationPlot, ContrastMatrixPlot, GlassBrainPlot)
from ..interfaces.utils import MergeAll


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
        LoadBIDSModel(bids_dir=bids_dir,
                      preproc_dir=preproc_dir,
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

    select_l1_contrasts = pe.Node(
        niu.Select(index=0),
        name='select_l1_contrasts')

    select_l1_entities = pe.Node(
        niu.Select(index=0),
        name='select_l1_entities')

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

    collate_first_level = pe.Node(MergeAll(['contrast_maps', 'contrast_metadata']),
                                  name='collate_first_level')

    plot_design = pe.MapNode(
        DesignPlot(image_type='svg'),
        iterfield='data',
        name='plot_design')

    def _get_evs(info):
        import pandas as pd
        events = pd.read_hdf(info['events'], key='events')
        return len(events['condition'].unique())

    get_evs = pe.MapNode(niu.Function(function=_get_evs), iterfield='info', name='get_evs')

    plot_corr = pe.MapNode(
        DesignCorrelationPlot(image_type='svg'),
        iterfield=['data', 'explanatory_variables'],
        name='plot_corr')

    plot_contrast_matrix = pe.MapNode(
        ContrastMatrixPlot(image_type='svg'),
        iterfield='data',
        name='plot_contrast_matrix')

    plot_contrasts = pe.MapNode(
        GlassBrainPlot(image_type='png'),
        iterfield='data',
        name='plot_contrasts')

    def join_dict(base_dict, dict_list):
        return [{**base_dict, **iter_dict} for iter_dict in dict_list]

    l1_metadata = pe.MapNode(niu.Function(function=join_dict),
                             iterfield=['base_dict', 'dict_list'],
                             name='l1_metadata')

    select_l2_indices = pe.Node(
        niu.Select(index=1),
        name='select_l2_indices')

    select_l2_contrasts = pe.Node(
        niu.Select(index=1),
        name='select_l2_contrasts')

    # slm = pe.Node(
    slm = pe.MapNode(
        SecondLevelModel(),
        iterfield=['contrast_info', 'contrast_indices'],
        name='slm')

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
    ds_contrast_maps = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_pattern),
        run_without_submitting=True,
        name='ds_contrast_maps')

    contrast_plot_pattern = '[sub-{subject}/][ses-{session}/][sub-{subject}_]'\
        '[ses-{session}_]task-{task}_[run-{run}_]bold[_space-{space}]_' \
        'contrast-{contrast}_ortho.png'
    ds_contrast_plots = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_plot_pattern),
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
        (loader, select_l1_contrasts, [('contrast_info', 'inlist')]),
        (loader, select_l1_entities, [('entities', 'inlist')]),
        (loader, flm, [('session_info', 'session_info')]),
        (loader, ds_model_warnings, [('warnings', 'in_file')]),
        (select_l1_entities, getter,  [('out', 'entities')]),
        (select_l1_contrasts, flm,  [('out', 'contrast_info')]),
        (select_l1_entities, ds_model_warnings,  [('out', 'entities')]),
        (getter, flm, [('bold_files', 'bold_file'),
                       ('mask_files', 'mask_file')]),
        (getter, l1_metadata, [('entities', 'base_dict')]),
        (flm, collate_first_level, [('contrast_maps', 'contrast_maps')]),
        (flm, l1_metadata, [('contrast_metadata', 'dict_list')]),
        (l1_metadata, collate_first_level, [('out', 'contrast_metadata')]),
        (collate_first_level, ds_contrast_maps, [('contrast_maps', 'in_file'),
                                                 ('contrast_metadata', 'entities')]),
        (collate_first_level, plot_contrasts, [('contrast_maps', 'data')]),
        (collate_first_level, ds_contrast_plots, [('contrast_metadata', 'entities')]),
        (plot_contrasts, ds_contrast_plots, [('figure', 'in_file')]),
        (select_l1_entities, ds_design, [('out', 'entities')]),
        (select_l1_entities, ds_corr, [('out', 'entities')]),
        (select_l1_entities, ds_contrasts, [('out', 'entities')]),
        (flm, plot_design, [('design_matrix', 'data')]),
        (plot_design, ds_design, [('figure', 'in_file')]),
        (loader, get_evs, [('session_info', 'info')]),
        (flm, plot_corr, [('design_matrix', 'data')]),
        (get_evs, plot_corr, [('out', 'explanatory_variables')]),
        (plot_corr, ds_corr, [('figure', 'in_file')]),
        (flm, plot_contrast_matrix, [('contrast_matrix', 'data')]),
        (plot_contrast_matrix, ds_contrasts, [('figure', 'in_file')]),
        (loader, select_l2_contrasts, [('contrast_info', 'inlist')]),
        (loader, select_l2_indices, [('contrast_indices', 'inlist')]),
        (flm, slm, [('contrast_maps', 'stat_files')]),
        (l1_metadata, slm, [('out', 'stat_metadata')]),
        (select_l2_contrasts, slm, [('out', 'contrast_info')]),
        (select_l2_indices, slm, [('out', 'contrast_indices')]),
        ])

    return wf
