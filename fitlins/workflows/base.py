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

    # Find the appropriate model file(s)
    specs = ModelSpecLoader(bids_dir=bids_dir)
    if model is not None:
        specs.inputs.model = model

    all_models = specs.run().outputs.model_spec
    if not all_models:
        raise RuntimeError("Unable to find or construct models")

    #
    # Load and run the model
    #

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

    # Because pybids generates the entire model in one go, we will need
    # various helper nodes to select the correct portions of the model
    select_l1_entities = pe.Node(niu.Select(index=0), name='select_l1_entities')

    # Select preprocessed BOLD series to analyze
    getter = pe.Node(
        BIDSSelect(bids_dir=bids_dir,
                   selectors={'type': 'preproc', 'space': space}),
        name='getter')

    if preproc_dir is not None:
        getter.inputs.preproc_dir = preproc_dir

    select_l1_contrasts = pe.Node(niu.Select(index=0), name='select_l1_contrasts')

    # Run first level model
    l1_model = pe.MapNode(
        FirstLevelModel(),
        iterfield=['session_info', 'contrast_info', 'bold_file', 'mask_file'],
        name='l1_model')

    def join_dict(base_dict, dict_list):
        return [{**base_dict, **iter_dict} for iter_dict in dict_list]

    # Accumulate metadata
    l1_metadata = pe.MapNode(niu.Function(function=join_dict),
                             iterfield=['base_dict', 'dict_list'],
                             name='l1_metadata')

    # Squash the results of MapNodes that may have generated multiple maps
    # into single lists.
    # Do the same with corresponding metadata - interface will complain if shapes mismatch
    collate_first_level = pe.Node(MergeAll(['contrast_maps', 'contrast_metadata']),
                                  name='collate_first_level')

    select_l2_entities = pe.Node(niu.Select(index=1), name='select_l2_entities')
    select_l2_indices = pe.Node(niu.Select(index=1), name='select_l2_indices')
    select_l2_contrasts = pe.Node(niu.Select(index=1), name='select_l2_contrasts')

    # Run second-level model
    # TODO: Iterate over all higher levels
    l2_model = pe.MapNode(
        SecondLevelModel(),
        iterfield=['contrast_info', 'contrast_indices'],
        name='l2_model')

    collate_second_level = pe.Node(MergeAll(['contrast_maps', 'contrast_metadata']),
                                   name='collate_second_level')

    #
    # Plotting
    #

    plot_design = pe.MapNode(
        DesignPlot(image_type='svg'),
        iterfield='data',
        name='plot_design')

    def _get_evs(info):
        import pandas as pd
        events = pd.read_hdf(info['events'], key='events')
        return len(events['condition'].unique())

    # Number of explanatory variables is used to mark off sections of the
    # correlation matrix
    get_evs = pe.MapNode(niu.Function(function=_get_evs), iterfield='info', name='get_evs')

    plot_corr = pe.MapNode(
        DesignCorrelationPlot(image_type='svg'),
        iterfield=['data', 'explanatory_variables'],
        name='plot_corr')

    plot_l1_contrast_matrix = pe.MapNode(
        ContrastMatrixPlot(image_type='svg'),
        iterfield='data',
        name='plot_l1_contrast_matrix')

    plot_l1_contrasts = pe.MapNode(
        GlassBrainPlot(image_type='png'),
        iterfield='data',
        name='plot_l1_contrasts')

    plot_l2_contrast_matrix = pe.MapNode(
        ContrastMatrixPlot(image_type='svg'),
        iterfield='data',
        name='plot_l2_contrast_matrix')

    plot_l2_contrasts = pe.MapNode(
        GlassBrainPlot(image_type='png'),
        iterfield='data',
        name='plot_l2_contrasts')

    #
    # HTML snippets to be included directly in report, not
    # saved as individual derivative files
    #

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

    #
    # Derivatives
    #

    # NIfTIs
    contrast_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold' \
        '[_space-{space}]_contrast-{contrast}_{type<effect|stat>}.nii.gz'
    ds_l1_contrast_maps = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_pattern),
        run_without_submitting=True,
        name='ds_l1_contrast_maps')

    ds_l2_contrast_maps = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_pattern),
        run_without_submitting=True,
        name='ds_l2_contrast_maps')

    # Images
    image_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold_' \
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

    ds_l1_contrasts = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     fixed_entities={'type': 'contrasts'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_l1_contrasts')

    ds_l2_contrasts = pe.MapNode(
        BIDSDataSink(base_directory=out_dir,
                     fixed_entities={'type': 'contrasts'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_l2_contrasts')

    contrast_plot_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold' \
        '[_space-{space}]_contrast-{contrast}_ortho.png'
    ds_l1_contrast_plots = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_plot_pattern),
        run_without_submitting=True,
        name='ds_l1_contrast_plots')

    ds_l2_contrast_plots = pe.Node(
        BIDSDataSink(base_directory=out_dir,
                     path_patterns=contrast_plot_pattern),
        run_without_submitting=True,
        name='ds_l2_contrast_plots')

    #
    # Connections
    #

    wf.connect([
        #
        # Modeling
        #
        (loader, select_l1_entities, [('entities', 'inlist')]),

        (select_l1_entities, getter,  [('out', 'entities')]),

        (loader, select_l1_contrasts, [('contrast_info', 'inlist')]),

        (loader, l1_model, [('session_info', 'session_info')]),
        (getter, l1_model, [('bold_files', 'bold_file'),
                            ('mask_files', 'mask_file')]),
        (select_l1_contrasts, l1_model,  [('out', 'contrast_info')]),

        (getter, l1_metadata, [('entities', 'base_dict')]),
        (l1_model, l1_metadata, [('contrast_metadata', 'dict_list')]),

        (l1_model, collate_first_level, [('contrast_maps', 'contrast_maps')]),
        (l1_metadata, collate_first_level, [('out', 'contrast_metadata')]),

        (loader, select_l2_entities, [('entities', 'inlist')]),
        (loader, select_l2_indices, [('contrast_indices', 'inlist')]),
        (loader, select_l2_contrasts, [('contrast_info', 'inlist')]),

        (l1_model, l2_model, [('contrast_maps', 'stat_files')]),
        (l1_metadata, l2_model, [('out', 'stat_metadata')]),
        (select_l2_indices, l2_model, [('out', 'contrast_indices')]),
        (select_l2_contrasts, l2_model, [('out', 'contrast_info')]),

        (l2_model, collate_second_level, [('contrast_maps', 'contrast_maps'),
                                          ('contrast_metadata', 'contrast_metadata')]),

        #
        # Plotting
        #
        (l1_model, plot_design, [('design_matrix', 'data')]),

        (loader, get_evs, [('session_info', 'info')]),
        (l1_model, plot_corr, [('design_matrix', 'data')]),
        (get_evs, plot_corr, [('out', 'explanatory_variables')]),

        (l1_model, plot_l1_contrast_matrix, [('contrast_matrix', 'data')]),

        (collate_first_level, plot_l1_contrasts, [('contrast_maps', 'data')]),

        (l2_model, plot_l2_contrast_matrix, [('contrast_matrix', 'data')]),

        (collate_second_level, plot_l2_contrasts, [('contrast_maps', 'data')]),

        #
        # HTML snippets
        #
        (loader, ds_model_warnings, [('warnings', 'in_file')]),
        (select_l1_entities, ds_model_warnings,  [('out', 'entities')]),

        #
        # Derivatives
        #
        (collate_first_level, ds_l1_contrast_maps, [('contrast_maps', 'in_file'),
                                                    ('contrast_metadata', 'entities')]),

        (collate_second_level, ds_l2_contrast_maps, [('contrast_maps', 'in_file'),
                                                     ('contrast_metadata', 'entities')]),

        (select_l1_entities, ds_design, [('out', 'entities')]),
        (plot_design, ds_design, [('figure', 'in_file')]),

        (select_l1_entities, ds_corr, [('out', 'entities')]),
        (plot_corr, ds_corr, [('figure', 'in_file')]),


        (select_l1_entities, ds_l1_contrasts, [('out', 'entities')]),
        (plot_l1_contrast_matrix, ds_l1_contrasts, [('figure', 'in_file')]),

        (collate_first_level, ds_l1_contrast_plots, [('contrast_metadata', 'entities')]),
        (plot_l1_contrasts, ds_l1_contrast_plots, [('figure', 'in_file')]),

        (select_l2_entities, ds_l2_contrasts, [('out', 'entities')]),
        (plot_l2_contrast_matrix, ds_l2_contrasts, [('figure', 'in_file')]),

        (collate_second_level, ds_l2_contrast_plots, [('contrast_metadata', 'entities')]),
        (plot_l2_contrasts, ds_l2_contrast_plots, [('figure', 'in_file')]),
        ])

    return wf
