from pathlib import Path
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..interfaces.bids import (
    ModelSpecLoader, LoadBIDSModel, BIDSSelect, BIDSDataSink)
from ..interfaces.nistats import FirstLevelModel, SecondLevelModel
from ..interfaces.visualizations import (
    DesignPlot, DesignCorrelationPlot, ContrastMatrixPlot, GlassBrainPlot)
from ..interfaces.utils import MergeAll


def join_dict(base_dict, dict_list):
    return [{**base_dict, **iter_dict} for iter_dict in dict_list]


def init_fitlins_wf(bids_dir, derivatives, out_dir, space, exclude_pattern=None,
                    include_pattern=None, model=None, participants=None,
                    base_dir=None, name='fitlins_wf'):
    wf = pe.Workflow(name=name, base_dir=base_dir)

    # Find the appropriate model file(s)
    specs = ModelSpecLoader(bids_dir=bids_dir)
    if model is not None:
        specs.inputs.model = model

    model_dict = specs.run().outputs.model_spec
    if not model_dict:
        raise RuntimeError("Unable to find or construct models")
    if isinstance(model_dict, list):
        raise RuntimeError("Currently unable to run multiple models in parallel - "
                           "please specify model")

    #
    # Load and run the model
    #

    loader = pe.Node(
        LoadBIDSModel(bids_dir=bids_dir,
                      derivatives=derivatives,
                      model=model_dict),
        name='loader')

    if exclude_pattern is not None:
        loader.inputs.exclude_pattern = exclude_pattern
    if include_pattern is not None:
        loader.inputs.include_pattern = include_pattern
    if participants is not None:
        loader.inputs.selectors = {'subject': participants}

    # Select preprocessed BOLD series to analyze
    getter = pe.Node(
        BIDSSelect(
            bids_dir=bids_dir, derivatives=derivatives,
            selectors={
                'type': 'preproc', 'suffix': 'bold', 'space': space}),
        name='getter')

    # Accumulate metadata
    l1_metadata = pe.MapNode(
        niu.Function(function=join_dict),
        iterfield=['base_dict', 'dict_list'],
        name='l1_metadata',
        run_without_submitting=True)

    l1_model = pe.MapNode(
        FirstLevelModel(),
        iterfield=['session_info', 'contrast_info', 'bold_file', 'mask_file'],
        name='l1_model')

    # Set up common patterns
    image_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold_' \
        '{suffix<design|corr|contrasts>}.svg'
    contrast_plot_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold' \
        '[_space-{space}]_contrast-{contrast}_ortho.png'
    contrast_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_bold' \
        '[_space-{space}]_contrast-{contrast}_{suffix<effect|stat>}.nii.gz'

    # Set up general interfaces
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

    plot_design = pe.MapNode(
        DesignPlot(image_type='svg'),
        iterfield='data',
        name='plot_design')

    plot_l1_contrast_matrix = pe.MapNode(
        ContrastMatrixPlot(image_type='svg'),
        iterfield=['data', 'contrast_info'],
        name='plot_l1_contrast_matrix')

    ds_design = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'suffix': 'design'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_design')

    ds_l1_contrasts = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'suffix': 'contrasts'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_l1_contrasts')

    #
    # General Connections
    #
    wf.connect([
        (loader, ds_model_warnings, [('warnings', 'in_file')]),
        (loader, l1_model, [('session_info', 'session_info')]),
        (getter, l1_model, [('bold_files', 'bold_file'),
                            ('mask_files', 'mask_file')]),
        (getter, l1_metadata, [('entities', 'base_dict')]),
        (l1_model, l1_metadata, [('contrast_metadata', 'dict_list')]),
        (l1_model, plot_design, [('design_matrix', 'data')]),
        ])

    models = []
    for ix, step in enumerate(step['Level'] for step in model_dict['Steps']):
        # Set up elements common across levels

        #
        # Because pybids generates the entire model in one go, we will need
        # various helper nodes to select the correct portions of the model
        #

        level = 'l{:d}'.format(ix + 1)

        select_entities = pe.Node(
            niu.Select(index=ix),
            name='select_{}_entities'.format(level),
            run_without_submitting=True)

        select_contrasts = pe.Node(
            niu.Select(index=ix),
            name='select_{}_contrasts'.format(level),
            run_without_submitting=True)

        # Squash the results of MapNodes that may have generated multiple maps
        # into single lists.
        # Do the same with corresponding metadata - interface will complain if shapes mismatch
        collate = pe.Node(
            MergeAll(['contrast_maps', 'contrast_metadata']),
            name='collate_{}'.format(level),
            run_without_submitting=True)

        #
        # Plotting
        #

        plot_contrasts = pe.MapNode(
            GlassBrainPlot(image_type='png'),
            iterfield='data',
            name='plot_{}_contrasts'.format(level))

        #
        # Derivatives
        #

        ds_contrast_maps = pe.Node(
            BIDSDataSink(base_directory=out_dir,
                         path_patterns=contrast_pattern),
            run_without_submitting=True,
            name='ds_{}_contrast_maps'.format(level))

        ds_contrast_plots = pe.Node(
            BIDSDataSink(base_directory=out_dir,
                         path_patterns=contrast_plot_pattern),
            run_without_submitting=True,
            name='ds_{}_contrast_plots'.format(level))

        if ix == 0:
            model = l1_model
            wf.connect([
                (loader, select_entities, [('entities', 'inlist')]),
                (select_entities, getter,  [('out', 'entities')]),
                (select_entities, ds_model_warnings,  [('out', 'entities')]),
                (l1_metadata, collate, [('out', 'contrast_metadata')]),
                (select_entities, ds_design, [('out', 'entities')]),
                (plot_design, ds_design, [('figure', 'in_file')]),
                (select_contrasts, plot_l1_contrast_matrix,  [('out', 'contrast_info')]),
                (model, plot_l1_contrast_matrix,  [('design_matrix', 'data')]),
                (select_entities, ds_l1_contrasts, [('out', 'entities')]),
                (plot_l1_contrast_matrix, ds_l1_contrasts,  [('figure', 'in_file')]),
            ])

        #  Set up higher levels
        else:
            model = pe.MapNode(
                SecondLevelModel(),
                iterfield=['contrast_info'],
                name='{}_model'.format(level))

            wf.connect([
                (models[-1], model, [('contrast_maps', 'stat_files')]),
                (l1_metadata, model, [('out', 'stat_metadata')]),
                (model, collate, [('contrast_metadata', 'contrast_metadata')]),
            ])

        wf.connect([
            (loader, select_contrasts, [('contrast_info', 'inlist')]),
            (select_contrasts, model,  [('out', 'contrast_info')]),
            (model, collate, [('contrast_maps', 'contrast_maps')]),
            (collate, plot_contrasts, [('contrast_maps', 'data')]),
            (collate, ds_contrast_maps, [('contrast_maps', 'in_file'),
                                         ('contrast_metadata', 'entities')]),
            (collate, ds_contrast_plots, [('contrast_metadata', 'entities')]),
            (plot_contrasts, ds_contrast_plots, [('figure', 'in_file')]),

            ])

        models.append(model)

    return wf
