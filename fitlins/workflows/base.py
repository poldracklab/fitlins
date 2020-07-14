from pathlib import Path
import warnings


def init_fitlins_wf(database_path, out_dir, analysis_level, space,
                    desc=None, model=None, participants=None,
                    smoothing=None, drop_missing=False,
                    base_dir=None, name='fitlins_wf'):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from ..interfaces.bids import (
        ModelSpecLoader, LoadBIDSModel, BIDSSelect, BIDSDataSink)
    from ..interfaces.nistats import DesignMatrix, FirstLevelModel, SecondLevelModel
    from ..interfaces.visualizations import (
        DesignPlot, DesignCorrelationPlot, ContrastMatrixPlot, GlassBrainPlot)
    from ..interfaces.utils import MergeAll, CollateWithMetadata

    wf = pe.Workflow(name=name, base_dir=base_dir)

    # Find the appropriate model file(s)
    specs = ModelSpecLoader(database_path=database_path)
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
        LoadBIDSModel(database_path=database_path,
                      model=model_dict,
                      selectors={'desc': desc, 'space': space}),
        name='loader')

    if participants is not None:
        loader.inputs.selectors['subject'] = participants
    if database_path is not None:
        loader.inputs.database_path = database_path

    # Select preprocessed BOLD series to analyze
    getter = pe.Node(
        BIDSSelect(
            database_path=database_path,
            selectors={'suffix': 'bold',
                       'desc': desc,
                       'space': space,
                       'extension': ['.nii', '.nii.gz', '.dtseries.nii', '.func.gii']}),
        name='getter')

    if smoothing:
        smoothing_params = smoothing.split(':', 2)
        # Convert old style and warn; this should turn into an (informative) error around 0.5.0
        if smoothing_params[0] == 'iso':
            smoothing_params = (smoothing_params[1], 'l1', smoothing_params[0])
            warnings.warn(
                "The format for smoothing arguments has changed. Please use "
                f"{':'.join(smoothing_params)} instead of {smoothing}.", FutureWarning)
        # Add defaults to simplify later logic
        if len(smoothing_params) == 1:
            smoothing_params.extend(('l1', 'iso'))
        elif len(smoothing_params) == 2:
            smoothing_params.append('iso')

        smoothing_fwhm, smoothing_level, smoothing_type = smoothing_params
        smoothing_fwhm = float(smoothing_fwhm)
        if smoothing_type not in ('iso'):
            raise ValueError(f"Unknown smoothing type {smoothing_type}")

        # Check that smmoothing level exists in model
        if smoothing_level.lower().startswith("l"):
            if int(smoothing_level[1:]) > len(model_dict['Steps']):
                raise ValueError(f"Invalid smoothing level {smoothing_level}")
        elif smoothing_level.lower() not in (step['Level'].lower()
                                             for step in model_dict['Steps']):
            raise ValueError(f"Invalid smoothing level {smoothing_level}")

    design_matrix = pe.MapNode(
        DesignMatrix(drop_missing=drop_missing),
        iterfield=['session_info', 'bold_file'],
        name='design_matrix')

    l1_model = pe.MapNode(
        FirstLevelModel(),
        iterfield=['design_matrix', 'contrast_info', 'bold_file', 'mask_file'],
        mem_gb=3,
        name='l1_model')

    def _deindex(tsv):
        from pathlib import Path
        import pandas as pd
        out_tsv = str(Path.cwd() / Path(tsv).name)
        pd.read_csv(tsv, sep='\t', index_col=0).to_csv(out_tsv, sep='\t', index=False)
        return out_tsv

    deindex_tsv = pe.MapNode(niu.Function(function=_deindex),
                             iterfield=['tsv'], name='deindex_tsv')

    # Set up common patterns
    image_pattern = 'reports/[sub-{subject}/][ses-{session}/]figures/[run-{run}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_' \
        '{suffix<design|corr|contrasts>}{extension<.svg>|.svg}'
    contrast_plot_pattern = 'reports/[sub-{subject}/][ses-{session}/]figures/[run-{run}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_' \
        'contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F|FEMA>}_ortho{extension<.png>|.png}'
    design_matrix_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<design>}{extension<.tsv>|.tsv}'
    contrast_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_' \
        'contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F|FEMA>}_' \
        'statmap{extension<.nii.gz|.dscalar.nii>}'
    model_map_pattern = '[sub-{subject}/][ses-{session}/]' \
        '[sub-{subject}_][ses-{session}_]task-{task}[_acq-{acquisition}]' \
        '[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_space-{space}]_' \
        'stat-{stat<rSquare|logLikelihood>}_statmap{extension<.nii.gz|.dscalar.nii>}'
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

    plot_corr = pe.MapNode(
        DesignCorrelationPlot(image_type='svg'),
        iterfield=['data', 'contrast_info'],
        name='plot_corr')

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

    ds_design_matrix = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'suffix': 'design'},
                     path_patterns=design_matrix_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_design_matrix')

    ds_corr = pe.MapNode(
        BIDSDataSink(base_directory=out_dir, fixed_entities={'suffix': 'corr'},
                     path_patterns=image_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_corr')

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
        (loader, design_matrix, [('design_info', 'session_info')]),
        (getter, design_matrix, [('bold_files', 'bold_file')]),
        (getter, l1_model, [('bold_files', 'bold_file'),
                            ('mask_files', 'mask_file')]),
        (design_matrix, l1_model, [('design_matrix', 'design_matrix')]),
        (design_matrix, plot_design, [('design_matrix', 'data')]),
        (design_matrix, plot_l1_contrast_matrix,  [('design_matrix', 'data')]),
        (design_matrix, plot_corr,  [('design_matrix', 'data')]),
        (design_matrix, deindex_tsv, [('design_matrix', 'tsv')]),
        (deindex_tsv, ds_design_matrix, [('out', 'in_file')]),
        ])

    stage = None
    model = l1_model
    for ix, step in enumerate(step['Level'] for step in model_dict['Steps']):
        # Set up elements common across levels

        #
        # Because pybids generates the entire model in one go, we will need
        # various helper nodes to select the correct portions of the model
        #

        level = 'l{:d}'.format(ix + 1)

        # TODO: No longer used at higher level, suggesting we can simply return
        # entities from loader as a single list
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
            MergeAll(['effect_maps', 'variance_maps', 'stat_maps', 'zscore_maps',
                      'pvalue_maps', 'contrast_metadata'],
                     check_lengths=(not drop_missing)),
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

        collate_outputs = pe.Node(
            CollateWithMetadata(
                fields=['effect_maps', 'variance_maps', 'stat_maps', 'pvalue_maps', 'zscore_maps'],
                field_to_metadata_map={
                    'effect_maps': {'stat': 'effect'},
                    'variance_maps': {'stat': 'variance'},
                    'pvalue_maps': {'stat': 'p'},
                    'zscore_maps': {'stat': 'z'},
                }),
            name=f'collate_{level}_outputs')

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
            ds_model_maps = pe.Node(
                BIDSDataSink(base_directory=out_dir,
                             path_patterns=model_map_pattern),
                run_without_submitting=True,
                name='ds_{}_model_maps'.format(level))

            collate_mm = pe.Node(
                MergeAll(['model_maps', 'model_metadata'],
                         check_lengths=(not drop_missing)),
                name='collate_mm_{}'.format(level),
                run_without_submitting=True)

            wf.connect([
                (loader, select_entities, [('entities', 'inlist')]),
                (select_entities, getter,  [('out', 'entities')]),
                (select_entities, ds_model_warnings,  [('out', 'entities')]),
                (select_entities, ds_design, [('out', 'entities')]),
                (select_entities, ds_design_matrix, [('out', 'entities')]),
                (plot_design, ds_design, [('figure', 'in_file')]),
                (select_contrasts, plot_l1_contrast_matrix,  [('out', 'contrast_info')]),
                (select_contrasts, plot_corr,  [('out', 'contrast_info')]),
                (select_entities, ds_l1_contrasts, [('out', 'entities')]),
                (select_entities, ds_corr, [('out', 'entities')]),
                (plot_l1_contrast_matrix, ds_l1_contrasts,  [('figure', 'in_file')]),
                (plot_corr, ds_corr,  [('figure', 'in_file')]),
                (model, collate_mm, [('model_maps', 'model_maps'),
                                     ('model_metadata', 'model_metadata')]),
                (collate_mm, ds_model_maps, [('model_maps', 'in_file'),
                                             ('model_metadata', 'entities')]),
            ])

        #  Set up higher levels
        else:
            model = pe.MapNode(
                SecondLevelModel(),
                iterfield=['contrast_info'],
                name='{}_model'.format(level))

            wf.connect([
                (stage, model, [('effect_maps', 'effect_maps'),
                                ('variance_maps', 'variance_maps'),
                                ('contrast_metadata', 'stat_metadata')]),
            ])

        if smoothing and smoothing_level in (step, level):
            model.inputs.smoothing_fwhm = smoothing_fwhm

        wf.connect([
            (loader, select_contrasts, [('contrast_info', 'inlist')]),
            (select_contrasts, model,  [('out', 'contrast_info')]),
            (model, collate, [('effect_maps', 'effect_maps'),
                              ('variance_maps', 'variance_maps'),
                              ('stat_maps', 'stat_maps'),
                              ('zscore_maps', 'zscore_maps'),
                              ('pvalue_maps', 'pvalue_maps'),
                              ('contrast_metadata', 'contrast_metadata')]),
            (collate, collate_outputs, [
                ('contrast_metadata', 'metadata'),
                ('effect_maps', 'effect_maps'),
                ('variance_maps', 'variance_maps'),
                ('stat_maps', 'stat_maps'),
                ('zscore_maps', 'zscore_maps'),
                ('pvalue_maps', 'pvalue_maps'),
                ]),
            (collate, plot_contrasts, [('stat_maps', 'data')]),
            (collate_outputs, ds_contrast_maps, [('out', 'in_file'),
                                                 ('metadata', 'entities')]),
            (collate, ds_contrast_plots, [('contrast_metadata', 'entities')]),
            (plot_contrasts, ds_contrast_plots, [('figure', 'in_file')]),

            ])

        stage = model
        if step == analysis_level:
            break

    return wf
