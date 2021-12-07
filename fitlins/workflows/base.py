import warnings

from collections import OrderedDict
from pathlib import Path


def init_fitlins_wf(
    database_path,
    out_dir,
    graph,
    analysis_level,
    space,
    desc=None,
    model=None,
    participants=None,
    smoothing=None,
    drop_missing=False,
    estimator=None,
    errorts=False,
    drift_model=None,
    base_dir=None,
    name='fitlins_wf',
):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu
    from ..interfaces.bids import ModelSpecLoader, LoadBIDSModel, BIDSSelect, BIDSDataSink
    from ..interfaces.nistats import DesignMatrix, SecondLevelModel
    from ..interfaces.visualizations import (
        DesignPlot,
        DesignCorrelationPlot,
        ContrastMatrixPlot,
        GlassBrainPlot,
    )
    from ..interfaces.utils import MergeAll, CollateWithMetadata
    from ..utils import snake_to_camel

    if estimator == 'afni':
        from ..interfaces.afni import FirstLevelModel
    else:
        from ..interfaces.nistats import FirstLevelModel

    wf = pe.Workflow(name=name, base_dir=base_dir)

    # Find the appropriate model file(s)
    specs = ModelSpecLoader(database_path=database_path)
    if model is not None:
        specs.inputs.model = model

    model_dict = specs.run().outputs.model_spec
    if not model_dict:
        raise RuntimeError("Unable to find or construct models")
    if isinstance(model_dict, list):
        raise RuntimeError(
            "Currently unable to run multiple models in parallel - " "please specify model"
        )
    #
    # Load and run the model
    #
    loader = pe.Node(
        LoadBIDSModel(
            database_path=database_path, model=model_dict, selectors={'desc': desc, 'space': space}
        ),
        name='loader',
    )

    if participants is not None:
        loader.inputs.selectors['subject'] = participants
    if database_path is not None:
        loader.inputs.database_path = database_path

    # Select preprocessed BOLD series to analyze
    getter = pe.Node(
        BIDSSelect(
            database_path=database_path,
            selectors={
                'suffix': 'bold',
                'desc': desc,
                'space': space,
                'extension': ['.nii', '.nii.gz', '.dtseries.nii', '.func.gii'],
            },
        ),
        name='getter',
    )

    levels = list(OrderedDict.fromkeys([node.level for node in graph.nodes.values()]))
    if smoothing:
        smoothing_params = smoothing.split(':', 2)
        # Convert old style and warn; this should turn into an (informative) error around 0.5.0
        if smoothing_params[0] == 'iso':
            smoothing_params = (smoothing_params[1], 'l1', smoothing_params[0])
            warnings.warn(
                "The format for smoothing arguments has changed. Please use "
                f"{':'.join(smoothing_params)} instead of {smoothing}.",
                FutureWarning,
            )
        # Add defaults to simplify later logic
        if len(smoothing_params) == 1:
            smoothing_params.extend(('l1', 'iso'))
        elif len(smoothing_params) == 2:
            smoothing_params.append('iso')

        smoothing_fwhm, smoothing_level, smoothing_type = smoothing_params
        smoothing_fwhm = float(smoothing_fwhm)
        if smoothing_type not in (['iso', 'isoblurto']):
            raise ValueError(f"Unknown smoothing type {smoothing_type}")

        # Check that smoothing level exists in model
        if smoothing_level.lower().startswith("l"):
            if int(smoothing_level[1:]) > len(levels):
                raise ValueError(f"Invalid smoothing level {smoothing_level}")
            else:
                smoothing_level = levels[int(smoothing_level[1:]) - 1]

    design_matrix = pe.MapNode(
        DesignMatrix(drop_missing=drop_missing),
        iterfield=['design_info', 'bold_file'],
        name='design_matrix',
    )

    design_matrix.inputs.drift_model = drift_model

    def _deindex(tsv):
        from pathlib import Path
        import pandas as pd

        out_tsv = str(Path.cwd() / Path(tsv).name)
        pd.read_csv(tsv, sep='\t', index_col=0).to_csv(out_tsv, sep='\t', index=False)
        return out_tsv

    deindex_tsv = pe.MapNode(
        niu.Function(function=_deindex), iterfield=['tsv'], name='deindex_tsv'
    )

    # Set up common patterns
    image_pattern = (
        "reports/[sub-{subject}/][ses-{session}/]figures/[run-{run}/]"
        "[level-{level}_][name-{name}_][sub-{subject}_][ses-{session}_][task-{task}_]"
        "[acq-{acquisition}_][rec-{reconstruction}_][run-{run}_][echo-{echo}_]"
        "{suffix<design|corr|contrasts>}{extension<.svg>|.svg}"
    )

    contrast_plot_pattern = (
        "reports/[sub-{subject}/][ses-{session}/]figures/[run-{run}/]"
        "[level-{level}_][name-{name}_][sub-{subject}_][ses-{session}_][task-{task}_]"
        "[acq-{acquisition}_][rec-{reconstruction}_][run-{run}_][echo-{echo}_][space-{space}_]"
        "contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F|Meta>}_ortho{extension<.png>|.png}"
    )
    design_matrix_pattern = (
        "[sub-{subject}/][ses-{session}/]"
        "[level-{level}_][name-{name}_][sub-{subject}_][ses-{session}_][task-{task}_]"
        "[acq-{acquisition}_][rec-{reconstruction}_][run-{run}_][echo-{echo}_]"
        "{suffix<design>}{extension<.tsv>|.tsv}"
    )
    contrast_pattern = (
        "[sub-{subject}/][ses-{session}/]"
        "[level-{level}_][name-{name}_][sub-{subject}_][ses-{session}_][task-{task}_]"
        "[acq-{acquisition}_][rec-{reconstruction}_][run-{run}_][echo-{echo}_][space-{space}_]"
        "contrast-{contrast}_stat-{stat<effect|variance|z|p|t|F|Meta>}_"
        "statmap{extension<.nii.gz|.dscalar.nii>}"
    )
    model_map_pattern = (
        "[sub-{subject}/][ses-{session}/]"
        "[level-{level}_][name-{name}_][sub-{subject}_][ses-{session}_][task-{task}_]"
        "[acq-{acquisition}_][rec-{reconstruction}_][run-{run}_][echo-{echo}_][space-{space}_]"
        "stat-{stat<rSquare|logLikelihood|tsnr|errorts|a|b|lam|LjungBox|residtsnr|"
        "residsmoothness|residwhstd>}_statmap{extension<.nii.gz|.dscalar.nii|.tsv>}"
    )
    # Set up general interfaces
    #
    # HTML snippets to be included directly in report, not
    # saved as individual derivative files
    #

    reportlet_dir = Path(base_dir) / 'reportlets' / 'fitlins'
    reportlet_dir.mkdir(parents=True, exist_ok=True)
    snippet_pattern = (
        '[sub-{subject}/][ses-{session}/][level-{level}_][sub-{subject}_]'
        '[ses-{session}_][task-{task}_][run-{run}_]snippet.html'
    )
    ds_model_warnings = pe.MapNode(
        BIDSDataSink(base_directory=str(reportlet_dir), path_patterns=snippet_pattern),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_model_warning',
    )

    plot_design = pe.MapNode(DesignPlot(image_type='svg'), iterfield='data', name='plot_design')

    plot_corr = pe.MapNode(
        DesignCorrelationPlot(image_type='svg'),
        iterfield=['data', 'contrast_info'],
        name='plot_corr',
    )

    plot_run_contrast_matrix = pe.MapNode(
        ContrastMatrixPlot(image_type='svg'),
        iterfield=['data', 'contrast_info'],
        name='plot_run_contrast_matrix',
    )

    ds_design = pe.MapNode(
        BIDSDataSink(
            base_directory=out_dir,
            fixed_entities={"level": "run", 'suffix': 'design'},
            path_patterns=image_pattern,
        ),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_design',
    )

    ds_design_matrix = pe.MapNode(
        BIDSDataSink(
            base_directory=out_dir,
            fixed_entities={"level": "run", 'suffix': 'design'},
            path_patterns=design_matrix_pattern,
        ),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_design_matrix',
    )

    ds_corr = pe.MapNode(
        BIDSDataSink(
            base_directory=out_dir,
            fixed_entities={"level": "run", 'suffix': 'corr'},
            path_patterns=image_pattern,
        ),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_corr',
    )

    ds_run_contrasts = pe.MapNode(
        BIDSDataSink(
            base_directory=out_dir,
            fixed_entities={"level": "run", 'suffix': 'contrasts'},
            path_patterns=image_pattern,
        ),
        iterfield=['entities', 'in_file'],
        run_without_submitting=True,
        name='ds_run_contrasts',
    )

    #
    # General Connections
    #
    wf.connect(
        [
            (loader, ds_model_warnings, [('warnings', 'in_file')]),
            (loader, design_matrix, [('design_info', 'design_info')]),
            (getter, design_matrix, [('bold_files', 'bold_file')]),
            (design_matrix, plot_design, [('design_matrix', 'data')]),
            (design_matrix, plot_run_contrast_matrix, [('design_matrix', 'data')]),
            (design_matrix, plot_corr, [('design_matrix', 'data')]),
            (design_matrix, deindex_tsv, [('design_matrix', 'tsv')]),
            (deindex_tsv, ds_design_matrix, [('out', 'in_file')]),
        ]
    )

    def _select_specs(all_specs, name):
        spec = all_specs[name]
        entities = [c['entities'] for c in spec]
        contrasts = [c['contrasts'] for c in spec]
        return spec, entities, contrasts

    models = {}
    for node in graph.nodes.values():

        # Node names are unique, levels are not
        name = snake_to_camel(node.name.replace('-', '_'))
        level = node.level

        if level == "run":
            model = pe.MapNode(
                FirstLevelModel(errorts=errorts),
                iterfield=['design_matrix', 'spec', 'bold_file', 'mask_file'],
                mem_gb=3,
                name='l1_model',
            )
        else:
            model = pe.MapNode(SecondLevelModel(), iterfield=['spec'], name=f'{name}_model')
        models[node.name] = model

        select_specs = pe.Node(
            niu.Function(function=_select_specs, output_names=['spec', 'entities', 'contrasts']),
            name=f'select_{name}_specs',
            run_without_submitting=True,
        )
        select_specs.inputs.name = node.name

        # Squash the results of MapNodes that may have generated multiple maps
        # into single lists.
        # Do the same with corresponding metadata - interface will complain if shapes mismatch
        collate = pe.Node(
            MergeAll(
                [
                    'effect_maps',
                    'variance_maps',
                    'stat_maps',
                    'zscore_maps',
                    'pvalue_maps',
                    'contrast_metadata',
                ]
            ),
            name=f'collate_{name}',
            run_without_submitting=True,
        )

        #
        # Plotting
        #

        plot_contrasts = pe.MapNode(
            GlassBrainPlot(image_type='png'), iterfield='data', name=f'plot_{name}_contrasts'
        )

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
                },
            ),
            name=f'collate_{name}_outputs',
        )

        ds_contrast_maps = pe.Node(
            BIDSDataSink(base_directory=out_dir, path_patterns=contrast_pattern),
            run_without_submitting=True,
            name=f'ds_{name}_contrast_maps',
        )

        ds_contrast_plots = pe.Node(
            BIDSDataSink(base_directory=out_dir, path_patterns=contrast_plot_pattern),
            run_without_submitting=True,
            name=f'ds_{name}_contrast_plots',
        )

        if level == 'run':
            ds_model_maps = pe.Node(
                BIDSDataSink(base_directory=out_dir, path_patterns=model_map_pattern),
                run_without_submitting=True,
                name=f'ds_{name}_model_maps',
            )

            collate_mm = pe.Node(
                MergeAll(['model_maps', 'model_metadata'], check_lengths=(not drop_missing)),
                name=f'collate_mm_{name}',
                run_without_submitting=True,
            )

            wf.connect(
                [
                    (getter, model, [('bold_files', 'bold_file'), ('mask_files', 'mask_file')]),
                    (design_matrix, model, [('design_matrix', 'design_matrix')]),
                    (select_specs, getter, [('entities', 'entities')]),
                    (select_specs, ds_model_warnings, [('entities', 'entities')]),
                    (select_specs, ds_design, [('entities', 'entities')]),
                    (select_specs, ds_design_matrix, [('entities', 'entities')]),
                    (select_specs, ds_run_contrasts, [('entities', 'entities')]),
                    (select_specs, ds_corr, [('entities', 'entities')]),
                    (select_specs, plot_run_contrast_matrix, [('contrasts', 'contrast_info')]),
                    (select_specs, plot_corr, [('contrasts', 'contrast_info')]),
                    (plot_design, ds_design, [('figure', 'in_file')]),
                    (plot_run_contrast_matrix, ds_run_contrasts, [('figure', 'in_file')]),
                    (plot_corr, ds_corr, [('figure', 'in_file')]),
                    (
                        model,
                        collate_mm,
                        [('model_maps', 'model_maps'), ('model_metadata', 'model_metadata')],
                    ),
                    (
                        collate_mm,
                        ds_model_maps,
                        [('model_maps', 'in_file'), ('model_metadata', 'entities')],
                    ),
                ]
            )

        else:
            prev = node.parents[0].source.name
            wf.connect(
                [
                    (
                        models[prev],
                        model,
                        [
                            ('effect_maps', 'effect_maps'),
                            ('variance_maps', 'variance_maps'),
                            ('contrast_metadata', 'stat_metadata'),
                        ],
                    ),
                ]
            )

        if smoothing and smoothing_level == level:
            # No need to do smoothing independently if it's nistats iso
            if (smoothing_type == "iso") and (estimator == "nistats"):
                model.inputs.smoothing_fwhm = smoothing_fwhm
                model.inputs.smoothing_type = smoothing_type
            else:
                if smoothing_type == "isoblurto":
                    from nipype.interfaces.afni.preprocess import BlurToFWHM as smooth_interface
                elif smoothing_type == "iso":
                    from nipype.interfaces.afni.preprocess import BlurInMask as smooth_interface
                smooth = pe.MapNode(
                    smooth_interface(), iterfield=["in_file", "mask"], name="smooth"
                )
                smooth.inputs.fwhm = smoothing_fwhm
                smooth.inputs.outputtype = 'NIFTI_GZ'
                wf.disconnect([(getter, model, [('bold_files', 'bold_file')])])
                wf.connect(
                    [
                        (getter, smooth, [('mask_files', 'mask')]),
                        (getter, smooth, [('bold_files', 'in_file')]),
                        (smooth, model, [('out_file', 'bold_file')]),
                    ]
                )

        wf.connect(
            [
                (loader, select_specs, [('all_specs', 'all_specs')]),
                (select_specs, model, [('spec', 'spec')]),
                (
                    model,
                    collate,
                    [
                        ('effect_maps', 'effect_maps'),
                        ('variance_maps', 'variance_maps'),
                        ('stat_maps', 'stat_maps'),
                        ('zscore_maps', 'zscore_maps'),
                        ('pvalue_maps', 'pvalue_maps'),
                        ('contrast_metadata', 'contrast_metadata'),
                    ],
                ),
                (
                    collate,
                    collate_outputs,
                    [
                        ('contrast_metadata', 'metadata'),
                        ('effect_maps', 'effect_maps'),
                        ('variance_maps', 'variance_maps'),
                        ('stat_maps', 'stat_maps'),
                        ('zscore_maps', 'zscore_maps'),
                        ('pvalue_maps', 'pvalue_maps'),
                    ],
                ),
                (collate, plot_contrasts, [('stat_maps', 'data')]),
                (
                    collate_outputs,
                    ds_contrast_maps,
                    [('out', 'in_file'), ('metadata', 'entities')],
                ),
                (collate, ds_contrast_plots, [('contrast_metadata', 'entities')]),
                (plot_contrasts, ds_contrast_plots, [('figure', 'in_file')]),
            ]
        )

    return wf
