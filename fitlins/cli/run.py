#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI model-fitting
==================
"""

import os
import os.path as op
import logging
import sys
import uuid
import warnings
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
from time import strftime

logging.addLevelName(25, 'INFO')  # Add a new level between INFO and WARNING
logger = logging.getLogger('cli')
logger.setLevel(25)

INIT_MSG = """
Running FitLins version {version}:
  * Participant list: {subject_list}.
  * Run identifier: {uuid}.
""".format


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def get_parser():
    """Build parser object"""
    from fitlins.info import __version__

    verstr = 'fitlins v{}'.format(__version__)

    parser = ArgumentParser(description='FitLins: Workflows for Fitting Linear models to fMRI',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store',
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store',
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['run', 'session', 'participant', 'dataset'],
                        help='processing stage to be runa (see BIDS-Apps specification).')

    # optional arguments
    parser.add_argument('-v', '--version', action='version', version=verstr)

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('-s', '--session-id', action='store', default='single_session',
                        help='select a specific session to be processed')
    g_bids.add_argument('-t', '--task-id', action='store',
                        help='select a specific task to be processed')
    g_bids.add_argument('-m', '--model', action='store', default='model.json',
                        help='location of BIDS model description (default bids_dir/model.json)')
    g_bids.add_argument('-p', '--preproc-dir', action='store', default='fmriprep',
                        help='location of preprocessed data (default output_dir/fmriprep)')
    g_bids.add_argument('--space', action='store',
                        choices=['MNI152NLin2009cAsym', 'T1w'], default='MNI152NLin2009cAsym',
                        help='registered space of input datasets')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_perfm.add_argument('--nthreads', '--n_cpus', '-n-cpus', action='store', default=0, type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--omp-nthreads', action='store', type=int, default=0,
                         help='maximum number of threads per-process')
    g_perfm.add_argument('--mem_mb', '--mem-mb', action='store', default=0, type=int,
                         help='upper bound memory limit for FMRIPREP processes')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', default='work',
                         help='path where intermediate results should be stored')
    g_other.add_argument(
        '--reports-only', action='store_true', default=False,
        help='only generate reports, don\'t run workflows. This will only rerun report '
             'aggregation, not reportlet generation for specific nodes.')
    g_other.add_argument(
        '--run-uuid', action='store', default=None,
        help='Specify UUID of previous run, to include error logs in report. '
             'No effect without --reports-only.')
    g_other.add_argument('--write-graph', action='store_true', default=False,
                         help='Write workflow graph.')

    return parser


def main():
    """Entry point"""
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()
    if opts.debug:
        logger.setLevel(logging.DEBUG)

    create_workflow(opts)


def create_workflow(opts):
    """Build workflow"""
    from niworkflows.nipype import config as ncfg
    from niworkflows.nipype.pipeline import engine as pe
    from fitlins.info import __version__
    from fitlins.utils.bids import collect_participants
    from fitlins.base import init, first_level, second_level

    # Set up some instrumental utilities
    errno = 0
    run_uuid = strftime('%Y%m%d-%H%M%S_') + str(uuid.uuid4())

    # First check that bids_dir looks like a BIDS folder
    bids_dir = op.abspath(opts.bids_dir)
    subject_list = collect_participants(
        bids_dir, participant_label=opts.participant_label)

    # Nipype plugin configuration
    plugin_settings = {'plugin': 'Linear'}
    nthreads = opts.nthreads
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
    else:
        # Setup multiprocessing
        nthreads = opts.nthreads
        if nthreads == 0:
            nthreads = cpu_count()

        if nthreads > 1:
            plugin_settings['plugin'] = 'MultiProc'
            plugin_settings['plugin_args'] = {'n_procs': nthreads}
            if opts.mem_mb:
                plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        raise RuntimeError(
            'Per-process threads (--omp-nthreads={:d}) cannot exceed total '
            'threads (--nthreads/--n_cpus={:d})'.format(omp_nthreads, nthreads))

    # Set up directories
    output_dir = op.abspath(opts.output_dir)
    log_dir = op.join(output_dir, 'fitlins', 'logs')
    work_dir = op.abspath(opts.work_dir)

    # Check and create output and working directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {'log_directory': log_dir, 'log_to_file': True},
        'execution': {'crashdump_dir': log_dir, 'crashfile_format': 'txt'},
    })

    # Called with reports only
    if opts.reports_only:
        logger.log(25, 'Running --reports-only on participants %s', ', '.join(subject_list))
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
        # report_errors = [
        #     run_reports(op.join(work_dir, 'reportlets'), output_dir, subject_label,
        #                 run_uuid=run_uuid)
        #     for subject_label in subject_list]
        # sys.exit(int(sum(report_errors) > 0))
        sys.exit(0)

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=__version__,
        subject_list=subject_list,
        uuid=run_uuid)
    )

    model = opts.model
    if model is None:
        model = 'model.json'
    if not op.isabs(model):
        model = op.join(bids_dir, model)

    preproc_dir = opts.preproc_dir
    if preproc_dir is None:
        preproc_dir = 'fmriprep'
    if not op.isabs(preproc_dir):
        preproc_dir = op.join(output_dir, preproc_dir)

    deriv_dir = op.join(output_dir, 'fitlins')

    level = 'subject' if opts.analysis_level == 'participant' else opts.analysis_level

    analysis = init(model, bids_dir, preproc_dir)
    imgs = first_level(analysis, analysis.blocks[0], deriv_dir)
    if analysis.blocks[0].level == opts.analysis_level:
        sys.exit(0)
    for block in analysis.blocks[1:]:
        imgs = second_level(analysis, block, imgs, deriv_dir)
        if block.level == opts.analysis_level:
            break

    sys.exit(0)

    fitlins_wf = pe.Workflow(name='fitlins_wf')

    if opts.write_graph:
        fitlins_wf.write_graph(graph2use="colored", format='svg', simple_form=True)

    try:
        fitlins_wf.run(**plugin_settings)
    except RuntimeError as e:
        if "Workflow did not execute cleanly" in str(e):
            errno = 1
        else:
            raise(e)

    # Generate reports phase
    # report_errors = [run_reports(
    #     op.join(work_dir, 'reportlets'), output_dir, subject_label, run_uuid=run_uuid)
    #     for subject_label in subject_list]

    # if sum(report_errors):
    #     logger.warning('Errors occurred while generating reports for participants: %s.',
    #                    ', '.join(['%s (%d)' % (subid, err)
    #                               for subid, err in zip(subject_list, report_errors)]))

    # errno += sum(report_errors)
    sys.exit(int(errno > 0))


if __name__ == '__main__':
    # raise RuntimeError("fitlins/cli/run.py should not be run directly;\n"
    #                    "Please `pip install` fitlins and use the `fitlins` command")
    main()
