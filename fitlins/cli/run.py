#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fMRI model-fitting
==================
"""

import sys
import os
import os.path as op
import time
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count

import bids
from bids.analysis import auto_model, Analysis

from .. import __version__
from ..workflows import init_fitlins_wf
from ..utils import bids as fub, config
from ..viz.reports import build_report_dict, write_full_report

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logger = logging.getLogger('cli')
logger.setLevel(25)

INIT_MSG = """
Running FitLins version {version}:
  * Participant list: {subject_list}.
""".format


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def default_path(arg, base_dir, default_name):
    """Generate absolute path from absolute, relative, or missing path

    Missing paths are given a default value, which may be absolute or relative,
    and relative paths are resolved relative to a base directory.
    """
    path = default_name if arg is None else arg
    if not op.isabs(path):
        path = op.abspath(op.join(base_dir, path))
    return path


def get_parser():
    """Build parser object"""
    verstr = 'fitlins v{}'.format(__version__)

    parser = ArgumentParser(description='FitLins: Workflows for Fitting Linear models to fMRI',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument('bids_dir', action='store', type=op.abspath,
                        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
                             'be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=op.abspath,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['run', 'session', 'participant', 'dataset'],
                        help='processing stage to be runa (see BIDS-Apps specification).')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase log verbosity for each occurence, debug level is -vvv")
    parser.add_argument('-q', '--quiet', action='count', default=0,
                        help="decrease log verbosity for each occurence, debug level is -vvv")

    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant-label', action='store', nargs='+', default=None,
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('-m', '--model', action='store',
                        help='location of BIDS model description')
    g_bids.add_argument('-d', '--derivatives', action='store', nargs='+',
                        help='location of derivatives (including preprocessed images).'
                        'If none specified, indexes all derivatives under bids_dir/derivatives.')
    g_bids.add_argument('--derivative-label', action='store', type=str,
                        help='execution label to append to derivative directory name')
    g_bids.add_argument('--space', action='store',
                        default='MNI152NLin2009cAsym',
                        help='registered space of input datasets. '
                             'Empty value for no explicit space.')
    g_bids.add_argument('--force-index', action='store', default=None, nargs='+',
                        help='regex pattern or string to include files')
    g_bids.add_argument('--ignore', action='store', default=None, nargs='+',
                        help='regex pattern or string to ignore files')
    g_bids.add_argument('--desc-label', action='store', default='preproc',
                        help="use BOLD files with the provided description label")
    g_bids.add_argument('--database-path', action='store', default=None,
                        help="Path to directory containing SQLite database indicies "
                             "for this BIDS dataset. "
                             "If a value is passed and the file already exists, "
                             "indexing is skipped.")

    g_prep = parser.add_argument_group('Options for preprocessing BOLD series')
    g_prep.add_argument('-s', '--smoothing', action='store', metavar="FWHM[:LEVEL:[TYPE]]",
                        help="Smooth BOLD series with FWHM mm kernel prior to fitting at LEVEL. "
                             "Optional analysis LEVEL (default: l1) may be specified numerically "
                             "(e.g., `l1`) or by name (`run`, `subject`, `session` or `dataset`). "
                             "Optional smoothing TYPE (default: iso) must be one of:  "
                             " `iso` (isotropic additive smoothing), `isoblurto` (isotropic "
                             "smoothing progressivley applied till "
                             "the target smoothness is reached). "
                             "e.g., `--smoothing 5:dataset:iso` will perform "
                             "a 5mm FWHM isotropic smoothing on subject-level maps, "
                             "before evaluating the dataset level.")

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--n-cpus', action='store', default=0, type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--mem-gb', action='store', default=0, type=float,
                         help='maximum amount of memory to allocate across all processes')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')
    g_perfm.add_argument('--reports-only', action='store_true', default=False,
                         help='skip running of workflow and generate reports')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=op.abspath,
                         help='path where intermediate results should be stored')
    g_other.add_argument('--drop-missing', action='store_true', default=False,
                         help='drop missing inputs/contrasts in model fitting.')

    g_other.add_argument("--estimator", action="store", type=str,
                         help="estimator to use to fit the model",
                         default="nistats", choices=["nistats", "afni"])
    g_other.add_argument("--drift-model", action="store", type=str,
                         help="specifies the desired drift model",
                         default=None, choices=["polynomial", "cosine", None])
    g_other.add_argument("--error-ts", action='store_true', default=False,
                         help='save error time series for first level models.'
                         ' Currently only implemented for afni estimator.')

    return parser


def run_fitlins(argv=None):
    import re
    from nipype import logging as nlogging
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args(argv)

    force_index = [
        # If entry looks like `/<pattern>/`, treat `<pattern>` as a regex
        re.compile(ign[1:-1]) if (ign[0], ign[-1]) == ('/', '/') else ign
        # Iterate over empty tuple if undefined
        for ign in opts.force_index or ()]
    ignore = [
        # If entry looks like `/<pattern>/`, treat `<pattern>` as a regex
        re.compile(ign[1:-1]) if (ign[0], ign[-1]) == ('/', '/') else ign
        # Iterate over empty tuple if undefined
        for ign in opts.ignore or ()]

    log_level = 25 + 5 * (opts.quiet - opts.verbose)
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    if not opts.space:
        # make it an explicit None
        opts.space = None
    if not opts.desc_label:
        # make it an explicit None
        opts.desc_label = None

    ncpus = opts.n_cpus
    if ncpus < 1:
        ncpus = cpu_count()

    plugin_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': ncpus,
            'raise_insufficient': False,
            'maxtasksperchild': 1,
        }
    }

    if opts.mem_gb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_gb

    model = default_path(opts.model, opts.bids_dir, 'model-default_smdl.json')
    if opts.model in (None, 'default') and not op.exists(model):
        model = 'default'

    derivatives = True if not opts.derivatives else opts.derivatives
    # Need this when specifying args directly (i.e. neuroscout)
    # god bless neuroscout, but let's make it work for others!
    if isinstance(derivatives, list) and len(derivatives) == 1:
        # WRONG AND EVIL to those who have spaces in their paths... bad bad practice
        # TODO - fix neuroscout
        derivatives = derivatives[0].split(" ")

    if opts.estimator != 'afni':
        if opts.error_ts:
            raise NotImplementedError("Saving the error time series is only implmented for"
                                      " the afni estimator. If this is a feature you want"
                                      f" for {opts.estimator} please let us know on github.")

    pipeline_name = 'fitlins'
    if opts.derivative_label:
        pipeline_name += '_' + opts.derivative_label
    deriv_dir = op.join(opts.output_dir, pipeline_name)
    os.makedirs(deriv_dir, exist_ok=True)
    fub.write_derivative_description(
        opts.bids_dir, deriv_dir, vars(opts)
    )

    work_dir = mkdtemp() if opts.work_dir is None else opts.work_dir

    # Go ahead and initialize the layout database
    if opts.database_path is None:
        database_path = Path(work_dir) / 'dbcache'
        reset_database = True
    else:
        database_path = opts.database_path
        reset_database = False

    indexer = bids.BIDSLayoutIndexer(ignore=ignore, force_index=force_index)
    layout = bids.BIDSLayout(opts.bids_dir,
                             derivatives=derivatives,
                             database_path=database_path,
                             reset_database=reset_database,
                             indexer=indexer)

    subject_list = None
    if opts.participant_label is not None:
        subject_list = fub.collect_participants(
            layout, participant_label=opts.participant_label)

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=__version__,
        subject_list=subject_list)
    )

    fitlins_wf = init_fitlins_wf(
        database_path, deriv_dir,
        analysis_level=opts.analysis_level, model=model,
        space=opts.space, desc=opts.desc_label,
        participants=subject_list, base_dir=work_dir,
        smoothing=opts.smoothing, drop_missing=opts.drop_missing,
        drift_model=opts.drift_model,
        estimator=opts.estimator, errorts=opts.error_ts
        )
    fitlins_wf.config = deepcopy(config.get_fitlins_config()._sections)

    if opts.work_dir:
        # dump crashes in working directory (non /tmp)
        fitlins_wf.config['execution']['crashdump_dir'] = opts.work_dir
    retcode = 0
    if not opts.reports_only:
        try:
            fitlins_wf.run(**plugin_settings)
        except Exception:
            retcode = 1

    models = auto_model(layout) if model == 'default' else [model]

    run_context = {'version': __version__,
                   'command': ' '.join(sys.argv),
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                   }

    selectors = {'desc': opts.desc_label, 'space': opts.space}
    if subject_list is not None:
        selectors['subject'] = subject_list

    for model in models:
        analysis = Analysis(layout, model=model)
        analysis.setup(**selectors)
        report_dict = build_report_dict(deriv_dir, work_dir, analysis)
        write_full_report(report_dict, run_context, deriv_dir)

    return retcode


def main():
    sys.exit(run_fitlins(sys.argv[1:]))


if __name__ == '__main__':
    raise RuntimeError("fitlins/cli/run.py should not be run directly;\n"
                       "Please `pip install` fitlins and use the `fitlins` command")
