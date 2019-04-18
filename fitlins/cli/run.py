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
from tempfile import mkdtemp
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count

from bids.layout import BIDSLayout
from bids.analysis import auto_model, Analysis

from .. import __version__
from ..workflows import init_fitlins_wf
from ..utils import bids
from ..viz.reports import write_report, parse_directory

logging.addLevelName(25, 'INFO')  # Add a new level between INFO and WARNING
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
    parser.add_argument('-v', '--version', action='version', version=verstr)

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
                        choices=['MNI152NLin2009cAsym', ''],
                        default='MNI152NLin2009cAsym',
                        help='registered space of input datasets. Empty value for no explicit space.')
    g_bids.add_argument('--force-index', action='store', default=None,
                        help='regex pattern or string to include files')
    g_bids.add_argument('--ignore', action='store', default=None,
                        help='regex pattern or string to ignore files')
    g_bids.add_argument('--desc-label', action='store', default='preproc',
                        help="use BOLD files with the provided description label")

    g_prep = parser.add_argument_group('Options for preprocessing BOLD series')
    g_prep.add_argument('-s', '--smoothing', action='store', metavar="TYPE:FWHM",
                        help="Smooth BOLD series with FWHM mm kernel prior to fitting. "
                             "Valid types: iso (isotropic); "
                             "e.g. `--smothing iso:5` will use an isotropic 5mm FWHM kernel")

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--n-cpus', action='store', default=0, type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=op.abspath,
                         help='path where intermediate results should be stored')

    return parser


def run_fitlins(argv=None):
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args(argv)
    if opts.debug:
        logger.setLevel(logging.DEBUG)
    if not opts.space:
        # make it an explicit None
        opts.space = None

    subject_list = None
    if opts.participant_label is not None:
        subject_list = bids.collect_participants(
            opts.bids_dir, participant_label=opts.participant_label)

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

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=__version__,
        subject_list=subject_list)
    )

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

    pipeline_name = 'fitlins'
    if opts.derivative_label:
        pipeline_name += '_' + opts.derivative_label
    deriv_dir = op.join(opts.output_dir, pipeline_name)
    os.makedirs(deriv_dir, exist_ok=True)

    bids.write_derivative_description(opts.bids_dir, deriv_dir)

    work_dir = mkdtemp() if opts.work_dir is None else opts.work_dir

    fitlins_wf = init_fitlins_wf(
        opts.bids_dir, derivatives, deriv_dir,
        analysis_level=opts.analysis_level, model=model,
        space=opts.space, desc=opts.desc_label,
        participants=subject_list, base_dir=work_dir,
        force_index=opts.force_index, ignore=opts.ignore,
        smoothing=opts.smoothing,
        )

    if opts.work_dir:
        # dump crashes in working directory (non /tmp)
        fitlins_wf.config['execution']['crashdump_dir'] = opts.work_dir
    # easy to read crashfiles
    fitlins_wf.config['execution']['crashfile_format'] = 'txt'
    retcode = 0
    try:
        fitlins_wf.run(**plugin_settings)
    except Exception:
        retcode = 1

    layout = BIDSLayout(opts.bids_dir)
    models = auto_model(layout) if model == 'default' else [model]

    run_context = {'version': __version__,
                   'command': ' '.join(sys.argv),
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                   }

    for model in models:
        analysis = Analysis(layout, model=model)
        report_dicts = parse_directory(deriv_dir, work_dir, analysis)
        write_report(report_dicts, run_context, deriv_dir)

    return retcode


def main():
    sys.exit(run_fitlins(sys.argv[1:]))


if __name__ == '__main__':
    raise RuntimeError("fitlins/cli/run.py should not be run directly;\n"
                       "Please `pip install` fitlins and use the `fitlins` command")
