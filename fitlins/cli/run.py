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
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

from .. import info
from ..utils import bids
from ..base import init, first_level, second_level
from ..viz.reports import write_report

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
    verstr = 'fitlins v{}'.format(info.__version__)

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
    g_bids.add_argument('--participant-label', action='store', nargs='+',
                        help='one or more participant identifiers (the sub- prefix can be '
                             'removed)')
    g_bids.add_argument('-m', '--model', action='store', default='model.json',
                        help='location of BIDS model description (default bids_dir/model.json)')
    g_bids.add_argument('-p', '--preproc-dir', action='store', default='fmriprep',
                        help='location of preprocessed data (default output_dir/fmriprep)')
    g_bids.add_argument('--space', action='store',
                        choices=['MNI152NLin2009cAsym'], default='MNI152NLin2009cAsym',
                        help='registered space of input datasets')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument('--debug', action='store_true', default=False,
                         help='run debug version of workflow')

    return parser


def main(args=None):
    """Entry point"""
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args(args)
    if opts.debug:
        logger.setLevel(logging.DEBUG)

    create_workflow(opts)


def create_workflow(opts):
    """Build workflow"""

    # First check that bids_dir looks like a BIDS folder
    bids_dir = op.abspath(opts.bids_dir)
    subject_list = bids.collect_participants(
        bids_dir, participant_label=opts.participant_label)

    output_dir = op.abspath(opts.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build main workflow
    logger.log(25, INIT_MSG(
        version=info.__version__,
        subject_list=subject_list)
    )

    model = default_path(opts.model, bids_dir, 'model.json')
    preproc_dir = default_path(opts.preproc_dir, output_dir, 'fmriprep')
    deriv_dir = op.join(output_dir, 'fitlins')

    # BIDS-Apps prefers 'participant', BIDS-Model prefers 'subject'
    level = 'subject' if opts.analysis_level == 'participant' else opts.analysis_level

    try:
        retcode = run_model(model, opts.space, level, bids_dir, preproc_dir,
                            deriv_dir)
    except Exception:
        retcode = 1

    sys.exit(retcode)


def run_model(model, space, target_level, bids_dir, preproc_dir, deriv_dir):
    run_context = {'version': info.__version__,
                   'command': ' '.join(sys.argv),
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                   }

    analysis = init(model, bids_dir, preproc_dir)
    report_dicts = first_level(analysis, analysis.blocks[0], space, deriv_dir)
    write_report(analysis.blocks[0].level, report_dicts, run_context,
                 deriv_dir)
    if analysis.blocks[0].level == target_level:
        return 0
    for block in analysis.blocks[1:]:
        report_dicts = second_level(analysis, block, space, deriv_dir)
        write_report(block.level, report_dicts, run_context, deriv_dir)
        if block.level == target_level:
            break

    return 0


if __name__ == '__main__':
    raise RuntimeError("fitlins/cli/run.py should not be run directly;\n"
                       "Please `pip install` fitlins and use the `fitlins` command")
