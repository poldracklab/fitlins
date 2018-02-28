from os import path as op
import jinja2
import pkg_resources as pkgr
from bids import grabbids

from ..utils import snake_to_camel

PATH_PATTERNS = [
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'model-{model}.html'
]


def first_level_reports(level, report_dicts, run_context, deriv_dir):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        extensions=['derivatives',
                    pkgr.resource_filename('fitlins', 'data/fitlins.json')])
    fl_layout.path_patterns = PATH_PATTERNS

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=pkgr.resource_filename('fitlins', '/')))

    tpl = env.get_template('data/first_level_report.tpl')

    for context in report_dicts:
        html = tpl.render({'level': level, **context, **run_context})
        ents = context['ents'].copy()
        ents['model'] = snake_to_camel(context['model_name'])
        target_file = op.join(deriv_dir, fl_layout.build_path(ents))
        with open(target_file, 'w') as fobj:
            fobj.write(html)


def second_level_reports(level, report_dicts, run_context, deriv_dir):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        extensions=['derivatives',
                    pkgr.resource_filename('fitlins', 'data/fitlins.json')])
    fl_layout.path_patterns = PATH_PATTERNS

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=pkgr.resource_filename('fitlins', '/')))

    tpl = env.get_template('data/second_level_report.tpl')

    for context in report_dicts:
        html = tpl.render({'level': level, **context, **run_context})
        ents = context['ents'].copy()
        ents['model'] = snake_to_camel(context['model_name'])
        target_file = op.join(deriv_dir, fl_layout.build_path(ents))
        with open(target_file, 'w') as fobj:
            fobj.write(html)
