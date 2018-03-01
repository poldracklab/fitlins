from os import path as op
import jinja2
import pkg_resources as pkgr
from bids import grabbids

from ..utils import snake_to_camel

PATH_PATTERNS = [
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'model-{model}.html'
]


def deroot(val, root):
    if isinstance(val, str):
        if val.startswith(root):
            idx = len(root)
            if val[idx] == '/':
                idx += 1
            val = val[idx:]
    elif isinstance(val, list):
        val = [deroot(elem, root) for elem in val]
    elif isinstance(val, dict):
        val = {key: deroot(value, root) for key, value in val.items()}

    return val


def write_report(level, report_dicts, run_context, deriv_dir):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        extensions=['derivatives',
                    pkgr.resource_filename('fitlins', 'data/fitlins.json')])
    fl_layout.path_patterns = PATH_PATTERNS

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=pkgr.resource_filename('fitlins', '/')))

    tpl = env.get_template('data/report.tpl')

    for context in report_dicts:
        ents = context['ents'].copy()
        ents['model'] = snake_to_camel(context['model_name'])
        target_file = op.join(deriv_dir, fl_layout.build_path(ents))
        html = tpl.render(deroot({'level': level, **context, **run_context},
                                 op.dirname(target_file)))
        with open(target_file, 'w') as fobj:
            fobj.write(html)
