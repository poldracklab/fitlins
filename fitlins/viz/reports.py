from os import path as op
from pathlib import Path
import jinja2
import pkg_resources as pkgr
import bids

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


def parse_directory(deriv_dir, work_dir, analysis):
    fl_layout = bids.BIDSLayout(
        (deriv_dir, ['bids', 'derivatives',
                     pkgr.resource_filename('fitlins', 'data/fitlins.json')]))
    wd_layout = bids.BIDSLayout(str(Path(work_dir) / 'reportlets' / 'fitlins'))
    contrast_svgs = fl_layout.get(extensions='.svg', type='contrasts')

    analyses = []
    for contrast_svg in contrast_svgs:
        ents = fl_layout.parse_file_entities(contrast_svg.filename)
        ents.pop('type')
        ents.setdefault('subject', None)
        correlation_matrix = fl_layout.get(extensions='.svg', type='corr',
                                           **ents)
        design_matrix = fl_layout.get(extensions='.svg', type='design', **ents)
        job_desc = {
            'ents': {k: v for k, v in ents.items() if v is not None},
            'dataset': analysis.layout.root,
            'model_name': analysis.model['name'],
            'contrasts_svg': contrast_svg.filename,
            }
        if ents.get('subject'):
            job_desc['subject_id'] = ents.get('subject')
        if correlation_matrix:
            job_desc['correlation_matrix_svg'] = correlation_matrix[0].filename
        if design_matrix:
            job_desc['design_matrix_svg'] = design_matrix[0].filename

        snippet = wd_layout.get(extensions='.html', type='snippet', **ents)
        if snippet:
            with open(snippet[0].filename) as fobj:
                job_desc['warning'] = fobj.read()

        contrasts = fl_layout.get(extensions='.png', type='ortho', **ents)
        # TODO: Split contrasts from estimates
        job_desc['contrasts'] = [{'image_file': c.filename,
                                  'name':
                                      fl_layout.parse_file_entities(
                                          c.filename)['contrast']}
                                 for c in contrasts]
        analyses.append(job_desc)

    return analyses


def write_report(level, report_dicts, run_context, deriv_dir):
    fl_layout = bids.BIDSLayout(
        (deriv_dir, ['bids', 'derivatives',
                     pkgr.resource_filename('fitlins', 'data/fitlins.json')]))
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
