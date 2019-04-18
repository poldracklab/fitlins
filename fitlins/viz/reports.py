from os import path as op
from pathlib import Path
import jinja2
import pkg_resources as pkgr
from bids.layout import add_config_paths, BIDSLayout

from ..utils import snake_to_camel

PATH_PATTERNS = [
    'reports/[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    '[run-{run}_]model-{model}.html'
]

add_config_paths(fitlins=pkgr.resource_filename('fitlins', 'data/fitlins.json'))


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
    fl_layout = BIDSLayout(
        deriv_dir,
        config=['bids', 'derivatives', 'fitlins'],
        validate=False)
    wd_layout = BIDSLayout(
        str(Path(work_dir) / 'reportlets' / 'fitlins'),
        validate=False)
    all_pngs = fl_layout.get(extensions='.png')
    fig_dirs = set(
        (png.dirname, tuple(ent for ent in png.entities.items()
                            if ent[0] not in ('suffix', 'contrast')))
        for png in fl_layout.get(extensions='.png'))

    analyses = []
    for figdir, ent_tuple in fig_dirs:
        ents = dict(ent_tuple)
        ents.setdefault('subject', None)
        ents.setdefault('session', None)
        ents.setdefault('run', None)
        contrast_matrix = fl_layout.get(extensions='.svg', suffix='contrasts', **ents)
        correlation_matrix = fl_layout.get(extensions='.svg', suffix='corr',
                                           **ents)
        design_matrix = fl_layout.get(extensions='.svg', suffix='design', **ents)
        job_desc = {
            'ents': {k: v for k, v in ents.items() if v is not None},
            'dataset': analysis.layout.root,
            'model_name': analysis.model['name'],
            }
        if ents.get('subject'):
            job_desc['subject_id'] = ents.get('subject')
        if contrast_matrix:
            job_desc['contrasts_svg']: contrast_matrix[0].path
        if correlation_matrix:
            job_desc['correlation_matrix_svg'] = correlation_matrix[0].path
        if design_matrix:
            job_desc['design_matrix_svg'] = design_matrix[0].path
        if ents['run'] is not None:
            job_desc['level'] = 'run'
        elif ents['session'] is not None:
            job_desc['level'] = 'session'
        elif ents['subject'] is not None:
            job_desc['level'] = 'subject'
        else:
            job_desc['level'] = 'dataset'

        snippet = wd_layout.get(extensions='.html', suffix='snippet', **ents)
        if snippet:
            with open(snippet[0].path) as fobj:
                job_desc['warning'] = fobj.read()

        contrasts = fl_layout.get(extensions='.png', suffix='ortho', **ents)
        # TODO: Split contrasts from estimates
        job_desc['contrasts'] = [{'image_file': c.path,
                                  'name':
                                      fl_layout.parse_file_entities(
                                          c.path)['contrast']}
                                 for c in contrasts]
        analyses.append(job_desc)

    return analyses


def write_report(report_dicts, run_context, deriv_dir):
    fl_layout = BIDSLayout(
        deriv_dir, config=['bids', 'derivatives', 'fitlins'])

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=pkgr.resource_filename('fitlins', '/')))

    tpl = env.get_template('data/report.tpl')

    for context in report_dicts:
        ents = context['ents'].copy()
        ents['model'] = snake_to_camel(context['model_name'])
        target_file = op.join(deriv_dir, fl_layout.build_path(ents, PATH_PATTERNS))
        html = tpl.render(deroot({**context, **run_context},
                                 op.dirname(target_file)))
        with open(target_file, 'w') as fobj:
            fobj.write(html)
