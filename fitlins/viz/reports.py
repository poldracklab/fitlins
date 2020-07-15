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


def displayify(contrast_name):
    for match, repl in (('_gt_', ' &gt; '),
                        ('_lt_', ' &lt; '),
                        ('_vs_', ' vs. ')):
        contrast_name = contrast_name.replace(match, repl)
    return contrast_name


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


def build_report_dict(deriv_dir, work_dir, analysis):
    fl_layout = BIDSLayout(
        deriv_dir,
        config=['bids', 'derivatives', 'fitlins'],
        validate=False)
    wd_layout = BIDSLayout(
        Path(work_dir) / 'reportlets' / 'fitlins',
        validate=False)
    all_pngs = fl_layout.get(extension='.png')
    fig_dirs = set(
        (png.dirname, tuple(ent for ent in png.entities.items()
                            if ent[0] not in ('suffix', 'contrast')))
        for png in fl_layout.get(extension='.png'))

    report = {
        'dataset': {
            'name': analysis.layout.description['Name'],
            },
        'model': analysis.model,
        'steps': []
        }

    if 'DatasetDOI' in analysis.layout.description:
        report['dataset']['doi'] = analysis.layout.description['DatasetDOI']

    for step in analysis.steps:
        report_step = {'name': step.level, 'analyses': []}
        report['steps'].append(report_step)
        for coll in step.get_collections():
            ents = coll.entities.copy()
            contrasts = step.get_contrasts(coll)
            for key in ('datatype', 'desc', 'suffix', 'extension'):
                ents.pop(key, None)
            for key in analysis.layout.get_entities(metadata=True):
                ents.pop(key, None)

            analysis_dict = {
                'entities': {
                    key: val
                    for key, val in ents.items()
                    if key in ('subject', 'session', 'task', 'run') and val},
                'contrasts': []
                }

            for contrast in contrasts:
                glassbrain = fl_layout.get(
                    contrast=snake_to_camel(contrast.name),
                    suffix='ortho', extension='png', **ents)
                analysis_dict['contrasts'].append(
                    {'name': displayify(contrast.name),
                     'glassbrain': glassbrain[0].path if glassbrain else None}
                )
            report_step['analyses'].append(analysis_dict)

            # Space doesn't apply to design/contrast matrices
            ents.pop('space', None)
            design_matrix = fl_layout.get(suffix='design', extension='svg', **ents)
            correlation_matrix = fl_layout.get(suffix='corr', extension='svg', **ents)
            contrast_matrix = fl_layout.get(suffix='contrasts', extension='svg', **ents)
            warning = wd_layout.get(extension='.html', suffix='snippet', **ents)
            if design_matrix:
                analysis_dict['design_matrix'] = design_matrix[0].path
            if correlation_matrix:
                analysis_dict['correlation_matrix'] = correlation_matrix[0].path
            if contrast_matrix:
                analysis_dict['contrast_matrix'] = contrast_matrix[0].path
            if warning:
                analysis_dict['warning'] = Path(warning[0].path).read_text()

    # Get subjects hackily
    report['subjects'] = sorted({
        analysis_dict['entities']['subject']
        for analysis_dict in report['steps'][0]['analyses']})

    return report


def write_full_report(report_dict, run_context, deriv_dir):
    fl_layout = BIDSLayout(
        deriv_dir, config=['bids', 'derivatives', 'fitlins'])

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            searchpath=pkgr.resource_filename('fitlins', '/')))

    tpl = env.get_template('data/full_report.tpl')

    model = snake_to_camel(report_dict['model']['name'])
    target_file = op.join(
        deriv_dir, fl_layout.build_path(
            {'model': model}, PATH_PATTERNS, validate=False))
    html = tpl.render(deroot({**report_dict, **run_context}, op.dirname(target_file)))
    Path(target_file).write_text(html)
