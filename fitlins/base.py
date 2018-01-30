import os
from os import path as op
from functools import reduce
import numpy as np
import pandas as pd
import nibabel as nb
import nilearn.image as nli
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

import pkg_resources as pkgr

from grabbit import merge_layouts
from bids import grabbids
from bids.analysis import base as ba

PATH_PATTERNS = (
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'task-{task}_bold[_space-{space}]_contrast-{contrast}_{type}.nii.gz',
    'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_]'
    'task-{task}_bold_design.tsv',
    )

def dict_intersection(dict1, dict2):
    return {k: v for k, v in dict1.items() if dict2.get(k) == v}


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def init(model_fname, bids_dir, preproc_dir):
    orig_layout = grabbids.BIDSLayout(bids_dir)
    prep_layout = grabbids.BIDSLayout(preproc_dir, extensions=['derivatives'])
    analysis = ba.Analysis(model=model_fname,
                           layout=merge_layouts([orig_layout, prep_layout]))
    analysis.setup()
    analysis.layout.path_patterns[:0] = PATH_PATTERNS
    return analysis


def first_level(analysis, block, deriv_dir):
    for paradigm, ents in block.get_design_matrix():
        preproc_files = analysis.layout.get(type='preproc', space='MNI152NLin2009cAsym',
                                            **ents, **analysis.selectors)
        if len(preproc_files) != 1:
            print(preproc_files)
            raise ValueError("Too many potential PREPROC files")

        fname = preproc_files[0].filename

        # confounds_file = analysis.layout.get(type='confounds', **ents)[0]
        # confounds = pd.read_csv(confounds_file.filename, sep="\t", na_values="n/a").fillna(0)
        # names = [col for col in confounds.columns
        #          if col.startswith('NonSteadyStateOutlier') or
        #          col in block.model['variables']]
        img = nb.load(fname)
        TR = img.header.get_zooms()[3]
        vols = img.shape[3]

        mat = dm.make_design_matrix(np.arange(vols) * TR,
                                    paradigm.rename(columns={'condition': 'trial_type'}),
                                    drift_model=None,
                                    # add_regs=confounds[names],
                                    # add_reg_names=names,
                                    )

        preproc_ents = analysis.layout.parse_entities(fname)

        dm_ents = {k: v for k, v in preproc_ents.items()
                    if k in ('subject', 'session', 'task')}

        design_fname = op.join(deriv_dir,
                               analysis.layout.build_path(dm_ents, strict=True))
        os.makedirs(op.dirname(design_fname), exist_ok=True)
        mat.to_csv(design_fname, sep='\t')

        base = op.basename(fname)

        brainmask = analysis.layout.get(type='brainmask', space='MNI152NLin2009cAsym',
                                        **ents, **analysis.selectors)[0]
        fmri_glm = None

        for contrast in block.contrasts:
            stat_ents = preproc_ents.copy()
            stat_ents.pop('modality', None)
            stat_ents.update({'contrast': snake_to_camel(contrast['name']),
                              'type': 'stat'})
            stat_fname = op.join(deriv_dir,
                                 analysis.layout.build_path(stat_ents,
                                                            strict=True))

            if op.exists(stat_fname):
                continue

            if fmri_glm is None:
                fmri_glm = level1.FirstLevelModel(mask=brainmask.filename)
                fmri_glm.fit(fname, design_matrices=mat)

            indices = [mat.columns.get_loc(cond)
                       for cond in contrast['condition_list']]

            weights = np.zeros(len(mat.columns))
            weights[indices] = contrast['weights']

            stat = fmri_glm.compute_contrast(weights, {'T': 't', 'F': 'F'}[contrast['type']])
            stat.to_filename(stat_fname)


def second_level(analysis, block, deriv_dir, mapping=None):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        extensions=['derivatives',
                    pkgr.resource_filename('fitlins', 'data/fitlins.json')])
    fl_layout.path_patterns[:0] = PATH_PATTERNS

    if mapping is None:
        mapping = {}
    for xform in block.transformations:
        if xform['name'] == 'split':
            for in_col in xform['input']:
                by = xform['by']
                splitter = {'session': analysis.layout.get_sessions,
                            'subject': analysis.layout.get_subjects}[by]()
                # Update mapping
                for var in splitter:
                    mapping['{}.{}'.format(var, in_col)] = (in_col, {by: var})
        else:
            raise ValueError("Unhandled transformation: " + xform['name'])

    for i, (_, ents) in enumerate(block.get_design_matrix()):
        fmri_glm = level2.SecondLevelModel()

        for contrast in block.contrasts:
            data = []
            for condition in contrast['condition_list']:
                real_cond, mapped_ents = mapping.get(condition, (condition, {}))
                matches = fl_layout.get(
                    type='stat',
                    contrast=snake_to_camel(real_cond),
                    **ents, **analysis.selectors, **mapped_ents)
                data.extend(match.filename for match in matches)

            out_ents = reduce(dict_intersection,
                              map(fl_layout.parse_entities, data))
            out_ents['contrast'] = snake_to_camel(contrast['name'])

            stat_fname = op.join(deriv_dir,
                                 fl_layout.build_path(out_ents, strict=True))

            if op.exists(stat_fname):
                continue

            cols = {'intercept': np.ones(len(data))}
            cname = 'intercept'
            if not np.allclose(contrast['weights'], 1):
                cname = contrast['name']
                cols[cname] = contrast['weights']

            paradigm = pd.DataFrame(cols)

            fmri_glm.fit(data, design_matrix=paradigm)
            stat = fmri_glm.compute_contrast(
                cname,
                second_level_stat_type={'T': 't', 'F': 'F'}[contrast['type']])
            data = stat.get_data()
            masked_vals = data[data != 0]
            if np.isnan(masked_vals).all():
                raise ValueError("nistats was unable to perform this contrast")
            stat.to_filename(stat_fname)

    return mapping
