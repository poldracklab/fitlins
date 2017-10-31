import os
from os import path as op
import json
import numpy as np
import pandas as pd
import nibabel as nb
import nilearn.image as nli
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

from bids import grabbids, events as be


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def run(model_fname, bids_dir, preproc_dir, deriv_dir,
        subject=None, session=None, task=None, space=None):
    with open(model_fname) as fobj:
        model = json.load(fobj)

    run_blocks = [block for block in model['blocks'] if block['level'] == 'run']
    if len(run_blocks) != 1:
        raise RuntimeError('run() requires a single run block; {} run blocks found'
                           .format('No' if not run_blocks else len(run_blocks)))
    block = run_blocks[0]

    selectors = model['input'].copy()
    for key, val in (('subject', subject), ('session', session), ('task', task)):
        if val and selectors.setdefault(key, val) != val:
            raise ValueError("Conflicting {} selection: {} {}".format(key, val, selectors[key]))

    bec = be.BIDSEventCollection(bids_dir)
    bec.read(**selectors)

    if space:
        selectors.setdefault('space', space)
    prep_layout = grabbids.BIDSLayout(preproc_dir)
    preproc = prep_layout.get(type='preproc', **selectors)[0]
    brainmask = prep_layout.get(type='brainmask', **selectors)[0]

    conditions = []
    durations = []
    onsets = []

    for hrf_var in block['model']['HRF_variables']:
        # Select the column name that forms longest prefix of hrf_var
        col = sorted((cname for cname in bec.columns
                      if hrf_var == cname or hrf_var.startswith(cname + '_')),
                     key=len, reverse=True)[0]
        if col == hrf_var:
            conditions.extend(col for _ in bec[col].durations)
            durations.extend(bec[col].durations)
            onsets.extend(bec[col].onsets)
        else:
            val = hrf_var[len(col) + 1:]
            entries = bec[col].values == val
            conditions.extend(hrf_var for _ in bec[col].values[entries])
            durations.extend(bec[col].durations[entries])
            onsets.extend(bec[col].onsets[entries])

    paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets,
                             'duration': durations})
    img = nb.load(preproc.filename)
    TR = img.header.get_zooms()[3]
    vols = img.shape[3]

    mat = dm.make_design_matrix(np.arange(vols) * TR, paradigm)
    out_dir = deriv_dir
    if subject:
        out_dir = os.path.join(out_dir, 'sub-' + subject)
    if session:
        out_dir = os.path.join(out_dir, 'ses-' + session)

    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.join(
        out_dir, os.path.basename(preproc.filename).replace('_preproc.nii.gz',
                                                            '_design.tsv'))
    mat.to_csv(fname, sep='\t')

    # Run GLM
    fmri_glm = level1.FirstLevelModel(mask=brainmask.filename)
    fmri_glm.fit(preproc.filename, design_matrices=mat)

    # Run contrast
    contrast = block['contrasts']
    cond_list = contrast['condition_list']

    var_list = mat.columns.tolist()
    indices = [var_list.index(cond) for cond in cond_list]

    weights = np.zeros(len(mat.columns))
    weights[indices] = contrast['weights']

    stat = fmri_glm.compute_contrast(weights, {'T': 't', 'F': 'F'}[contrast['type']])

    fname = op.join(out_dir, op.basename(preproc.filename)).replace(
            '_preproc.nii.gz', '_contrast-{}_stat.nii.gz'.format(snake_to_camel(contrast['name'])))
    stat.to_filename(fname)


def ttest(model_fname, bids_dir, preproc_dir, deriv_dir, session=None, task=None, space=None):
    with open(model_fname) as fobj:
        model = json.load(fobj)

    selectors = model['input'].copy()
    for key, val in (('session', session), ('task', task)):
        if val and selectors.setdefault(key, val) != val:
            raise ValueError("Conflicting {} selection: {} {}".format(key, val, selectors[key]))

    if space:
        selectors.setdefault('space', space)

    prep_layout = grabbids.BIDSLayout(preproc_dir)
    brainmasks = nli.concat_imgs(img.filename
                                 for img in prep_layout.get(type='brainmask', **selectors))
    brainmask = nli.math_img('img.any(axis=3)', img=brainmasks)

    fl_layout = grabbids.BIDSLayout(deriv_dir)
    stat_files = fl_layout.get(type='stat', **selectors)

    paradigm = pd.DataFrame({'intercept': np.ones(len(stat_files))})
    fmri_glm = level2.SecondLevelModel(mask=brainmask)
    fmri_glm.fit([img.filename for img in stat_files], design_matrix=paradigm)
    stat = fmri_glm.compute_contrast(second_level_stat_type='t')
    fname = os.path.join(deriv_dir, os.path.basename(stat_files[0].filename))
    stat.to_filename(fname)
