import os
from os import path as op
import numpy as np
import pandas as pd
import nibabel as nb
import nilearn.image as nli
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

from bids import grabbids
from bids.analysis import base as ba


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def run(model_fname, bids_dir, preproc_dir, deriv_dir,
        subject=None, session=None, task=None, space=None):

    varsel = {key: val
              for key, val in (('subject', subject), ('session', session), ('task', task)) if val}

    analysis = ba.Analysis([bids_dir, preproc_dir], model_fname, **varsel)
    block = analysis.blocks[0]
    # analysis.setup()
    analysis.manager.load()
    block.setup(analysis.manager, None)

    varsel.update(analysis.model['input'])

    prep_layout = grabbids.BIDSLayout(preproc_dir, extensions=['derivatives'])
    confounds_file = prep_layout.get(type='confounds', **varsel)[0]

    imgsel = varsel.copy()
    if space:
        imgsel.setdefault('space', space)
    preproc = prep_layout.get(type='preproc', **imgsel)[0]
    brainmask = prep_layout.get(type='brainmask', **imgsel)[0]

    paradigm = pd.DataFrame({'trial_type': np.vectorize(lambda x: 'trial_type_' + x)(
                                analysis.manager['trial_type'].values),
                             'onset': analysis.manager['trial_type'].onsets,
                             'duration': analysis.manager['trial_type'].durations})

    confounds = pd.read_csv(confounds_file.filename, sep="\t", na_values="n/a").fillna(0)
    names = [col for col in confounds.columns
             if col.startswith('NonSteadyStateOutlier') or
             col in block.model['variables']]
    # a/tCompCor may be calculated assuming a low-pass filter
    # If used, check for a DCT basis and include
    if any(col.startswith('aCompCor') or col.startswith('tCompCor') for col in names):
        names.extend(col for col in confounds.columns
                     if col.startswith('Cosine') and col not in names)

    img = nb.load(preproc.filename)
    TR = img.header.get_zooms()[3]
    vols = img.shape[3]

    mat = dm.make_design_matrix(np.arange(vols) * TR, paradigm, drift_model=None,
                                add_regs=confounds[names], add_reg_names=names)
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
    for contrast in block.contrasts:
        cond_list = contrast['condition_list']

        var_list = mat.columns.tolist()
        indices = [var_list.index(cond) for cond in cond_list]

        weights = np.zeros(len(mat.columns))
        weights[indices] = contrast['weights']

        stat = fmri_glm.compute_contrast(weights, {'T': 't', 'F': 'F'}[contrast['type']])

        fname = op.join(out_dir, op.basename(preproc.filename)).replace(
            '_preproc.nii.gz',
            '_contrast-{}_stat.nii.gz'.format(snake_to_camel(contrast['name'])))
        stat.to_filename(fname)


def ttest(model_fname, bids_dir, preproc_dir, deriv_dir, session=None, task=None, space=None):

    varsel = {key: val
              for key, val in (('session', session), ('task', task)) if val}

    analysis = ba.Analysis([bids_dir, preproc_dir], model_fname, **varsel)
    block = analysis.blocks[0]
    # analysis.setup()
    analysis.manager.load()
    block.setup(analysis.manager, None)

    varsel.update(analysis.model['input'])

    if space:
        varsel.setdefault('space', space)

    prep_layout = grabbids.BIDSLayout(preproc_dir, extensions=['derivatives'])
    brainmasks = nli.concat_imgs(img.filename
                                 for img in prep_layout.get(type='brainmask', **varsel))
    brainmask = nli.math_img('img.any(axis=3)', img=brainmasks)
    fmri_glm = level2.SecondLevelModel(mask=brainmask)

    fl_layout = grabbids.BIDSLayout(deriv_dir, extensions=['derivatives'])
    for contrast in block.contrasts:
        # No contrast selector at this point
        stat_files = [f for f in fl_layout.get(type='stat', **varsel)
                      if 'contrast-{}'.format(snake_to_camel(contrast['name'])) in f.filename]

        basename = os.path.basename(stat_files[0].filename).split('_', 1)[1]

        paradigm = pd.DataFrame({'intercept': np.ones(len(stat_files))})
        fmri_glm.fit([img.filename for img in stat_files], design_matrix=paradigm)
        stat = fmri_glm.compute_contrast(second_level_stat_type='t')
        stat.to_filename(os.path.join(deriv_dir, basename))
