import os
import json
import numpy as np
import pandas as pd
import nibabel as nb
from nistats import design_matrix as dm

from bids import grabbids, events as be

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

    fname = os.path.join(out_dir, os.path.basename(preproc.filename).replace('_preproc.nii.gz', '_design.tsv'))
    mat.to_csv(fname, sep='\t')
