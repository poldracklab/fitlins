import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'

import os
from os import path as op
from functools import reduce
import numpy as np
import pandas as pd
import nibabel as nb
import nilearn.image as nli
import nistats as nis
import nistats.reporting
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

import pkg_resources as pkgr

from grabbit import merge_layouts
from bids import grabbids
from bids import analysis as ba

from fitlins.viz import plot_and_save, plot_corr_matrix, plot_contrast_matrix

PATH_PATTERNS = (
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'task-{task}_bold[_space-{space}]_contrast-{contrast}_{type}.nii.gz',
    'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_]'
    'task-{task}_bold_{type<design>}.tsv',
    'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_]'
    'task-{task}_bold_{type<corr|contrasts>}.svg',
    )

def dict_intersection(dict1, dict2):
    return {k: v for k, v in dict1.items() if dict2.get(k) == v}


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def expand_contrast_matrix(contrast_matrix, design_matrix):
    """ Ensure contrast matrix has entries for every regressor

    Parameters
    ----------
    contrast_matrix : DataFrame
        Weight matrix with contrasts as columns and regressors as rows
    design_matrix : DataFrame
        GLM design matrix with regressors as columns and TR time as rows

    Returns
    -------
    contrast_matrix : DataFrame
        Updated matrix with row for every column in ``design_matrix``, matching
        column order
    """
    full_mat = pd.DataFrame(index=design_matrix.columns,
                            columns=contrast_matrix.columns, data=0)
    full_mat.loc[contrast_matrix.index,
                 contrast_matrix.columns] = contrast_matrix

    return full_mat


def init(model_fname, bids_dir, preproc_dir):
    orig_layout = grabbids.BIDSLayout(bids_dir)
    prep_layout = grabbids.BIDSLayout(preproc_dir, extensions=['derivatives'])
    analysis = ba.Analysis(model=model_fname,
                           layout=merge_layouts([orig_layout, prep_layout]))
    analysis.setup(**analysis.model['input'])
    analysis.layout.path_patterns[:0] = PATH_PATTERNS
    return analysis


def first_level(analysis, block, deriv_dir):
    for paradigm, _, ents in block.get_design_matrix(block.model['HRF_variables'],
                                                     mode='sparse'):
        preproc_files = analysis.layout.get(type='preproc',
                                            space='MNI152NLin2009cAsym',
                                            **ents)
        # Temporary hack; pybids should never return implicit entities
        if len(preproc_files) == 0:
            del ents['run']
            preproc_files = analysis.layout.get(type='preproc',
                                                space='MNI152NLin2009cAsym',
                                                **ents)
            if len(preproc_files) == 0:
                raise ValueError("No PREPROC files found")

        if len(preproc_files) != 1:
            print(preproc_files)
            raise ValueError("Too many potential PREPROC files")

        fname = preproc_files[0].filename

        img = nb.load(fname)
        TR = img.header.get_zooms()[3]
        vols = img.shape[3]

        # Get dense portion of design matrix once TR is known
        _, confounds, _ = block.get_design_matrix(mode='dense',
                                                  sampling_rate=1/TR, **ents)[0]
        names = [col for col in confounds.columns
                 if col.startswith('NonSteadyStateOutlier') or
                 col in block.model['variables']]

        mat = dm.make_design_matrix(np.arange(vols) * TR,
                                    paradigm.rename(columns={'condition': 'trial_type'}),
                                    add_regs=confounds[names].fillna(0),
                                    add_reg_names=names,
                                    drift_model=None if 'Cosine00' in names else 'cosine',
                                    )

        preproc_ents = analysis.layout.parse_entities(fname)

        dm_ents = {k: v for k, v in preproc_ents.items()
                   if k in ('subject', 'session', 'task')}

        dm_ents['type'] = 'design'
        design_fname = op.join(deriv_dir,
                               analysis.layout.build_path(dm_ents, strict=True))
        os.makedirs(op.dirname(design_fname), exist_ok=True)
        mat.to_csv(design_fname, sep='\t')
        plot_and_save(design_fname.replace('.tsv', '.svg'),
                      nis.reporting.plot_design_matrix, mat)

        corr_ents = dm_ents.copy()
        corr_ents['type'] = 'corr'
        corr_fname = op.join(deriv_dir,
                             analysis.layout.build_path(corr_ents, strict=True))
        plot_and_save(corr_fname, plot_corr_matrix,
                      mat.drop(columns=['constant']).corr(),
                      len(block.model['HRF_variables']))

        cnames = [contrast['name'] for contrast in block.contrasts]
        if cnames:
            contrasts_ents = corr_ents.copy()
            contrasts_ents['type'] = 'contrasts'
            contrasts_fname = op.join(
                deriv_dir,
                analysis.layout.build_path(contrasts_ents, strict=True))

            contrast_matrix = expand_contrast_matrix(
                block.get_contrasts(cnames, **ents)[0][0], mat)
            plot_and_save(contrasts_fname, plot_contrast_matrix,
                          contrast_matrix.drop(['constant'], 'index'),
                          ornt='horizontal')

        base = op.basename(fname)

        brainmask = analysis.layout.get(type='brainmask', space='MNI152NLin2009cAsym',
                                        **ents)[0]
        fmri_glm = None

        for contrast in block.contrasts:
            stat_ents = preproc_ents.copy()
            stat_ents.pop('modality', None)
            stat_ents.update({'contrast': snake_to_camel(contrast['name']),
                              'type': 'stat'})
            stat_fname = op.join(deriv_dir,
                                 analysis.layout.build_path(stat_ents,
                                                            strict=True))
            indices = [mat.columns.get_loc(cond)
                       for cond in contrast['condition_list']]

            weights = np.zeros(len(mat.columns))
            weights[indices] = contrast['weights']

            if op.exists(stat_fname):
                continue

            if fmri_glm is None:
                fmri_glm = level1.FirstLevelModel(mask=brainmask.filename)
                fmri_glm.fit(fname, design_matrices=mat)

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
