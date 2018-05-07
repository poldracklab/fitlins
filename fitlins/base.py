import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import os
from os import path as op
from functools import reduce
from scipy import stats as sps
import pandas as pd
from nilearn import plotting as nlp
import nistats.reporting  # noqa: F401
from nistats import second_level_model as level2

import pkg_resources as pkgr

from bids import grabbids
from bids import analysis as ba

from fitlins.utils import dict_intersection, snake_to_camel
from fitlins.viz import plot_and_save, plot_contrast_matrix

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['image.interpolation'] = 'nearest'

PATH_PATTERNS = (
    ('[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
     'task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
     '[_echo-{echo}]_bold[_space-{space}]_contrast-{contrast}_'
     '{type<stat>}.nii.gz'),
    ('[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
     'task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
     '[_echo-{echo}]_bold[_space-{space}]_contrast-{contrast}_'
     '{type<ortho>}.png'),
    ('sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_]'
     'task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
     '[_echo-{echo}]_bold_{type<design>}.tsv'),
    ('[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
     'task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
     '[_echo-{echo}]_bold_{type<corr|contrasts>}.svg'),
    )


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
    analysis = ba.Analysis(model=model_fname,
                           layout=grabbids.BIDSLayout([bids_dir, preproc_dir]))
    analysis.setup(**analysis.model['input'])
    analysis.layout.path_patterns[:0] = PATH_PATTERNS
    return analysis


def second_level(analysis, block, space, deriv_dir):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        config=['bids', 'derivatives',
                pkgr.resource_filename('fitlins', 'data/fitlins.json')])
    fl_layout.path_patterns[:0] = PATH_PATTERNS

    analyses = []

    # pybids likes to give us a lot of extraneous columns
    cnames = [contrast['name'] for contrast in block.contrasts]
    fmri_glm = level2.SecondLevelModel()
    for contrasts, idx, ents in block.get_contrasts(names=cnames):
        if contrasts.empty:
            continue

        data = []
        for in_name, sub_ents in zip(contrasts.index, idx.to_dict(orient='record')):
            # The underlying contrast name might have been added to by a transform
            for option in [in_name] + in_name.split('.'):
                files = fl_layout.get(contrast=snake_to_camel(option),
                                      type='stat', space=space, **sub_ents)
                if files:
                    data.append(files[0].filename)
                    break
            else:
                raise ValueError("Unknown input: {}".format(in_name))

        out_ents = reduce(dict_intersection,
                          map(fl_layout.parse_file_entities, data))

        contrasts_ents = out_ents.copy()
        contrasts_ents['type'] = 'contrasts'
        contrasts_ents.pop('contrast', None)
        contrasts_ents.pop('space', None)
        contrasts_fname = op.join(
            deriv_dir,
            fl_layout.build_path(contrasts_ents, strict=True))

        # Make parent results directory
        os.makedirs(os.path.dirname(contrasts_fname), exist_ok=True)
        plot_and_save(contrasts_fname, plot_contrast_matrix, contrasts,
                      ornt='horizontal')

        job_desc = {
            'ents': out_ents,
            'subject_id': ents.get('subject'),
            'dataset': analysis.layout.root,
            'model_name': analysis.model['name'],
            'contrasts_svg': contrasts_fname,
            }

        for contrast in contrasts:
            out_ents['contrast'] = snake_to_camel(contrast)

            stat_fname = op.join(deriv_dir,
                                 fl_layout.build_path(out_ents, strict=True))

            ortho_ents = out_ents.copy()
            ortho_ents['type'] = 'ortho'
            ortho_fname = op.join(deriv_dir,
                                  analysis.layout.build_path(ortho_ents,
                                                             strict=True))

            desc = {'name': contrast, 'image_file': ortho_fname}
            job_desc.setdefault('contrasts', []).append(desc)

            if op.exists(stat_fname):
                continue

            cols = {'intercept': np.ones(len(data))}
            cname = 'intercept'
            if not np.allclose(contrasts[contrast], 1):
                cname = contrast
                cols[contrast] = contrasts[contrast]

            paradigm = pd.DataFrame(cols)

            fmri_glm.fit(data, design_matrix=paradigm)
            stat_type = [c['type'] for c in block.contrasts if c['name'] == contrast][0]
            stat = fmri_glm.compute_contrast(
                cname,
                second_level_stat_type={'T': 't', 'F': 'F'}[stat_type],
                )
            data = stat.get_data()
            masked_vals = data[data != 0]
            if np.isnan(masked_vals).all():
                raise ValueError("nistats was unable to perform this contrast")
            stat.to_filename(stat_fname)

            nlp.plot_glass_brain(stat, colorbar=True,
                                 threshold=sps.norm.isf(0.001), plot_abs=False,
                                 display_mode='lyrz', output_file=ortho_fname)

        analyses.append(job_desc)

    return analyses
