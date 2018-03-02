import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import os
from os import path as op
from functools import reduce
from scipy import stats as sps
import pandas as pd
import nibabel as nb
from nilearn import plotting as nlp
import nistats as nis
import nistats.reporting
from nistats import design_matrix as dm
from nistats import first_level_model as level1, second_level_model as level2

import pkg_resources as pkgr

from grabbit import merge_layouts
from bids import grabbids
from bids import analysis as ba

from fitlins.utils import dict_intersection, snake_to_camel
from fitlins.viz import plot_and_save, plot_corr_matrix, plot_contrast_matrix

sns.set_style('white')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['image.interpolation'] = 'nearest'

PATH_PATTERNS = (
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'task-{task}_bold[_space-{space}]_contrast-{contrast}_{type<stat>}.nii.gz',
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'task-{task}_bold[_space-{space}]_contrast-{contrast}_{type<ortho>}.png',
    'sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_]'
    'task-{task}_bold_{type<design>}.tsv',
    '[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]'
    'task-{task}_bold_{type<corr|contrasts>}.svg',
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
    orig_layout = grabbids.BIDSLayout(bids_dir)
    prep_layout = grabbids.BIDSLayout(preproc_dir, extensions=['derivatives'])
    analysis = ba.Analysis(model=model_fname,
                           layout=merge_layouts([orig_layout, prep_layout]))
    analysis.setup(**analysis.model['input'])
    analysis.layout.path_patterns[:0] = PATH_PATTERNS
    return analysis


def first_level(analysis, block, space, deriv_dir):
    analyses = []
    for paradigm, _, ents in block.get_design_matrix(block.model['HRF_variables'],
                                                     mode='sparse'):
        preproc_files = analysis.layout.get(type='preproc', space=space, **ents)
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

        mat = dm.make_design_matrix(
            frame_times=np.arange(vols) * TR,
            paradigm=paradigm.rename(columns={'condition': 'trial_type',
                                              'amplitude': 'modulation'}),
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
        plt.set_cmap('viridis')
        plot_and_save(design_fname.replace('.tsv', '.svg'),
                      nis.reporting.plot_design_matrix, mat)

        corr_ents = dm_ents.copy()
        corr_ents['type'] = 'corr'
        corr_fname = op.join(deriv_dir,
                             analysis.layout.build_path(corr_ents, strict=True))
        plot_and_save(corr_fname, plot_corr_matrix,
                      mat.drop(columns=['constant']).corr(),
                      len(block.model['HRF_variables']))

        job_desc = {
            'ents': ents,
            'subject_id': ents['subject'],
            'dataset': analysis.layout.root,
            'model_name': analysis.model['name'],
            'design_matrix_svg': design_fname.replace('.tsv', '.svg'),
            'correlation_matrix_svg': corr_fname,
            }

        cnames = [contrast['name'] for contrast in block.contrasts] + block.model['HRF_variables']
        contrast_matrix = []
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

            job_desc['contrasts_svg'] = contrasts_fname

        brainmask = analysis.layout.get(type='brainmask', space=space,
                                        **ents)[0]
        fmri_glm = None

        for contrast in contrast_matrix:
            stat_ents = preproc_ents.copy()
            stat_ents.pop('modality', None)
            stat_ents.update({'contrast': snake_to_camel(contrast),
                              'type': 'stat'})
            stat_fname = op.join(deriv_dir,
                                 analysis.layout.build_path(stat_ents,
                                                            strict=True))

            ortho_ents = stat_ents.copy()
            ortho_ents['type'] = 'ortho'
            ortho_fname = op.join(deriv_dir,
                                  analysis.layout.build_path(ortho_ents,
                                                             strict=True))

            desc = {'name': contrast, 'image_file': ortho_fname}
            if contrast not in block.model['HRF_variables']:
                job_desc.setdefault('contrasts', []).append(desc)
            else:
                job_desc.setdefault('estimates', []).append(desc)

            if op.exists(stat_fname):
                continue

            if fmri_glm is None:
                fmri_glm = level1.FirstLevelModel(mask=brainmask.filename)
                fmri_glm.fit(fname, design_matrices=mat)

            stat_types = [c['type'] for c in block.contrasts if c['name'] == contrast]
            stat_type = stat_types[0] if stat_types else 'T'
            stat = fmri_glm.compute_contrast(contrast_matrix[contrast].values,
                                             {'T': 't', 'F': 'F'}[stat_type])
            stat.to_filename(stat_fname)

            nlp.plot_glass_brain(stat, colorbar=True, plot_abs=False,
                                 display_mode='lyrz', axes=None,
                                 output_file=ortho_fname)

        analyses.append(job_desc)

    return analyses


def second_level(analysis, block, space, deriv_dir):
    fl_layout = grabbids.BIDSLayout(
        deriv_dir,
        extensions=['derivatives',
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
                          map(fl_layout.parse_entities, data))

        contrasts_ents = out_ents.copy()
        contrasts_ents['type'] = 'contrasts'
        contrasts_ents.pop('contrast', None)
        contrasts_ents.pop('space', None)
        contrasts_fname = op.join(
            deriv_dir,
            fl_layout.build_path(contrasts_ents, strict=True))

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
