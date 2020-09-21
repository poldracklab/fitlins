import os
import numpy as np
import pandas as pd

from nipype.interfaces.base import LibraryBaseInterface, SimpleInterface, isdefined

from .abstract import (
    DesignMatrixInterface, FirstLevelEstimatorInterface, SecondLevelEstimatorInterface)


class NistatsBaseInterface(LibraryBaseInterface):
    _pkg = 'nistats'


def prepare_contrasts(contrasts, all_regressors):
    """ Make mutable copy of contrast list, and
    generate contrast design_matrix from dictionary weight mapping
    """
    if not isdefined(contrasts):
        return []

    out_contrasts = []
    for contrast in contrasts:
        # Are any necessary values missing for contrast estimation?
        missing = any([[n for n, v in row.items()
                        if v != 0 and n not in all_regressors]
                       for row in contrast['weights']])
        if not missing:
            # Fill in zeros
            weights = np.array([
                [row[col] if col in row else 0 for col in all_regressors]
                for row in contrast['weights']
                ])

            out_contrasts.append(
                (contrast['name'], weights, contrast['type']))

    return out_contrasts


def _get_voxelwise_stat(labels, results, stat):
    voxelwise_attribute = np.zeros((1, len(labels)))

    for label_ in results:
        label_mask = labels == label_
        voxelwise_attribute[:, label_mask] = getattr(results[label_], stat)

    return voxelwise_attribute


class DesignMatrix(NistatsBaseInterface, DesignMatrixInterface, SimpleInterface):

    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import design_matrix as dm
        info = self.inputs.session_info
        img = nb.load(self.inputs.bold_file)
        if isinstance(img, nb.Cifti2Image):
            vols = img.shape[0]
        elif isinstance(img, nb.Nifti1Image):
            vols = img.shape[3]
        elif isinstance(img, nb.GiftiImage):
            vols = len(img.darrays)
        else:
            raise ValueError(
                f"Unknown image type ({img.__class__.__name__}) <{self.inputs.bold_file}>")

        drop_missing = bool(self.inputs.drop_missing)

        if info['sparse'] not in (None, 'None'):
            sparse = pd.read_hdf(info['sparse'], key='sparse').rename(
                columns={'condition': 'trial_type',
                         'amplitude': 'modulation'})
            if 'modulation' in sparse.columns:
                sparse = sparse.dropna(subset=['modulation'])  # Drop NAs
        else:
            sparse = None

        if info['dense'] not in (None, 'None'):
            dense = pd.read_hdf(info['dense'], key='dense')

            missing_columns = dense.isna().all()
            if drop_missing:
                # Remove columns with NaNs
                dense = dense[dense.columns[~missing_columns]]
            elif missing_columns.any():
                missing_names = ', '.join(
                    dense.columns[missing_columns].tolist())
                raise RuntimeError(
                    f'The following columns are empty: {missing_names}. '
                    'Use --drop-missing to drop before model fitting.')

            column_names = dense.columns.tolist()
            drift_model = None if (('cosine00' in column_names) |
                                   ('cosine_00' in column_names)) else 'cosine'

            if dense.empty:
                dense = None
                column_names = None
        else:
            dense = None
            column_names = None
            drift_model = 'cosine'

        mat = dm.make_first_level_design_matrix(
            frame_times=np.arange(vols) * info['repetition_time'],
            events=sparse,
            add_regs=dense,
            add_reg_names=column_names,
            drift_model=drift_model,
        )

        mat.to_csv('design.tsv', sep='\t')
        self._results['design_matrix'] = os.path.join(runtime.cwd,
                                                      'design.tsv')
        return runtime


def dscalar_from_cifti(img, data, name):
    import numpy as np
    import nibabel as nb

    # Clear old CIFTI-2 extensions from NIfTI header and set intent
    nifti_header = img.nifti_header.copy()
    nifti_header.extensions.clear()
    nifti_header.set_intent('ConnDenseScalar')

    # Create CIFTI-2 header
    scalar_axis = nb.cifti2.ScalarAxis(np.atleast_1d(name))
    axes = [nb.cifti2.cifti2_axes.from_index_mapping(mim) for mim in img.header.matrix]
    if len(axes) != 2:
        raise ValueError(f"Can't generate dscalar CIFTI-2 from header with axes {axes}")
    header = nb.cifti2.cifti2_axes.to_header(
        axis if isinstance(axis, nb.cifti2.BrainModelAxis) else scalar_axis
        for axis in axes)

    new_img = nb.Cifti2Image(data.reshape(header.matrix.get_data_shape()), header=header,
                             nifti_header=nifti_header)
    return new_img


class FirstLevelModel(NistatsBaseInterface, FirstLevelEstimatorInterface, SimpleInterface):
    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import first_level_model as level1
        from nistats.contrasts import compute_contrast
        mat = pd.read_csv(self.inputs.design_matrix, delimiter='\t', index_col=0)
        img = nb.load(self.inputs.bold_file)

        is_cifti = isinstance(img, nb.Cifti2Image)
        if isinstance(img, nb.dataobj_images.DataobjImage):
            # Ugly hack to ensure that retrieved data isn't cast to float64 unless
            # necessary to prevent an overflow
            # For NIfTI-1 files, slope and inter are 32-bit floats, so this is
            # "safe". For NIfTI-2 (including CIFTI-2), these fields are 64-bit,
            # so include a check to make sure casting doesn't lose too much.
            slope32 = np.float32(img.dataobj._slope)
            inter32 = np.float32(img.dataobj._inter)
            if max(np.abs(slope32 - img.dataobj._slope),
                   np.abs(inter32 - img.dataobj._inter)) < 1e-7:
                img.dataobj._slope = slope32
                img.dataobj._inter = inter32

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None
        if is_cifti:
            fname_fmt = os.path.join(runtime.cwd, '{}_{}.dscalar.nii').format
            labels, estimates = level1.run_glm(img.get_fdata(dtype='f4'), mat.values)
            model_attr = {
                'r_square': dscalar_from_cifti(img,
                                               _get_voxelwise_stat(labels, estimates, 'r_square'),
                                               'r_square'),
                'log_likelihood': dscalar_from_cifti(img,
                                                     _get_voxelwise_stat(labels, estimates, 'logL'),
                                                     'log_likelihood')
            }
        else:
            fname_fmt = os.path.join(runtime.cwd, '{}_{}.nii.gz').format
            flm = level1.FirstLevelModel(
                minimize_memory=False,
                mask_img=mask_file, smoothing_fwhm=smoothing_fwhm)
            flm.fit(img, design_matrices=mat)
            model_attr = {
                'r_square': flm.r_square[0],
                'log_likelihood': flm.masker_.inverse_transform(
                    _get_voxelwise_stat(flm.labels_[0], flm.results_[0], 'logL'))
            }

        out_ents = self.inputs.contrast_info[0]['entities']

        # Save model level images

        model_maps = []
        model_metadata = []
        for attr, img in model_attr.items():
            model_metadata.append({'stat': attr, **out_ents})
            fname = fname_fmt('model', attr)
            img.to_filename(fname)
            model_maps.append(fname)

        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        contrast_metadata = []
        for name, weights, contrast_type in prepare_contrasts(
              self.inputs.contrast_info, mat.columns):
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents}
                )
            if is_cifti:
                contrast = compute_contrast(labels, estimates, weights,
                                            contrast_type=contrast_type)
                maps = {
                    map_type: dscalar_from_cifti(img, getattr(contrast, map_type)(), map_type)
                    for map_type in ['z_score', 'stat', 'p_value', 'effect_size',
                                     'effect_variance']
                }

            else:
                maps = flm.compute_contrast(weights, contrast_type,
                                            output_type='all')

            for map_type, map_list in (('effect_size', effect_maps),
                                       ('effect_variance', variance_maps),
                                       ('z_score', zscore_maps),
                                       ('p_value', pvalue_maps),
                                       ('stat', stat_maps)):

                fname = fname_fmt(name, map_type)
                maps[map_type].to_filename(fname)
                map_list.append(fname)

        self._results['effect_maps'] = effect_maps
        self._results['variance_maps'] = variance_maps
        self._results['stat_maps'] = stat_maps
        self._results['zscore_maps'] = zscore_maps
        self._results['pvalue_maps'] = pvalue_maps
        self._results['contrast_metadata'] = contrast_metadata
        self._results['model_maps'] = model_maps
        self._results['model_metadata'] = model_metadata

        return runtime


def _flatten(x):
    return [elem for sublist in x for elem in sublist]


def _match(query, metadata):
    for key, val in query.items():
        if metadata.get(key) != val:
            return False
    return True


class SecondLevelModel(NistatsBaseInterface, SecondLevelEstimatorInterface, SimpleInterface):
    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import second_level_model as level2
        from nistats import first_level_model as level1
        from nistats.contrasts import (compute_contrast, compute_fixed_effects,
                                       _compute_fixed_effects_params)

        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None

        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        contrast_metadata = []
        out_ents = self.inputs.contrast_info[0]['entities']  # Same for all

        # Only keep files which match all entities for contrast
        stat_metadata = _flatten(self.inputs.stat_metadata)
        input_effects = _flatten(self.inputs.effect_maps)
        input_variances = _flatten(self.inputs.variance_maps)

        filtered_effects = []
        filtered_variances = []
        names = []
        for m, eff, var in zip(stat_metadata, input_effects, input_variances):
            if _match(out_ents, m):
                filtered_effects.append(eff)
                filtered_variances.append(var)
                names.append(m['contrast'])

        mat = pd.get_dummies(names)
        contrasts = prepare_contrasts(self.inputs.contrast_info, mat.columns)

        is_cifti = filtered_effects[0].endswith('dscalar.nii')
        if is_cifti:
            fname_fmt = os.path.join(runtime.cwd, '{}_{}.dscalar.nii').format
        else:
            fname_fmt = os.path.join(runtime.cwd, '{}_{}.nii.gz').format

        # Only fit model if any non-FEMA contrasts at this level
        if any(c[2] != 'FEMA' for c in contrasts):
            if len(filtered_effects) < 2:
                raise RuntimeError(
                    "At least two inputs are required for a 't' for 'F' "
                    "second level contrast")
            if is_cifti:
                effect_data = np.squeeze([nb.load(effect).get_fdata(dtype='f4')
                                          for effect in filtered_effects])
                labels, estimates = level1.run_glm(effect_data, mat.values, noise_model='ols')
            else:
                model = level2.SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
                model.fit(filtered_effects, design_matrix=mat)

        for name, weights, contrast_type in contrasts:
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents})

            # Pass-through happens automatically as it can handle 1 input
            if contrast_type == 'FEMA':
                # Index design identity matrix on non-zero contrasts weights
                con_ix = weights[0].astype(bool)
                # Index of all input files "involved" with that contrast
                dm_ix = mat.iloc[:, con_ix].any(axis=1)

                contrast_imgs = np.array(filtered_effects)[dm_ix]
                variance_imgs = np.array(filtered_variances)[dm_ix]
                if is_cifti:
                    ffx_cont, ffx_var, ffx_t = _compute_fixed_effects_params(
                        np.squeeze([nb.load(fname).get_fdata(dtype='f4')
                                    for fname in contrast_imgs]),
                        np.squeeze([nb.load(fname).get_fdata(dtype='f4')
                                    for fname in variance_imgs]),
                        precision_weighted=False)
                    img = nb.load(filtered_effects[0])
                    maps = {
                        'effect_size': dscalar_from_cifti(img, ffx_cont, "effect_size"),
                        'effect_variance': dscalar_from_cifti(img, ffx_var, "effect_variance"),
                        'stat': dscalar_from_cifti(img, ffx_t, "stat")
                    }

                else:
                    ffx_res = compute_fixed_effects(contrast_imgs, variance_imgs)
                    maps = {
                        'effect_size': ffx_res[0],
                        'effect_variance': ffx_res[1],
                        'stat': ffx_res[2]
                    }
            else:
                if is_cifti:
                    contrast = compute_contrast(labels, estimates, weights,
                                                contrast_type=contrast_type)
                    img = nb.load(filtered_effects[0])
                    maps = {
                        map_type: dscalar_from_cifti(img, getattr(contrast, map_type)(), map_type)
                        for map_type in ['z_score', 'stat', 'p_value', 'effect_size',
                                         'effect_variance']
                    }
                else:
                    maps = model.compute_contrast(
                        second_level_contrast=weights,
                        second_level_stat_type=contrast_type,
                        output_type='all'
                    )

            for map_type, map_list in (('effect_size', effect_maps),
                                       ('effect_variance', variance_maps),
                                       ('z_score', zscore_maps),
                                       ('p_value', pvalue_maps),
                                       ('stat', stat_maps)):
                if map_type in maps:
                    fname = fname_fmt(name, map_type)
                    maps[map_type].to_filename(fname)
                    map_list.append(fname)

        self._results['effect_maps'] = effect_maps
        self._results['variance_maps'] = variance_maps
        self._results['stat_maps'] = stat_maps
        self._results['contrast_metadata'] = contrast_metadata

        # These are "optional" as fixed effects do not support these
        if zscore_maps:
            self._results['zscore_maps'] = zscore_maps
        if pvalue_maps:
            self._results['pvalue_maps'] = pvalue_maps

        return runtime
