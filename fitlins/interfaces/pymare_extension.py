import nibabel as nb
import numpy as np
from nilearn.input_data import NiftiMasker
from pymare import estimators
from nilearn.image import mean_img


def remove_zero_var_voxels_(effect_data, variance_data):
    nonzero_var_mask = np.sum(variance_data == 0, 0) == 0
    variance_data_masked = variance_data[:, nonzero_var_mask]
    effect_data_masked = effect_data[:, nonzero_var_mask]
    return effect_data_masked, variance_data_masked, nonzero_var_mask

    

class pymare_model:
    def __init__(
        self,
        smoothing_fwhm=None,
        is_cifti=False
    ):
        self.smoothing_fwhm = smoothing_fwhm
        self.is_cifti = is_cifti

    def fit(
            self,
            filtered_effects=None,
            filtered_variances=None,
            design_matrix=None
    ):
        self.design_matrix_ = design_matrix.to_numpy()
        if self.is_cifti:
            effect_data = np.squeeze(
                [nb.load(effect).get_fdata(dtype='f4') for effect in filtered_effects]
            )
            variance_data = np.squeeze(
                [nb.load(variance).get_fdata(dtype='f4') for variance in filtered_variances]
            )
            self.effect_data_, self.variance_data_, self.nonzero_var_mask_ = \
                remove_zero_var_voxels_(effect_data, variance_data)

        else:
            self.masker_ = NiftiMasker(
                smoothing_fwhm=self.smoothing_fwhm,
            )
            sample_map = mean_img(filtered_effects)
            self.masker_.fit(sample_map)
            effect_data = self.masker_.transform(filtered_effects)
            variance_data = self.masker_.transform(filtered_variances)
            self.effect_data_, self.variance_data_, self.nonzero_var_mask_ = \
                remove_zero_var_voxels_(effect_data, variance_data)

        self.wls_ = estimators.WeightedLeastSquares()
        self.wls_.fit(y=self.effect_data_,
                      v=self.variance_data_,
                      X=self.design_matrix_
                      )

    def compute_contrast(
        self,
        con_val=None,
    ):
        outputs = {}
        tmp = np.einsum('ij, jkl->kl', con_val, self.wls_.params_['inv_cov'])
        outputs['effect_variance'] = np.einsum('ij, jk->k', con_val, tmp)
        outputs['effect_size'] = np.einsum('ij, jk->k', con_val, self.wls_.params_['fe_params'])
        outputs['stat'] = outputs['effect_size'] / outputs['effect_variance']**.5

        outputs_array = {}
        for image_type, image in outputs.items():
            outputs_array[image_type] = np.zeros(self.nonzero_var_mask_.shape)
            outputs_array[image_type][self.nonzero_var_mask_] = image

        if self.is_cifti is False:
            outputs_nifti = {}
            for image_type, image in outputs_array.items():
                outputs_nifti[image_type] = self.masker_.inverse_transform(image)
            return outputs_nifti
        else:
            return outputs_array
