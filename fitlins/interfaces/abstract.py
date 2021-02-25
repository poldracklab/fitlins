"""Abstract interfaces for design matrix creation and estimation

Estimator tools should be subclassed from these interfaces, ensuring
trivial swapping of one tool for another at single points. For example,
if there is interest in comparing the design matrix construction between
tools without changing estimators, a new ``DesignMatrixInterface``
subclass can be written and swapped in without demanding that an estimator
also be written.
"""

from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits


class DesignMatrixInputSpec(TraitedSpec):
    bold_file = File(exists=True, mandatory=True)
    session_info = traits.Dict()
    drop_missing = traits.Bool(
        desc='Drop columns in design matrix with all missing values')
    drift_model = traits.Either(
        traits.String(), None,
        desc='Optional drift model to apply to design matrix'
    )


class DesignMatrixOutputSpec(TraitedSpec):
    design_matrix = File()


class DesignMatrixInterface(BaseInterface):
    input_spec = DesignMatrixInputSpec
    output_spec = DesignMatrixOutputSpec


class FirstLevelEstimatorInputSpec(TraitedSpec):
    bold_file = File(exists=True, mandatory=True)
    mask_file = traits.Either(File(exists=True), None)
    design_matrix = File(exists=True, mandatory=True)
    contrast_info = traits.List(traits.Dict)
    smoothing_fwhm = traits.Float(desc='Full-width half max (FWHM) in mm for smoothing in mask')
    smoothing_type = traits.Enum('iso', 'isoblurto', desc='Type of smoothing (iso or isoblurto)')


class EstimatorOutputSpec(TraitedSpec):
    effect_maps = traits.List(File)
    variance_maps = traits.List(File)
    stat_maps = traits.List(File)
    zscore_maps = traits.List(File)
    pvalue_maps = traits.List(File)
    contrast_metadata = traits.List(traits.Dict)
    model_maps = traits.List(File)
    model_metadata = traits.List(traits.Dict)


class FirstLevelEstimatorInterface(BaseInterface):
    input_spec = FirstLevelEstimatorInputSpec
    output_spec = EstimatorOutputSpec


class SecondLevelEstimatorInputSpec(TraitedSpec):
    effect_maps = traits.List(traits.List(File(exists=True)), mandatory=True)
    variance_maps = traits.List(traits.List(File(exists=True)))
    stat_metadata = traits.List(traits.List(traits.Dict), mandatory=True)
    contrast_info = traits.List(traits.Dict, mandatory=True)
    smoothing_fwhm = traits.Float(desc='Full-width half max (FWHM) in mm for smoothing in mask')
    smoothing_type = traits.Enum('iso', 'isoblurto', desc='Type of smoothing (iso or isoblurto)')


class SecondLevelEstimatorOutputSpec(EstimatorOutputSpec):
    contrast_metadata = traits.List(traits.Dict)


class SecondLevelEstimatorInterface(BaseInterface):
    input_spec = SecondLevelEstimatorInputSpec
    output_spec = SecondLevelEstimatorOutputSpec
