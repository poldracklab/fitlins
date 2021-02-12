# coding: utf-8
from pathlib import Path
import os
import os.path as op
import numpy as np
import pandas as pd
import io
import sys
import xml.etree.ElementTree as ET
import nibabel as nb

from nipype.interfaces.afni.base import (
    AFNICommand,
    AFNICommandInputSpec,
    AFNICommandOutputSpec,
    Info,
)
from nipype.interfaces.base import traits, isdefined, File
from nipype.utils.filemanip import fname_presuffix

from .nistats import FirstLevelModel, prepare_contrasts, _flatten


STAT_CODES = nb.volumeutils.Recoder(
    (
        (0, "none", "none"),
        (1, "none", "none"),
        (2, "correlation", "Correl"),
        (3, "t test", "Ttest"),
        (4, "f test", "Ftest"),
        (5, "z score", "Zscore"),
        (6, "chi2", "Chisq"),
        (7, "beta", "Beta"),
        (8, "binomial", "Binom"),
        (9, "gamma", "Gamma"),
        (10, "poisson", "Poisson"),
        (11, "normal", "Normal"),
        (12, "non central f test", "Ftest_nonc"),
        (13, "non central chi2", "Chisq_nonc"),
        (14, "logistic", "Logistic"),
        (15, "laplace", "Laplace"),
        (16, "uniform", "Uniform"),
        (17, "non central t test", "Ttest_nonc"),
        (18, "weibull", "Weibull"),
        (19, "chi", "Chi"),
        (20, "inverse gaussian", "Invgauss"),
        (21, "extreme value 1", "Extval"),
        (22, "p value", "Pval"),
        (23, "log p value", "LogPval"),
        (24, "log10 p value", "Log10Pval"),
    ),
    fields=("code", "label", "stat_code"),
)


class FirstLevelModel(FirstLevelModel):
    def __init__(self, errorts=False, *args, **kwargs):
        super(FirstLevelModel, self).__init__(*args, **kwargs)
        self.errorts = errorts

    def _run_interface(self, runtime):
        """
        Fit a GLM using AFNI's 3dREMLfit
        """
        from nipype import logging
        import nibabel as nb
        from nipype.interfaces import afni

        import pandas as pd

        logger = logging.getLogger("nipype.interface")

        mat = pd.read_csv(self.inputs.design_matrix, delimiter="\t", index_col=0)
        contrasts = prepare_contrasts(self.inputs.contrast_info, mat.columns.tolist())
        t_r = mat.index[1]
        design_fname = op.join(runtime.cwd, "design.xmat.1D")
        stim_labels = self.get_stim_labels()
        fname_fmt = op.join(runtime.cwd, "{}_{}.nii.gz").format

        # Write AFNI style design matrix to file
        afni_design = get_afni_design_matrix(mat, contrasts, stim_labels, t_r)
        Path(design_fname).write_text(afni_design)

        img_path = self.inputs.bold_file
        img = nb.load(img_path)

        # Signal scaling occurs by default (the
        # nistats.first_level_model.FirstLevelModel class rewrites the default
        # signal_scaling argument of 0 to be True and then sets the
        # scaling_axis attribute to 0
        signal_scaling = True
        scaling_axis = 0

        # Since this estimator uses a 4d image instead of a 2d matrix the
        # dataset axes must be mapped:
        axis_mapping = {
            # mean scaling each voxel with respect to time
            0: (-1),
            # mean scaling each time point with respect to all voxels
            1: (0, 1, 2),
            # scaling with respect to voxels  and time, which is known as grand mean scaling
            (0, 1): None,
        }
        if signal_scaling:
            img_mat = img.get_fdata()
            mean = img_mat.mean(axis=axis_mapping[scaling_axis], keepdims=True)
            if (mean == 0).any():
                logger.warning(
                    "Mean values of 0 observed."
                    "The data have probably been centered."
                    "Scaling might not work as expected"
                )
            mean = np.maximum(mean, 1)
            img_mat = 100 * (img_mat / mean - 1)
            img = type(img)(img_mat, img.affine)
            img_path = "_scaled.".join(img_path.split("/")[-1].split(".", 1))
            img.to_filename(img_path)

        # Execute commands
        logger.info(
            f"3dREMLfit and 3dPval computation will be performed in: {runtime.cwd}\n"
        )

        # Define 3dREMLfit command
        remlfit = afni.Remlfit()
        remlfit.inputs.in_files = img_path
        remlfit.inputs.matrix = design_fname
        remlfit.inputs.out_file = "glt_results.nii.gz"
        remlfit.inputs.var_file = "glt_extra_variables.nii.gz"
        remlfit.inputs.wherr_file = "wherrorts.nii.gz"
        remlfit.inputs.errts_file = "errorts.nii.gz"
        remlfit.inputs.rbeta_file = "rbetas.nii.gz"
        remlfit.inputs.tout = True
        remlfit.inputs.rout = True
        remlfit.inputs.fout = True
        remlfit.inputs.verb = True
        remlfit.inputs.usetemp = True
        remlfit.inputs.goforit = True
        remlfit.inputs.mask = self.inputs.mask_file
        reml_res = remlfit.run()

        # calc smoothness
        fwhm = afni.FWHMx()
        fwhm.inputs.in_file = reml_res.outputs.wherr_file
        fwhm.inputs.out_file = fname_fmt("model", "residsmoothness").replace('.nii.gz', '.tsv')
        fwhm_res = fwhm.run()
        fwhm_dat = pd.read_csv(fwhm_res.outputs.out_file,  delim_whitespace=True, header=None)
        fwhm_dat.to_csv(fwhm_res.outputs.out_file, index=None, header=False, sep='\t')

        out_ents = self.inputs.contrast_info[0]["entities"]
        out_maps = nb.load(reml_res.outputs.out_file)
        var_maps = nb.load(reml_res.outputs.var_file)
        beta_maps = nb.load(reml_res.outputs.rbeta_file)

        model_attr_extract = {
            'r_square': (out_maps, 0),
            'log_likelihood': (var_maps, 4),
            'a': (var_maps, 0),
            'b': (var_maps, 1),
            'lam': (var_maps, 2),
            'residwhstd': (var_maps, 3),
            'LjungBox': (var_maps, 5),
        }
        # Save model level maps
        model_maps = []
        model_metadata = []
        for attr, (imgs, idx) in model_attr_extract.items():
            model_metadata.append({'stat': attr, **out_ents})
            fname = fname_fmt('model', attr)
            extract_volume(imgs, idx, f"{attr} of model", fname)
            model_maps.append(fname)

        # separate dict for maps that don't need to be extracted
        model_attr = {
            'residtsnr': self.save_tsnr(runtime, beta_maps, var_maps),
            'residsmoothness': fwhm_res.outputs.out_file
        }
        # Save error time series if people want it
        if self.errorts:
            model_attr["errorts"] = reml_res.outputs.wherr_file

        for attr, fname in model_attr.items():
            model_metadata.append({'stat': attr, **out_ents})
            model_maps.append(fname)

        # get pvals and zscore buckets (niftis with heterogenous intent codes)
        pval = Pval()
        pval.inputs.in_file = reml_res.outputs.out_file
        pval.inputs.out_file = "pval_maps.nii.gz"
        pvals = pval.run()

        zscore = Pval()
        zscore.inputs.in_file = reml_res.outputs.out_file
        zscore.inputs.out_file = "zscore_maps.nii.gz"
        zscore.inputs.zscore = True
        zscores = zscore.run()

        # create maps object
        maps = {
            "stat": out_maps,
            "z_score": nb.load(zscores.outputs.out_file),
            "p_value": nb.load(pvals.outputs.out_file),
        }
        maps["effect_size"] = maps["stat"]
        self.save_remlfit_results(maps, contrasts, runtime)
        self._results['model_maps'] = model_maps
        self._results['model_metadata'] = model_metadata
        #########################
        # Results are saved to self in save_remlfit_results, if the
        # memory saving is required it should be implemented there
        # nistats_flm.labels_.append(labels)
        # # We save memory if inspecting model details is not necessary
        # if nistats_flm.minimize_memory:
        #     for key in results:
        #         results[key] = SimpleRegressionResults(results[key])
        # nistats_flm.results_.append(results)

        return runtime

    def save_remlfit_results(self, maps, contrasts, runtime):
        """Parse  the AFNI "bucket" datasets written by 3dREMLfit and
        subsequently read using nibabel. Save the results to disk according to
        fitlins expectation.
        Parameters
        ----------
        maps : A dictionary of nibabel.interfaces.brikhead.AFNIImage objects
            keyed by output map type Description contrasts : TYPE Description
            runtime : TYPE Description
        contrasts : Object returned by nistats.contrasts.prepare_constrasts
        runtime : nipype runtime object
        """
        import nibabel as nb
        import numpy as np

        contrast_metadata = []
        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        fname_fmt = op.join(runtime.cwd, "{}_{}.nii.gz").format

        out_ents = self.inputs.contrast_info[0]["entities"]

        stats_img_info = parse_afni_ext(maps["stat"])

        stat_types = np.array(
            [
                x.split("(")[0].replace("none", "").replace("test", "")
                for x in stats_img_info["BRICK_STATSYM"].split(";")
            ]
        )

        vol_labels = parse_afni_ext(maps["effect_size"])["BRICK_LABS"].split("~")

        effect_bool = np.array([x.endswith("Coef") for x in vol_labels])
        clean_vol_labels = vol_labels[:2]
        for x in vol_labels[2:]:
            if x.endswith('_Coef') or x.endswith('_Tstat'):
                clean_vol_labels.append(x.rsplit('#', 1)[0])
            elif x.endswith('_R^2') or x.endswith('_Fstat'):
                clean_vol_labels.append(x.rsplit('_', 1)[0])
            else:
                clean_vol_labels.append(x)
        for (name, weights, contrast_type) in contrasts:
            contrast_metadata.append(
                {"contrast": name, "stat": contrast_type, **out_ents}
            )

            # Get boolean to index appropriate values
            stat_bool = stat_types == contrast_type.upper()
            contrast_bool = np.array(clean_vol_labels) == name

            # Indices for multi image nibabel object  should have length 1 and be integers
            stat_idx = np.where(contrast_bool & stat_bool)[0]
            # For multirow ftests there will be more than one index
            effect_idx = np.where(contrast_bool & effect_bool)[0]

            # Append maps:
            # for each index into the result objects stored in maps, apply a
            # modifying function as required and then append it to the
            # appropriate output list type represented by map_list and write
            # map_list to disk
            for map_type, map_list, idx_list in (
                ("effect_size", effect_maps, effect_idx),
                ("z_score", zscore_maps, stat_idx),
                ("p_value", pvalue_maps, stat_idx),
                ("stat", stat_maps, stat_idx),
            ):

                if len(effect_idx) > 1:
                    continue

                # Extract maps and info from bucket and append to relevant
                # list of maps
                for idx in idx_list:
                    imgs = maps[map_type]
                    fname = fname_fmt(name, map_type)
                    extract_volume(
                        imgs,
                        idx,
                        f"{map_type} of contrast {name}",
                        fname_fmt(name, map_type)
                    )
                    map_list.append(fname)

        # calculate effect variance
        for (name, weights, contrast_type), effect_fname, stat_fname in zip(contrasts, effect_maps, stat_maps):
            map_type = "effect_variance"
            effect_img = nb.load(effect_fname)
            effect = effect_img.get_fdata()
            stat_img = nb.load(stat_fname)
            stat = stat_img.get_fdata()
            variance = ((effect/stat)) ** 2
            variance_img = nb.Nifti1Image(variance, effect_img.affine, effect_img.header)
            variance_img.header['descrip'] = f"{map_type} of contrast {name}"

            fname = fname_fmt(name, map_type)
            variance_img.to_filename(fname)
            variance_maps.append(fname)

        self._results["effect_maps"] = effect_maps
        self._results["variance_maps"] = variance_maps
        self._results["stat_maps"] = stat_maps
        self._results["zscore_maps"] = zscore_maps
        self._results["pvalue_maps"] = pvalue_maps
        self._results["contrast_metadata"] = contrast_metadata

    def get_stim_labels(self):
        # Iterate through all weight specifications to get a list of stimulus
        # column labels.
        weights = _flatten([x["weights"] for x in self.inputs.contrast_info])
        return list(set(_flatten([x.keys() for x in weights])))

    def save_tsnr(self, runtime, rbetas, rvars):
        vol_labels = parse_afni_ext(rbetas)["BRICK_LABS"].split("~")
        mat = pd.read_csv(self.inputs.design_matrix, delimiter="\t", index_col=0)
        # find the name of the constant column
        const_name = mat.columns[(mat != 1).sum(0) == 0].values[0]
        const_idx = np.where(np.array(vol_labels) == const_name)[0]
        const_dat = rbetas.slicer[..., int(const_idx)].get_fdata()
        std_img = rvars.slicer[..., 3]
        std_dat = std_img.get_fdata()
        # scaled units are percent signal change
        # afni convention is mean of 100
        # nistat convention is mean of 0
        # for the purposes of TSNR, we'll add 100
        tsnr_dat = np.abs(const_dat + 100) / std_dat
        tsnr_img = nb.Nifti1Image(tsnr_dat, std_img.affine, std_img.header)
        tsnr_img.header['descrip'] = "residual TSNR of model"
        fname = op.join(runtime.cwd, 'model_residtsnr.nii.gz')
        tsnr_img.to_filename(fname)
        return fname


def extract_volume(imgs, idx, intent_name, fname):
    img = imgs.slicer[..., int(idx)]
    intent_info = get_afni_intent_info_for_subvol(imgs, idx)
    outmap = nb.Nifti1Image.from_image(img)
    outmap = set_intents([outmap], [intent_info])[0]
    outmap.header['descrip'] = intent_name
    outmap.to_filename(fname)


def get_afni_design_matrix(design, contrasts, stim_labels, t_r):
    """Add appropriate metadata to the design matrix and write to file for
    calling 3dREMLfit.  For a description of the target format see
    https://docs.google.com/document/d/1zpujpZYuleB7HuIFjb2vC4sYXG5M97hJ655ceAj4vE0/edit

    Parameters
    ----------
    design : pandas.DataFrame
        Matrix containing regressor for model fit.
    contrasts : output of nistats.contrasts.prepare_contrasts
    stim_labels : List of strings specifying model conditions
    t_r : float
        TR in seconds

    Returns
    -------
    str
        Design matrix with AFNI niml header
    """

    cols = list(design.columns)
    stim_col_nums = sorted([cols.index(x) for x in stim_labels])

    # Currently multi-column stimuli not supported. If they were stim_tops
    # would need to be computed
    stim_pos = "; ".join([str(x) for x in stim_col_nums])

    column_labels = "; ".join(cols)
    test_info = create_glt_test_info(design, contrasts)
    design_vals = design.to_csv(sep=" ", index=False, header=False)
    stim_labels_with_tag = ['stim_' + sl for sl in stim_labels]

    design_mat = f"""\
        # <matrix
        # ni_type = "{design.shape[1]}*double"
        # ni_dimen = "{design.shape[0]}"
        # RowTR = "{t_r}"
        # GoodList = "0..{design.shape[0] - 1}"
        # NRowFull = "{design.shape[0]}"
        # CommandLine = "{' '.join(sys.argv)}"
        # ColumnLabels = "{column_labels}"
        # {test_info}
        # Nstim = {len(stim_labels)}
        # StimBots = "{stim_pos}"
        # StimTops = "{stim_pos}"
        # StimLabels = "{'; '.join(stim_labels_with_tag)}"
        # >
        {design_vals}
        # </matrix>
        """

    design_mat = "\n".join([line.lstrip() for line in design_mat.splitlines()])
    return design_mat


def create_glt_test_info(design, contrasts):

    labels, wts_arrays, test_vals = zip(*contrasts)

    # Start defining a list containing the rows for the glt values in the
    # afni design matrix header:
    glt_list = [f'Nglt = "{len(labels)}"', f'''GltLabels = "{'; '.join(labels)}"''']

    # Convert weight arrays to csv strings
    glt_list += get_glt_rows(wts_arrays)

    # Add an empty line and start all other lines with #
    test_info = "\n# ".join([""] + glt_list)
    return test_info


def get_glt_rows(wt_arrays):
    """Generates the appropriate text for generalized linear testing in 3dREMLfit.

    Parameters
    ----------
    wt_arrays : tuple of np.arrays One of the
        Description

    Returns
    -------
    TYPE
        Description
    """
    glt_rows = []
    for ii, wt_array in enumerate(wt_arrays):
        bio = io.BytesIO()
        np.savetxt(bio, wt_array, delimiter="; ", fmt="%g", newline="; ")
        wt_str = bio.getvalue().decode("latin1")

        glt_rows.append(
            f'GltMatrix_{ii:06d} = "{wt_array.shape[0]}; {wt_array.shape[1]}; {wt_str}"'
        )

    return glt_rows


def load_bucket_by_prefix(tmpdir, prefix):
    import nibabel as nb

    bucket_path = list(Path(tmpdir).glob(prefix + "*HEAD"))
    if not len(bucket_path) == 1:
        paths = ", ".join(bucket_path)
        raise ValueError(
            f"""
            Only one file should be found for {prefix}. Instead found '{paths}'
            """
        )
    bucket_path = str(bucket_path[0])
    bucket = nb.load(bucket_path)
    bucket_labels = bucket.header.get_volume_labels()
    return bucket, bucket_labels


def set_intents(img_list, intent_info):
    for img, intent in zip(img_list, intent_info):
        img.header.set_intent(*intent)
    return img_list


def get_afni_intent_info_for_subvol(img, idx=0):
    intent_info = get_afni_intent_info(img)
    return intent_info[idx]


def get_afni_intent_info(img):
    intent_info = []
    info = None
    if isinstance(img, nb.brikhead.AFNIImage):
        info = img.header.info
    elif isinstance(img, (nb.Nifti1Image, nb.Nifti2Image)):
        info = parse_afni_ext(img)
    if not info:
        raise NotImplementedError

    if "BRICK_STATSYM" not in info:
        intent_info = [("none", ()) for x in range(len(info["BRICK_TYPES"]))]
        return intent_info

    statsyms = info["BRICK_STATSYM"].split(";")
    # Not sure this is a particularly useful thing to check
    nlabels = len(info["BRICK_LABS"].split("~"))
    if nlabels != len(statsyms):
        raise ValueError(
            f"Unexpected number of BRICK_STATSYM values : '{len(statsyms)}' instead of '{nlabels}'"
        )
    for statsym in statsyms:
        val = statsym.replace(")", "").split("(")
        if val == ["none"]:
            val.append(tuple())
            intent_info.append(tuple(val))
        else:
            params = [x for x in val[1].split(",")]
            intent_info.append(
                (STAT_CODES.label[val[0]], tuple([float(x) for x in params if x]))
            )

    return intent_info


class PvalInputSpec(AFNICommandInputSpec):
    # mandatory files
    in_file = File(
        desc="input file to 3dPval",
        argstr="%s",
        position=-1,
        mandatory=True,
        exists=True,
    )
    zscore = traits.Bool(
        usedefault=False, argstr="-zscore", desc="convert to a z-score instead"
    )
    out_file = File(
        desc="Filename (AFNI prefix) for the output.",
        name_template="%s_stat",
        argstr="-prefix %s",
    )


class Pval(AFNICommand):
    """Converts a dataset statistical sub-bricks to p-values, or optionally
    zscores. All output volumes will be converted to float format, bub-bricks
    that are not marked as statistical volumes are otherwise unchanged.

    For complete details, see the `3dPval Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dPval.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni
    >>> pval = afni.Pval()
    >>> pval.inputs.in_file = 'functional.nii'
    >>> pval.inputs.out_file = 'output.nii'
    >>> pval.cmdline
    '3dPval -in_file functional.nii -prefix output.nii'
    >>> res = pval.run()  # doctest: +SKIP
    """

    _cmd = "3dPval"
    input_spec = PvalInputSpec
    output_spec = AFNICommandOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        return super(Pval, self)._parse_inputs(skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()

        if not isdefined(self.inputs.out_file):
            prefix = self._gen_fname(self.inputs.in_file, suffix="_pval")
            outputtype = self.inputs.outputtype
            if outputtype == "AFNI":
                ext = ".HEAD"
                suffix = "+tlrc"
            else:
                ext = Info.output_type_to_ext(outputtype)
                suffix = ""
        else:
            prefix = self.inputs.out_file
            ext_ind = max(
                [prefix.lower().rfind(".nii.gz"), prefix.lower().rfind(".nii")]
            )
            if ext_ind == -1:
                ext = ".HEAD"
                suffix = "+tlrc"
            else:
                ext = prefix[ext_ind:]
                suffix = ""

        # All outputs should be in the same directory as the prefix
        out_dir = os.path.dirname(os.path.abspath(prefix))

        outputs["out_file"] = (
            fname_presuffix(prefix, suffix=suffix, use_ext=False, newpath=out_dir) + ext
        )

        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_fname(self.inputs.in_file, suffix="_pval")


def parse_afni_ext(nifti_file):

    afni_extension = None
    for ext in nifti_file.header.extensions:
        if ext.get_code() == 4:
            afni_extension = ext

    if not afni_extension:
        return None

    root = ET.fromstring(afni_extension.get_content())
    type_mapping = {
        "int": "integer-attribute",
        "float": "float-attribute",
        "String": "string-attribute",
    }
    varlist = []
    for attribute in root:
        vtype = type_mapping[attribute.attrib["ni_type"]]
        vname = attribute.attrib["atr_name"]
        vcount = attribute.attrib["ni_dimen"]
        vval = attribute.text.strip('\n "').replace('"\n "','')
        # Create a string object equivalent to what is observed when
        # parsing an AFNI ".HEAD" file.
        tmp = "type = {vtype}\nname = {vname}\ncount = {vcount}\n{vval}\n"
        varlist.append(tmp.format(**locals()))

    return {k: v for k, v in map(nb.brikhead._unpack_var, varlist)}
