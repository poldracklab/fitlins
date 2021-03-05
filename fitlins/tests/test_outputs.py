from subprocess import run
from pathlib import Path
import nibabel as nib
import numpy as np
import json


def test_get_nan_diff():
    ref_data = np.array([[np.nan, 0]])
    out_data = np.array([[np.nan, 0]])
    assert len(get_nan_diff('ref', 'out', ref_data, out_data)) == 0

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data)[0]
    expected_dif = 'Absolute difference (max of 1.0) greater than 0 for ref and out.'
    assert dif == expected_dif

    ref_data = np.array([[np.nan, np.nan]])
    out_data = np.array([[np.nan, 0]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data)[0]
    expected_dif = "ref nans don't match out nans."
    assert dif == expected_dif

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_abs=1)
    assert len(dif) == 0

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_abs=0.01)[0]
    expected_dif = (
        'Absolute difference (max of 0.09999999999999998)'
        ' greater than 0.01 for ref and out.'
    )
    assert dif == expected_dif

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_rel=0.5)
    assert len(dif) == 0

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_rel=0.005)[0]
    expected_dif = (
        'Relative difference (max of 0.10526315789473682) greater than 0.005 for ref and out.'
    )
    assert dif == expected_dif

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_abs=1, max_rel=0.005)
    assert len(dif) == 0

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_abs=0.01, max_rel=0.005)[0]
    expected_dif = (
        'Relative difference (max of 0.10526315789473682) greater than 0.005 for ref and out.'
    )
    assert dif == expected_dif

    ref_data = np.array([[np.nan, 1]])
    out_data = np.array([[np.nan, 0.9]])
    dif = get_nan_diff('ref', 'out', ref_data, out_data, max_abs=0.01, max_rel=0.5)
    assert len(dif) == 0


def test_outputs(fitlins_path, bids_dir, output_dir, derivatives,
                 model, work_dir,
                 database_path,
                 test_name, reference_dir):
    if test_name == "afni_smooth":
        estimator = "afni"
        smoothing = "10.0:l1:iso"
    elif test_name == "afni_blurto":
        estimator = "afni"
        smoothing = "5.0:l1:isoblurto"
    elif test_name == "nistats_smooth":
        estimator = "nistats"
        smoothing = "10.0:l1:iso"
    elif test_name == "nistats_blurto":
        estimator = "nistats"
        smoothing = "5.0:l1:isoblurto"

    opts = [
        fitlins_path,
        bids_dir,
        output_dir,
        "dataset",
        "-d", derivatives,
        "-m", model,
        "-w", work_dir,
        "--participant-label", "01", "02", "03",
        "--space", "MNI152NLin2009cAsym",
        "--estimator", estimator,
        "--smoothing", smoothing,
        "--n-cpus", '2',
        "--mem-gb", '4',
        "--drift-model", "cosine"
    ]

    if database_path is not None:
        opts.extend(["--database-path", database_path])

    print(opts)

    # run fitlins from the command line
    run(opts, check=True)

    # TODO: parameterize this
    reference_root = Path(reference_dir)
    reference_dir = reference_root / f'{test_name}/out/fitlins'

    # check niftis against reference
    ref_niis = sorted(reference_dir.glob('**/*.nii.gz'))
    for ref_nii in ref_niis:
        out_nii = Path(output_dir) / "fitlins" / ref_nii.relative_to(reference_dir)
        ref_data = nib.load(ref_nii).get_fdata(dtype=np.float64)
        out_data = nib.load(out_nii).get_fdata(dtype=np.float64)
        difs = get_nan_diff(ref_nii, out_nii, ref_data, out_data, max_abs=1e-06, max_rel=1e-04)
        assert len(difs) == 0

    # check dataset description json
    ref_json = reference_dir / 'dataset_description.json'
    out_json = Path(output_dir) / "fitlins" / ref_json.relative_to(reference_dir)

    get_json_diff(ref_json, out_json)


def get_nan_diff(ref_nii, out_nii, ref_data, out_data, max_abs=0, max_rel=0):
    res = []

    if ref_data.shape == out_data.shape:

        ref_nans = np.isnan(ref_data)
        out_nans = np.isnan(out_data)

        if (ref_nans == out_nans).all():
            ref_nonan = ref_data[~ref_nans]
            out_nonan = out_data[~out_nans]

            diff = np.abs(ref_nonan - out_nonan)
            rel_diff = (diff) / ((np.abs(ref_nonan) + np.abs(out_nonan)) / 2)
            if max_abs and max_rel:
                over = (diff > max_abs)
                diff = diff[over]
                rel_diff = rel_diff[over]
                if (rel_diff > max_rel).any():
                    res.append(f"Relative difference (max of {rel_diff.max()})"
                               f" greater than {max_rel} for {ref_nii} and {out_nii}.")
            elif max_rel:
                if (rel_diff > max_rel).any():
                    res.append(f"Relative difference (max of {rel_diff.max()})"
                               f" greater than {max_rel} for {ref_nii} and {out_nii}.")
            else:
                if ((diff > max_abs).any()):
                    res.append(f"Absolute difference (max of {diff.max()})"
                               f" greater than {max_abs} for {ref_nii} and {out_nii}.")

        else:
            res.append(f"{ref_nii} nans don't match {out_nii} nans.")

    else:
        res.append(f"{ref_nii} shape {ref_nii.shape} does not match {out_nii}"
                   f" shape {out_nii.shape}.")
    return res


def get_json_diff(ref_json, out_json):
    ref_jdat = json.loads(ref_json.read_text())['PipelineDescription']['Parameters']
    out_jdat = json.loads(out_json.read_text())['PipelineDescription']['Parameters']
    pd_check_fields = ['estimator', 'smoothing', 'participant_label']
    for cf in pd_check_fields:
        assert ref_jdat[cf] == out_jdat[cf]
