# coding: utf-8
import os
import numpy as np
import pandas as pd
import os.path as op
import subprocess as sp
import pytest
import sys

from collections import namedtuple
from nibabel.testing import data_path
import nibabel as nb

from fitlins.interfaces.nistats import prepare_contrasts
from fitlins.interfaces.afni import (
    create_glt_test_info,
    set_intents,
    get_afni_design_matrix,
    get_glt_rows,
    get_afni_intent_info,
    get_afni_intent_info_for_subvol,
)

ContrastInfo = namedtuple('ContrastInfo', ('name', 'conditions', 'weights',
                                           'test', 'entities'))

def get_reml_bucket_test_data():
    afni_test_dir = "/tmp/afni_reml_test_data"
    base_url = "https://afni.nimh.nih.gov/pub/dist/data/afni_ci_test_data/"
    brik_url = ".git/annex/objects/Xf/MM/MD5E-s9120000--fcf7ebb9679ae37c9db37c02a927193f.BRIK/MD5E-s9120000--fcf7ebb9679ae37c9db37c02a927193f.BRIK"
    head_url = ".git/annex/objects/f4/62/MD5E-s12015--3c2c821cd46d0cd47ae32c1bb8f4d8cc.HEAD/MD5E-s12015--3c2c821cd46d0cd47ae32c1bb8f4d8cc.HEAD"
    brick = op.join(afni_test_dir, "reml_bucket.BRIK")
    os.makedirs(afni_test_dir, exist_ok=True)
    if not os.path.exists(brick):
        sp.run(f"wget -O {brick} {op.join(base_url,brik_url)}", shell=True)
    head = op.join(afni_test_dir, "reml_bucket.HEAD")
    if not os.path.exists(head):
        sp.run(f"wget -O {head} {op.join(base_url,head_url)}", shell=True)

    return nb.load(brick)


def test_get_afni_design_matrix():
    entities = {
            "space": "MNI152NLin2009cAsym",
            "subject": "01",
            "task": "rhymejudgment",
    }

    contrast_info = [
        ContrastInfo(
            'a_test',
            ['trial_type.pseudoword', 'trial_type.word'],
            [2, 5],
            'F',
            entities,
        ),
        ContrastInfo(
            'a_test',
            ['trial_type.pseudoword', 'trial_type.word'],
            [1, -5],
            'F',
            entities,
        )
    ]

    design = pd.DataFrame(
        {
            "trial_type.pseudoword": [11.2, 1],
            "trial_type.word": [20, -1],
            "noise": [7, 7],
            "drift": [1, 2],
        }
    )

    contrasts = prepare_contrasts(contrast_info, design.columns.tolist())

    t_r = 2
    stim_labels = ["trial_type.pseudoword", "trial_type.word"]
    stim_labels_with_tag = ['stim_' + sl for sl in stim_labels]
    test_info = create_glt_test_info(design, contrasts)
    design_vals = design.to_csv(sep=" ", index=False, header=False)
    cols = "; ".join(design.columns)

    expected = f"""\
        # <matrix
        # ni_type = "{design.shape[1]}*double"
        # ni_dimen = "{design.shape[0]}"
        # RowTR = "{t_r}"
        # GoodList = "0..{design.shape[0] - 1}"
        # NRowFull = "{design.shape[0]}"
        # CommandLine = "{' '.join(sys.argv)}"
        # ColumnLabels = "{cols}"
        # {test_info}
        # Nstim = 2
        # StimBots = "0; 1"
        # StimTops = "0; 1"
        # StimLabels = "{'; '.join(stim_labels_with_tag)}"
        # >
        {design_vals}
        # </matrix>
        """
    expected = "\n".join([x.lstrip() for x in expected.splitlines()])
    assert expected == get_afni_design_matrix(design, contrasts, stim_labels, t_r)


def test_create_glt_test_info():

    entities = {
            "space": "MNI152NLin2009cAsym",
            "subject": "01",
            "task": "rhymejudgment",
    }

    contrast_info = [
        ContrastInfo(
            'a_test',
            ['trial_type.pseudoword', 'trial_type.word'],
            [2, 5],
            'F',
            entities,
        ),
        ContrastInfo(
            'a_test',
            ['trial_type.pseudoword', 'trial_type.word'],
            [1, -5],
            'F',
            entities,
        )
    ]

    design = pd.DataFrame(
        {
            "trial_type.pseudoword": [11.2, 1],
            "trial_type.word": [20, -1],
            "noise": [7, 7],
            "drift": [1, 2],
        }
    )
    contrasts = prepare_contrasts(contrast_info, design.columns.tolist())

    expected = f"""
# Nglt = "2"
# GltLabels = "a_test; a_test"
# GltMatrix_000000 = "1; 4; 2; 5; 0; 0; "
# GltMatrix_000001 = "1; 4; 1; -5; 0; 0; "\
"""
    assert expected == create_glt_test_info(design, contrasts)


def test_get_glt_rows():
    wt_arrays = (np.array([[0, 1, -1]]), np.array([[0, 1, 1], [-1, -1, -1]]))
    assert get_glt_rows(wt_arrays) == [
        'GltMatrix_000000 = "1; 3; 0; 1; -1; "',
        'GltMatrix_000001 = "2; 3; 0; 1; 1; -1; -1; -1; "',
    ]


def test_set_intents():
    import nibabel as nb

    img = nb.Nifti1Image(np.zeros(5), None)
    img_list = set_intents([img], [("f test", (2, 10))])

    expected = ("f test", (2.0, 10.0), "")
    assert expected == img_list[0].header.get_intent()


def test_get_afni_intent_info_for_subvol():
    brick = get_reml_bucket_test_data()
    intent_info = get_afni_intent_info_for_subvol(brick, 1)
    assert intent_info == ("t test", (420,))


def test_get_afni_intent_info():

    img = nb.load(op.join(data_path, "example4d+orig.HEAD"))
    img.header.info["BRICK_STATSYM"] = "Ftest(2,10);none;none"

    expected = [("f test", (2, 10)), ("none", ()), ("none", ())]
    assert expected == get_afni_intent_info(img)

    img.header.info.pop("BRICK_STATSYM")
    assert get_afni_intent_info(img) == [("none", ()), ("none", ()), ("none", ())]

    img.header.info["BRICK_STATSYM"] = "Zscore();none;none"
    assert get_afni_intent_info(img) == [("z score", ()), ("none", ()), ("none", ())]

    with pytest.raises(ValueError):
        img.header.info["BRICK_STATSYM"] = "Zscore();none"
        get_afni_intent_info(img)
