from copy import deepcopy

import pytest

from ..base import init_fitlins_wf
from ...utils import config


@pytest.mark.parametrize("estimator", ["nistats", "afni"])
def test_init_fitlins_wf(estimator, tmp_path, bids_dir, bids_dset, sample_model):
    _, database_path = bids_dset
    out_dir = bids_dir / 'derivatives' / 'fitlins'
    analysis_level = 'participant'
    space = "T1w"
    model = str(sample_model)
    desc = "preproc"

    run_fitlins = init_fitlins_wf(
        database_path,
        out_dir,
        analysis_level,
        space,
        estimator=estimator,
        model=model,
        base_dir=tmp_path,
        desc=desc
    )
    run_fitlins.config = deepcopy(config.get_fitlins_config()._sections)
    run_fitlins.config['execution']['crashdump_dir'] = tmp_path
    run_fitlins.config['execution']['crashfile_format'] = 'txt'
    run_fitlins.run()
