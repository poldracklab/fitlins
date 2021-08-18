import os
from copy import deepcopy
from pathlib import Path

import pytest
from bids.modeling import BIDSStatsModelsGraph

from ...utils import config
from ..base import init_fitlins_wf


@pytest.mark.parametrize("estimator", ["nistats", "afni"])
def test_init_fitlins_wf(estimator, tmp_path, bids_dir, bids_dset, sample_model):
    layout, database_path = bids_dset
    out_dir = bids_dir / "derivatives" / "fitlins"
    analysis_level = "participant"
    space = "T1w"
    model = str(sample_model)
    desc = "preproc"

    import json

    if os.path.exists(model):
        model_dict = json.loads(Path(model).read_text())
    graph = BIDSStatsModelsGraph(layout, model_dict)

    run_fitlins = init_fitlins_wf(
        database_path,
        out_dir,
        graph,
        analysis_level=analysis_level,
        space=space,
        estimator=estimator,
        model=model,
        base_dir=tmp_path,
        desc=desc,
    )
    run_fitlins.config = deepcopy(config.get_fitlins_config()._sections)
    run_fitlins.config["execution"]["crashdump_dir"] = tmp_path
    run_fitlins.config["execution"]["crashfile_format"] = "txt"
    run_fitlins.run()
