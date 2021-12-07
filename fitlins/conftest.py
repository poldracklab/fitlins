import json

import pytest

from .generate_dset import DummyDerivatives


@pytest.fixture(scope="session")
def bids_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("bids")


@pytest.fixture(scope="session")
def bids_dset(bids_dir):
    database_path = bids_dir.dirpath() / "dbcache"
    layout = DummyDerivatives(base_dir=bids_dir, database_path=database_path).layout
    return layout, database_path


@pytest.fixture(scope="session")
def sample_model_dict():
    return {
        "Name": "junkfood_model001",
        "BIDSModelVersion": "1.0.0",
        "Description": "",
        "Input": {"task": "eating"},
        "Nodes": [
            {
                "Level": "run",
                "Name": "run",
                "GroupBy": ["run", "session", "subject"],
                "Transformations": {
                    "Transformer": "pybids-transforms-v1",
                    "Instructions": [
                        {"Name": "Factor", "Input": ["trial_type"]},
                        {
                            "Name": "Convolve",
                            "Input": ["trial_type.ice_cream", "trial_type.cake"],
                            "Model": "spm",
                        },
                    ],
                },
                "Model": {"X": ["trial_type.ice_cream", "trial_type.cake", "food_sweats"]},
                "DummyContrasts": {
                    "Conditions": ["trial_type.ice_cream", "trial_type.cake"],
                    "Test": "t",
                },
                "Contrasts": [
                    {
                        "Name": "icecream_gt_cake",
                        "ConditionList": ["trial_type.ice_cream", "trial_type.cake"],
                        "Weights": [1, -1],
                        "Test": "t",
                    },
                    {
                        "Name": "eating_vs_baseline",
                        "ConditionList": ["trial_type.ice_cream", "trial_type.cake"],
                        "Weights": [0.5, 0.5],
                        "Test": "t",
                    },
                ],
            },
            {
                "Level": "subject",
                "Name": "subject",
                "GroupBy": ["subject", "contrast"],
                "Model": {"X": [1], "Type": "Meta"},
                "DummyContrasts": {"Test": "t"},
            },
            {
                "Level": "dataset",
                "Name": "all_food_good_food",
                "GroupBy": ["contrast", "group"],
                "Model": {"X": [1]},
                "DummyContrasts": {
                    "Conditions": ["icecream_gt_cake", "eating_vs_baseline"],
                    "Test": "t",
                },
                "Contrasts": [
                    {
                        "Name": "all_food_good_food",
                        "ConditionList": ["trial_type.ice_cream", "trial_type.cake"],
                        "Weights": [[1, 0], [0, 1]],
                        "Test": "F",
                    }
                ],
            },
        ],
        "Edges": [
            {"Source": "run", "Destination": "subject"},
            {
                "Source": "subject",
                "Destination": "all_food_good_food",
                "Filter": {"contrast": ["icecream_gt_cake", "eating_vs_baseline"]},
            },
        ],
    }


@pytest.fixture(scope="session")
def sample_model(bids_dir, sample_model_dict):
    filepath = bids_dir.dirpath() / "sample_model.json"
    filepath.ensure()
    with open(str(filepath), "w") as model_f:
        json.dump(sample_model_dict, model_f)

    return filepath
