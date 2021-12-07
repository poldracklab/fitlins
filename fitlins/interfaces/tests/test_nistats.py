from fitlins.interfaces.nistats import prepare_contrasts
import numpy as np


def test_prepare_contrasts():
    contrasts = [
        {
            "name": "a",
            "conditions": ["x", "y"],
            "weights": [1, -1],
            "test": "t",
            "entities": {},
        },
        {
            "name": "b",
            "conditions": ["y", "z"],
            "weights": [1, -1],
            "test": "t",
            "entities": {},
        },
        {
            "name": "c",
            "conditions": ["x", "z"],
            "weights": [[1, 0], [0, 1]],
            "test": "F",
            "entities": {},
        },
        {
            "name": "d",
            "conditions": ["z", "y", "x"],
            "weights": [-0.5, -0.5, 1],
            "test": "t",
            "entities": {},
        },
    ]
    all_regressors = ["x", "y", "z"]
    ret = prepare_contrasts(contrasts, all_regressors)
    assert len(ret) == 4
    assert ret[0][::3] == ("a", "t")
    assert ret[1][::3] == ("b", "t")
    assert ret[2][::3] == ("c", "F")
    assert ret[3][::3] == ("d", "t")
    # Adding columns
    assert np.array_equal(ret[0][1], np.array([[1, -1, 0]]))
    assert np.array_equal(ret[1][1], np.array([[0, 1, -1]]))
    assert np.array_equal(ret[2][1], np.array([[1, 0, 0], [0, 0, 1]]))
    # Reordering columns
    assert np.array_equal(ret[3][1], np.array([[1, -0.5, -0.5]]))
