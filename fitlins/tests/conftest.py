
def pytest_addoption(parser):
    parser.addoption(
        '--fitlins-path', action="append", default=['fitlins'],
        help='path to fitlins executable'
    )
    parser.addoption(
        '--bids-dir', action="append", default=[],
        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
             'be found at the top level in this folder)'
    )
    parser.addoption('--output-dir', action="append", default=[],
                     help='the output path for the outcomes of preprocessing and visual reports')
    parser.addoption('--derivatives', action="append", default=[],
                     help='location of derivatives (including preprocessed images).')
    parser.addoption('--model', action="append", default=[],
                     help='location of BIDS model description')
    parser.addoption('--work-dir', action="append", default=[],
                     help='path where intermediate results should be stored')
    parser.addoption('--reference-dir', action="append", default=[],
                     help='directory containing reference results for evaluating test results')
    parser.addoption(
        '--database-path', action="append", default=[],
        help="Path to directory containing SQLite database indicies "
             "for this BIDS dataset. "
             "If a value is passed and the file already exists, "
             "indexing is skipped."
    )
    parser.addoption(
        "--test-name", action="append", default=[],
        help="name of test to run",
        choices=["nistats_smooth", "nistats_blurto", "afni_smooth", "afni_blurto"]
    )


def pytest_generate_tests(metafunc):
    opt_keys = {
        'fitlins_path',
        'bids_dir',
        'output_dir',
        'derivatives',
        'model',
        'work_dir',
        'database_path',
        'reference_dir',
        'test_name'
    }
    for ok in opt_keys:
        if ok in metafunc.fixturenames:
            metafunc.parametrize(ok, metafunc.config.getoption(ok))
