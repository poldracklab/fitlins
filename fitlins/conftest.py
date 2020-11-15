import json
from itertools import product
from tempfile import mkdtemp
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from nistats.hemodynamic_models import compute_regressor
import nibabel as nib
import bids
from bids.layout.writing import build_path


def write_metadata(filepath, metadata):
    filepath.ensure()
    with open(str(filepath), 'w') as meta_file:
        json.dump(metadata, meta_file)


class RegressorFileCreator():
    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}][_space-{space}][_cohort-{cohort}][_desc-{desc}]_"
        "{suffix<timeseries|regressors>|timeseries}{extension<.json|.tsv>|.tsv}"
    )
    FILE_PARAMS = {"suffix": "regressors", "datatype": "func"}

    def __init__(self, base_dir, fname_params, regr_names, n_tp, metadata=None):
        self.base_dir = base_dir
        self.metadata = metadata
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".tsv"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}

        self.init_data(regr_names, n_tp)
        self.create_fname(fname_params, meta_params)

    def init_data(self, regr_names, n_tp):
        self.noise_df = pd.DataFrame(
            {name: np.random.random(self.n_tp) for name in self.regr_names}
        )

    def create_fname(self):
        self.fname = self.base_dir / build_path(self.fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(self.meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.noise_df.to_csv(self.fname, sep='\t', index=False)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname


class DerivFuncFileCreator():
    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}][_space-{space}][_cohort-{cohort}][_res-{resolution}]"
        "[_desc-{desc}]_{suffix<bold|cbv|phase|sbref|boldref|dseg>}"
        "{extension<.nii|.nii.gz|.json>|.nii.gz}"
    )
    FILE_PARAMS = {"suffix": "bold"}

    def __init__(
        self, base_dir, fname_params, events_df, trial_type_weights, noise_df, n_tp, cnr, metadata
    ):
        self.base_dir = base_dir
        self.metadata = metadata
        self.fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".nii.gz"}
        self.meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.events_df = events_df
        self.noise_df = noise_df
        self.trial_type_weights = trial_type_weights
        self.n_tp = n_tp
        self.cnr = cnr
        self.init_data()
        self.create_fname()

    def _create_signal(tr, n_tp, events_df, trial_type_weights):
        frame_times = np.arange(0, int(n_tp * tr), step=int(tr))
        signal = np.zeros(frame_times.shape)
        trial_types = events_df['trial_type'].unique()
        for condition, weight in zip(trial_types, trial_type_weights):
            exp_condition = events_df.query(
                f"trial_type == '{condition}'"
            )[['onset', 'duration']].values.T
            exp_condition = np.vstack([exp_condition, np.repeat(weight, exp_condition.shape[1])])
            signal += compute_regressor(
                exp_condition, "glover", frame_times, con_id=condition)[0].squeeze()

        return signal

    def _aggregate_noise(noise_df):
        return noise_df.values.mean(axis=1)

    def _create_nii(timeseries):
        brain_data = np.zeros((9, 9, 9, len(timeseries)))
        brain_data[2:4, 2:4, 2:4, :] = timeseries
        return nib.Nifti1Image(brain_data, affine=np.eye(4))

    def init_data(self):
        tr = self.metadata['RepetitionTime']
        signal = self._create_signal(tr, self.n_tp, self.events_df)
        noise = self._aggregate_noise(self.noise_df)
        contrast = signal.max()
        signal_scaling_factor = contrast * self.cnr * noise.std()
        timeseries = (signal * signal_scaling_factor) + noise
        self.data = self._create_nii(timeseries)

    def create_fname(self):
        self.fname = self.base_dir / build_path(self.fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(self.meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.data.to_filename(self.fname.strpath)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname

class DerivMaskFileCreator():

    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}][_space-{space}][_cohort-{cohort}][_res-{resolution}]"
        "_desc-{desc}_{suffix<mask>|mask}{extension<.nii|.nii.gz|.json>|.nii.gz}"
    )
    FILE_PARAMS = {"suffix": "mask"}

    def __init__(
        self, base_dir, fname_params, func_img, metadata=None
    ):
        self.base_dir = base_dir
        self.metadata = metadata
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".nii.gz"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.init_data(func_img)
        self.create_fname(fname_params, meta_params)

    def init_data(self, func_img):
        mask_data = (func_img.get_fdata()[:, :, :, 0] != 0).astype(int)
        return nib.Nifti1Image(mask_data, np.eye(4))

    def create_fname(self, fname_params, meta_params):
        self.fname = self.base_dir / build_path(fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.data.to_filename(self.fname.strpath)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname


class FuncFileCreator():
    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}]_{suffix<bold|cbv|phase|sbref>}{extension<.nii|.nii.gz|.json>|.nii.gz}"
    )
    FILE_PARAMS = {"suffix": "bold"}

    def __init__(
        self, base_dir, fname_params, events_df, trial_type_weights, noise_df, n_tp, cnr, metadata
    ):
        self.base_dir = base_dir
        self.metadata = metadata
        self.fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".nii.gz"}
        self.meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.events_df = events_df
        self.noise_df = noise_df
        self.trial_type_weights = trial_type_weights
        self.n_tp = n_tp
        self.cnr = cnr
        self.init_data()
        self.create_fname()

    def _create_signal(tr, n_tp, events_df, trial_type_weights):
        frame_times = np.arange(0, int(n_tp * tr), step=int(tr))
        signal = np.zeros(frame_times.shape)
        trial_types = events_df['trial_type'].unique()
        for condition, weight in zip(trial_types, trial_type_weights):
            exp_condition = events_df.query(
                f"trial_type == '{condition}'"
            )[['onset', 'duration']].values.T
            exp_condition = np.vstack([exp_condition, np.repeat(weight, exp_condition.shape[1])])
            signal += compute_regressor(
                exp_condition, "glover", frame_times, con_id=condition)[0].squeeze()

        return signal

    def _aggregate_noise(noise_df):
        return noise_df.values.mean(axis=1)

    def _create_nii(timeseries):
        brain_data = np.zeros((9, 9, 9, len(timeseries)))
        brain_data[2:4, 2:4, 2:4, :] = timeseries
        return nib.Nifti1Image(brain_data, affine=np.eye(4))

    def init_data(self):
        tr = self.metadata['RepetitionTime']
        signal = self._create_signal(tr, self.n_tp, self.events_df)
        noise = self._aggregate_noise(self.noise_df)
        contrast = signal.max()
        signal_scaling_factor = contrast * self.cnr * noise.std()
        timeseries = (signal * signal_scaling_factor) + noise
        self.data = self._create_nii(timeseries)

    def create_fname(self):
        self.fname = self.base_dir / build_path(self.fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(self.meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.data.to_filename(self.fname.strpath)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname


class EventsFileCreator():
    PATTERN = (
        "sub-{subject}[/ses-{session}]/[{datatype<func|meg|beh>|func}/]"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_rec-{reconstruction}][_run-{run}][_echo-{echo}][_recording-{recording}]"
        "_{suffix<events>}{extension<.tsv|.json>|.tsv}"
    )
    FILE_PARAMS = {"suffix": "events", "datatype": "func"}

    def __init__(self, base_dir, fname_params, n_events, trial_types, event_duration,
                 inter_trial_interval, metadata=None):
        self.base_dir = base_dir
        self.metadata = metadata
        self.fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".tsv"}
        self.meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.n_events = n_events
        self.trial_types = trial_types
        self.event_duration = event_duration
        self.init_data()
        self.create_fname()

    def init_data(self):
        events_dict = {}
        n_trial_types = len(self.trial_types)
        experiment_duration = int(n_trial_types * self.n_events * self.inter_trial_interval)
        events_dict['onset'] = np.arange(0, experiment_duration, self.inter_trial_interval)
        events_dict['trial_type'] = self.trial_types * self.n_events
        events_dict['duration'] = [self.event_duration] * n_trial_types * self.n_events
        self.experiment_duration = experiment_duration
        self.events_df = pd.DataFrame(events_dict)

    def create_fname(self):
        self.fname = self.base_dir / build_path(self.fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(self.meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.events_df.to_csv(self.fname, sep='\t', index=False)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname


class DummyDerivatives():
    DERIVATIVE_DICT = {
        "Name": "fMRIPrep - fMRI PREProcessing workflow",
        "BIDSVersion": "1.4.1",
        "PipelineDescription": {
            "Name": "fMRIPrep",
            "Version": "1.5.0rc2+14.gf673eaf5",
            "CodeURL": "https://github.com/nipreps/fmriprep/archive/1.5.0.tar.gz"
        },
        "CodeURL": "https://github.com/nipreps/fmriprep",
        "HowToAcknowledge": "Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4)",
        "SourceDatasetsURLs": [
            "https://doi.org/"
        ],
        "License": ""
    }

    BIDS_DICT = {
        "Name": "ice cream and cake",
        "BIDSVersion": "1.4.1",
    }

    def __init__(
        self,
        base_dir=None,
        participant_labels=None,
        session_labels=None,
        task_labels=None,
        run_labels=None,
        trial_types=None,
        trial_type_weights=None,
        n_events=None,
        event_duration=None,
        inter_trial_interval=None,
        cnr=None,
        regr_names=None,
        func_metadata=None,
    ):
        self.base_dir = base_dir or Path(mkdtemp(suffix="bids"))
        self.participant_labels = participant_labels or ["bert", "ernie", "gritty"]
        self.session_labels = session_labels or ["breakfast", "lunch"]
        self.task_labels = task_labels or ["eating"]
        self.run_labels = run_labels or ["01", "02"]
        self.trial_types = trial_types or ["ice_cream", "cake"]
        self.trial_type_weights = trial_type_weights or list(range(1, len(self.trial_types)))
        self.n_events = n_events or 15
        self.event_duration = event_duration or 1
        self.inter_trial_interval or 20
        self.cnr = cnr or 2
        self.regr_names = regr_names or ["food_sweats", "sugar_jitters"]
        self.func_metadata = func_metadata or {"RepetitionTime": 2.0, "SkullStripped": False}

        bids_dataset_json = base_dir.ensure("dataset_description.json")

        with open(str(bids_dataset_json), 'w') as dj:
            json.dump(self.BIDS_DICT, dj)

        deriv_dir = base_dir.ensure('derivatives', 'fmriprep', dir=True)
        deriv_dataset_json = deriv_dir.ensure("dataset_description.json")

        with open(str(deriv_dataset_json), 'w') as dj:
            json.dump(self.DERIV_DICT, dj)

        unique_scans = product(
            self.participant_labels,
            self.session_labels or (None,),
            self.task_labels or (None,),
            self.run_labels or (None,),
        )
        param_order = ['subject', 'session', 'task', 'run']

        for scan_params in unique_scans:
            file_params = {k: v for k, v in zip(param_order, scan_params)}
            # create events file
            events = EventsFileCreator(
                self.base_dir, file_params, self.n_events, self.trial_types,
                self.event_duration, self.inter_trial_interval
            )
            events.write_file()
            # calculate number of timepoints
            n_tp = int(events.experiment_duration // self.func_metadata["RepetitionTime"])
            # create noise file
            noise = RegressorFileCreator(self.base_dir, file_params, self.regr_names, n_tp)
            noise.write_file()
            # create bids func file
            bids_func = FuncFileCreator(
                self.base_dir, file_params, events.events_df,
                self.trial_type_weights, noise.noise_df, n_tp,
                self.cnr, self.func_metadata,
            )
            # create deriv func file
            deriv_func = DerivFuncFileCreator(
                self.base_dir, file_params, events.events_df,
                self.trial_type_weights, noise.noise_df, n_tp,
                self.cnr, self.func_metadata,
            )


@pytest.fixture(scope="session")
def func_metadata():
    return {
        "RepetitionTime": 2.0,
        "SkullStripped": False,
    }


@pytest.fixture(scope="session")
def events_info():
    n_events = 15
    # in seconds
    inter_trial_interval = 20.0
    duration = 1.0
    trial_types = ["ice_cream", "cake"]
    trial_weights = [2, 1]

    return n_events, inter_trial_interval, duration, trial_types, trial_weights


@pytest.fixture(scope="session")
def events_df(events_info):
    events_dict = {}

    n_events, inter_trial_interval, duration, trial_types, _ = events_info

    n_trial_types = len(trial_types)
    experiment_duration = int(n_trial_types * n_events * inter_trial_interval)
    events_dict['onset'] = np.arange(0, experiment_duration, inter_trial_interval)
    events_dict['trial_type'] = trial_types * n_events
    events_dict['duration'] = [duration] * n_trial_types * n_events

    return pd.DataFrame(events_dict), experiment_duration


@pytest.fixture(scope="session")
def signal(events_df, func_metadata, events_info):
    n_events, inter_trial_interval, duration, trial_types, trial_weights = events_info
    events_df, experiment_duration = events_df
    tr = func_metadata['RepetitionTime']
    frame_times = np.arange(0, experiment_duration, step=int(tr))

    signal = np.zeros(frame_times.shape)
    for condition, weight in zip(trial_types, trial_weights):
        exp_condition = events_df.query(
            f"trial_type == '{condition}'"
        )[['onset', 'duration']].values.T
        exp_condition = np.vstack([exp_condition, np.repeat(weight, n_events)])
        signal += compute_regressor(
            exp_condition, "glover", frame_times, con_id=condition)[0].squeeze()

    return signal


@pytest.fixture(scope="session")
def noise(signal):
    return np.random.random(signal.shape)


@pytest.fixture(scope="session")
def func_img(signal, noise):
    cnr = 2
    contrast = signal.max()
    signal_scaling_factor = contrast * cnr * noise.std()
    voxel = (signal * signal_scaling_factor) + noise
    brain_data = np.zeros((9, 9, 9, len(voxel)))
    brain_data[2:4, 2:4, 2:4, :] = voxel
    return nib.Nifti1Image(brain_data, affine=np.eye(4))


@pytest.fixture(scope="session")
def mask_img(func_img):
    mask_data = (func_img.get_fdata()[:, :, :, 0] != 0).astype(int)
    return nib.Nifti1Image(mask_data, np.eye(4))


@pytest.fixture(scope='session')
def bids_dir(tmpdir_factory):
    bids_dir = tmpdir_factory.mktemp('bids')

    dataset_json = bids_dir.ensure("dataset_description.json")

    dataset_dict = {
        "Name": "ice cream and cake",
        "BIDSVersion": "1.4.1",
    }

    with open(str(dataset_json), 'w') as dj:
        json.dump(dataset_dict, dj)

    return bids_dir


@pytest.fixture(scope='session')
def deriv_dir(bids_dir):
    deriv_dir = bids_dir.ensure('derivatives',
                                'fmriprep',
                                dir=True)

    dataset_json = deriv_dir.ensure("dataset_description.json")

    dataset_dict = {
        "Name": "fMRIPrep - fMRI PREProcessing workflow",
        "BIDSVersion": "1.1.1",
        "PipelineDescription": {
            "Name": "fMRIPrep",
            "Version": "1.5.0rc2+14.gf673eaf5",
            "CodeURL": "https://github.com/poldracklab/fmriprep/archive/1.5.0.tar.gz"
        },
        "CodeURL": "https://github.com/poldracklab/fmriprep",
        "HowToAcknowledge": "Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4)",
        "SourceDatasetsURLs": [
            "https://doi.org/"
        ],
        "License": ""
    }

    with open(str(dataset_json), 'w') as dj:
        json.dump(dataset_dict, dj)

    return deriv_dir


@pytest.fixture(scope='session')
def participant_labels():
    return ['01', '02', '03']


@pytest.fixture(scope='session')
def run_labels():
    return None


@pytest.fixture(scope='session')
def session_labels():
    return None


@pytest.fixture(scope='session')
def task_labels():
    return ['eating']


def create_deriv_filepaths(base_dir, participant_labels, session_labels, task_labels, run_labels):
    with open(bids.config.get_option('config_paths')['bids']) as b_con:
        bids_config = json.load(b_con)

    func_pattern = bids_config['default_path_patterns'][2]
    deriv_pattern = func_pattern.replace(
        "[_echo-{echo}]", "[_echo-{echo}][_space-{space}][_desc-{desc}]"
        ).replace(
            "{suffix<", "{suffix<mask|regressors|"
        ).replace(
            "{extension<", "{extension<.tsv|"
        )

    func_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels, deriv_pattern,
        extra_params={"suffix": "bold", "space": "T1w", "desc": "preproc", "extension": ".nii.gz"}
    )

    meta_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels, deriv_pattern,
        extra_params={"suffix": "bold", "space": "T1w", "desc": "preproc", "extension": ".json"}
    )

    mask_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels, deriv_pattern,
        extra_params={"suffix": "mask", "space": "T1w", "desc": "brain", "extension": ".nii.gz"}
    )

    regressor_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels, deriv_pattern,
        extra_params={"desc": "confounds", "suffix": "regressors", "extension": ".tsv"}
    )

    return func_filepaths, meta_filepaths, mask_filepaths, regressor_filepaths


def create_event_filepaths(base_dir, participant_labels, session_labels, task_labels, run_labels):
    with open(bids.config.get_option('config_paths')['bids']) as b_con:
        bids_config = json.load(b_con)

    events_pattern = bids_config['default_path_patterns'][6]

    events_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels,
        events_pattern, extra_params={"suffix": "events", "datatype": "func", "extension": ".tsv"}
    )

    return events_filepaths


def create_func_filepaths(base_dir, participant_labels, session_labels, task_labels, run_labels):
    with open(bids.config.get_option('config_paths')['bids']) as b_con:
        bids_config = json.load(b_con)

    func_pattern = bids_config['default_path_patterns'][2]

    func_nii_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels,
        func_pattern, extra_params={"suffix": "bold", "extension": ".nii.gz"}
    )

    func_meta_filepaths = generate_filepaths(
        base_dir, participant_labels, session_labels, task_labels, run_labels,
        func_pattern, extra_params={"suffix": "bold", "extension": "json"}
    )

    return func_nii_filepaths, func_meta_filepaths


def generate_filepaths(
    base_dir, participant_labels, session_labels, task_labels,
    run_labels, file_patterns, extra_params=None,
):
    if not extra_params:
        extra_params = {}

    all_combos = product(
        participant_labels,
        session_labels or (None,),
        task_labels or (None,),
        run_labels or (None,),
    )
    param_order = ['subject', 'session', 'task', 'run']
    filepaths = []
    for combo in all_combos:
        file_params = {
            **{k: v for k, v in zip(param_order, combo)},
            **extra_params,
        }
        filepaths.append(base_dir / build_path(file_params, file_patterns))

    return filepaths


@pytest.fixture(scope='session')
def bids_dset(
    bids_dir, deriv_dir, participant_labels, session_labels, task_labels, run_labels,
    func_img, mask_img, func_metadata, events_df, noise
):
    events_df = events_df[0]
    label_args = (participant_labels, session_labels, task_labels, run_labels)

    func_nii_filepaths, func_meta_filepaths = create_func_filepaths(bids_dir, *label_args)

    for func_json in func_meta_filepaths:
        func_json.ensure()
        with open(func_json.strpath, "w") as jsn_file:
            json.dump(func_metadata, jsn_file)

    for func_nii in func_nii_filepaths:
        func_nii.dirpath().ensure_dir()
        func_img.to_filename(func_nii.strpath)

    events_filepaths = create_event_filepaths(bids_dir, *label_args)

    for event_fp in events_filepaths:
        event_fp.dirpath().ensure_dir()
        events_df.to_csv(event_fp, sep='\t', index=False)

    deriv_func_filepaths, deriv_meta_filepaths, deriv_mask_filepaths, deriv_regressor_filepaths = create_deriv_filepaths(
        deriv_dir, *label_args
    )

    noise_df = pd.DataFrame({"noise": noise})
    for d_func, d_meta, d_mask, d_regress in zip(
        deriv_func_filepaths, deriv_meta_filepaths, deriv_mask_filepaths, deriv_regressor_filepaths
    ):
        d_func.dirpath().ensure_dir()
        d_meta.ensure()
        with open(d_meta.strpath, "w") as jsn_file:
            json.dump(func_metadata, jsn_file)
        func_img.to_filename(d_func)
        mask_img.to_filename(d_mask)
        noise_df.to_csv(d_regress, sep="\t", index=False)

    database_path = bids_dir.dirpath() / 'dbcache'
    return bids.BIDSLayout(
        bids_dir, derivatives=True, database_path=database_path,
        reset_database=True)




def create_participant_files(
    participant_labels, func_img, mask_img, events_df, func_metadata, noise
):
    with open(bids.config.get_option('config_paths')['bids']) as b_con:
        bids_config = json.load(b_con)

    func_pattern = bids_config['default_path_patterns'][2]
    deriv_pattern = func_pattern.replace(
        "[_echo-{echo}]", "[_echo-{echo}][_space-{space}][_desc-{desc}]"
        ).replace(
            "{suffix<", "{suffix<mask|regressors|"
        ).replace(
            "{extension<", "{extension<.tsv|"
        )
    event_pattern = bids_config['default_path_patterns'][6]

    task_params = {
        "task": "eating",
    }
    for p_label in participant_labels:
        func_nii_file = build_path(
            {
                'subject': p_label,
                'suffix': 'bold',
                'extension': '.nii.gz',
            },
            func_pattern,
        )

        func_json_file = build_path(
            {
                'subject': p_label,
                'suffix': 'bold',
                'extension': 'json',
            },
            func_pattern,
        )

        func_deriv_nii
