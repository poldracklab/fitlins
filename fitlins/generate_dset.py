import json
from itertools import product
from tempfile import mkdtemp
from pathlib import Path

import pandas as pd
import numpy as np
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
import nibabel as nib
import bids
from bids.layout.writing import build_path


def write_metadata(filepath, metadata):
    filepath.ensure()
    with open(str(filepath), 'w') as meta_file:
        json.dump(metadata, meta_file)


class RegressorFileCreator():
    """Generator for _regressors files in bids derivatives dataset"""

    # pattern for file
    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}][_space-{space}][_cohort-{cohort}][_desc-{desc}]_"
        "{suffix<timeseries|regressors>|timeseries}{extension<.json|.tsv>|.tsv}"
    )

    # common file parameters
    FILE_PARAMS = {"suffix": "regressors", "datatype": "func", "desc": "confounds"}

    def __init__(self, base_dir, fname_params, regr_names, n_tp, metadata=None):
        self.base_dir = base_dir
        self.metadata = metadata
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".tsv"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}

        self.init_data(regr_names, n_tp)
        self.create_fname(fname_params, meta_params)

    def init_data(self, regr_names, n_tp):
        """create the regressor data"""
        self.noise_df = pd.DataFrame(
            {name: np.random.random(n_tp) for name in regr_names}
        )

    def create_fname(self, fname_params, meta_params):
        """create the bids derivatives regressor file path and path names"""
        self.fname = self.base_dir / build_path(fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(meta_params, self.PATTERN)

    def write_file(self):
        """write the data to files"""
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
    FILE_PARAMS = {"suffix": "bold", "space": "T1w", "desc": "preproc"}

    def __init__(
        self, base_dir, fname_params, events_df, trial_type_weights, noise_df, n_tp, cnr, metadata
    ):
        self.base_dir = base_dir
        self.metadata = metadata
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".nii.gz"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.init_data(events_df, trial_type_weights, noise_df, n_tp, cnr, metadata)
        self.create_fname(fname_params, meta_params)

    def _create_signal(self, tr, n_tp, events_df, trial_type_weights):
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

    def _aggregate_noise(self, noise_df):
        return noise_df.values.mean(axis=1)

    def _create_nii(self, timeseries):
        brain_data = np.random.random((9, 9, 9, len(timeseries)))
        brain_data[2:6, 2:6, 2:6, :] += timeseries
        return nib.Nifti1Image(brain_data, affine=np.eye(4))

    def init_data(self, events_df, trial_type_weights, noise_df, n_tp, cnr, metadata):
        tr = metadata['RepetitionTime']
        signal = self._create_signal(tr, n_tp, events_df, trial_type_weights)
        noise = self._aggregate_noise(noise_df)
        contrast = signal.max()
        signal_scaling_factor = contrast * cnr * noise.std()
        timeseries = (signal * signal_scaling_factor) + noise
        scaled_timeseries = (timeseries * 10) + 100
        self.data = self._create_nii(scaled_timeseries)

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


class DerivMaskFileCreator():

    PATTERN = (
        "sub-{subject}[/ses-{session}]/{datatype<func>|func}/"
        "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}]"
        "[_ce-{ceagent}][_dir-{direction}][_rec-{reconstruction}][_run-{run}]"
        "[_echo-{echo}][_space-{space}][_cohort-{cohort}][_res-{resolution}]"
        "_desc-{desc}_{suffix<mask>|mask}{extension<.nii|.nii.gz|.json>|.nii.gz}"
    )
    FILE_PARAMS = {"suffix": "mask", "desc": "brain", "space": "T1w"}

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
        mask_data = (func_img.get_fdata()[:, :, :, 0] > 10).astype(np.int32)
        self.data = nib.Nifti1Image(mask_data, np.eye(4))

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
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".nii.gz"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}
        self.init_data(events_df, trial_type_weights, noise_df, n_tp, cnr, metadata)
        self.create_fname(fname_params, meta_params)

    def _create_signal(self, tr, n_tp, events_df, trial_type_weights):
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

    def _aggregate_noise(self, noise_df):
        return noise_df.values.mean(axis=1)

    def _create_nii(self, timeseries):
        brain_data = np.random.random((9, 9, 9, len(timeseries)))
        brain_data[2:6, 2:6, 2:6, :] += timeseries
        return nib.Nifti1Image(brain_data, affine=np.eye(4))

    def init_data(self, events_df, trial_type_weights, noise_df, n_tp, cnr, metadata):
        tr = metadata['RepetitionTime']
        signal = self._create_signal(tr, n_tp, events_df, trial_type_weights)
        noise = self._aggregate_noise(noise_df)
        contrast = signal.max()
        signal_scaling_factor = contrast * cnr * noise.std()
        timeseries = (signal * signal_scaling_factor) + noise
        scaled_timeseries = (timeseries * 10) + 100
        self.data = self._create_nii(scaled_timeseries)

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
        fname_params = {**fname_params, **self.FILE_PARAMS, "extension": ".tsv"}
        meta_params = {**fname_params, **self.FILE_PARAMS, "extension": ".json"}

        self.init_data(n_events, trial_types, event_duration, inter_trial_interval)
        self.create_fname(fname_params, meta_params)

    def init_data(self, n_events, trial_types, event_duration, inter_trial_interval):
        events_dict = {}
        n_trial_types = len(trial_types)
        experiment_duration = int(n_trial_types * n_events * inter_trial_interval)
        events_dict['onset'] = np.arange(0, experiment_duration, inter_trial_interval)
        events_dict['trial_type'] = trial_types * n_events
        events_dict['duration'] = [event_duration] * n_trial_types * n_events
        self.experiment_duration = experiment_duration
        self.events_df = pd.DataFrame(events_dict)

    def create_fname(self, fname_params, meta_params):
        self.fname = self.base_dir / build_path(fname_params, self.PATTERN)
        self.meta_fname = self.base_dir / build_path(meta_params, self.PATTERN)

    def write_file(self):
        self.fname.dirpath().ensure_dir()
        self.events_df.to_csv(self.fname, sep='\t', index=False)
        if self.metadata:
            write_metadata(self.meta_fname, self.metadata)
            return self.fname, self.meta_fname
        return self.fname


class DummyDerivatives():
    """Create a minimal BIDS+Derivatives dataset for testing"""

    DERIVATIVES_DICT = {
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
        database_path=None,
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
        self.database_path = database_path or self.base_dir.dirpath() / 'dbcache'
        self.participant_labels = participant_labels or ["bert", "ernie", "gritty"]
        self.session_labels = session_labels or ["breakfast", "lunch"]
        self.task_labels = task_labels or ["eating"]
        self.run_labels = run_labels or ["01", "02"]
        self.trial_types = trial_types or ["ice_cream", "cake"]
        self.trial_type_weights = trial_type_weights or list(range(1, len(self.trial_types)))
        self.n_events = n_events or 15
        self.event_duration = event_duration or 1
        self.inter_trial_interval = inter_trial_interval or 20
        self.cnr = cnr or 2
        self.regr_names = regr_names or ["food_sweats", "sugar_jitters"]
        self.func_metadata = func_metadata or {"RepetitionTime": 2.0, "SkullStripped": False}
        self.deriv_dir = self.base_dir.ensure('derivatives', 'fmriprep', dir=True)

        self.create_dataset_descriptions()
        self.write_bids_derivatives_dataset()
        self.create_layout()

    def create_dataset_descriptions(self):
        # dataset_description.json files are needed in both bids and derivatives
        bids_dataset_json = self.base_dir.ensure("dataset_description.json")
        with open(str(bids_dataset_json), 'w') as dj:
            json.dump(self.BIDS_DICT, dj)

        deriv_dataset_json = self.deriv_dir.ensure("dataset_description.json")

        with open(str(deriv_dataset_json), 'w') as dj:
            json.dump(self.DERIVATIVES_DICT, dj)

    def write_bids_derivatives_dataset(self):
        # generate all combinations of relevant file parameters
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
            noise = RegressorFileCreator(self.deriv_dir, file_params, self.regr_names, n_tp)
            noise.write_file()
            # create bids func file
            FuncFileCreator(
                self.base_dir, file_params, events.events_df,
                self.trial_type_weights, noise.noise_df, n_tp,
                self.cnr, self.func_metadata,
            ).write_file()
            # create deriv func file
            deriv_func = DerivFuncFileCreator(
                self.deriv_dir, file_params, events.events_df,
                self.trial_type_weights, noise.noise_df, n_tp,
                self.cnr, self.func_metadata,
            )
            deriv_func.write_file()
            # create mask for deriv_func
            DerivMaskFileCreator(self.deriv_dir, file_params, deriv_func.data).write_file()

    def create_layout(self):
        # create bids layout
        self.layout = bids.BIDSLayout(
            self.base_dir, derivatives=True, database_path=self.database_path,
            reset_database=True)
