FitLins - Fitting Linear Models to BIDS Datasets
================================================

FitLins is a tool for estimating linear models, defined by the `BIDS Model`_
specification proposal, to BIDS-formatted datasets.

This software is in alpha stage, and should be considered unstable.
Users are welcome to test the software, and open issues.

Contributors
============

[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/0)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/0)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/1)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/1)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/2)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/2)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/3)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/3)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/4)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/4)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/5)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/5)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/6)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/6)[![](https://sourcerer.io/fame/effigies/poldracklab/fitlins/images/7)](https://sourcerer.io/fame/effigies/poldracklab/fitlins/links/7)

Usage
=====

The CLI follows the `BIDS-Apps`_ convention:

Usage::

    fitlins <bids_root> <out_dir> <analysis_level> [--model <model_name>]

See the output of ``fitlins --help`` for all valid options::

    usage: fitlins [-h] [-v]
                   [--participant-label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                   [-m MODEL] [-p PREPROC_DIR]
                   [--derivative-label DERIVATIVE_LABEL]
                   [--space {MNI152NLin2009cAsym}] [--include INCLUDE]
                   [--exclude EXCLUDE] [--n-cpus N_CPUS] [--debug] [-w WORK_DIR]
                   bids_dir output_dir {run,session,participant,dataset}

    FitLins: Workflows for Fitting Linear models to fMRI

    positional arguments:
      bids_dir              the root folder of a BIDS valid dataset (sub-XXXXX folders should be
                            found at the top level in this folder).
      output_dir            the output path for the outcomes of preprocessing and visual reports
      {run,session,participant,dataset}
                            processing stage to be runa (see BIDS-Apps specification).

    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit

    Options for filtering BIDS queries:
      --participant-label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                            one or more participant identifiers (the sub- prefix can be removed)
      -m MODEL, --model MODEL
                            location of BIDS model description (default bids_dir/model.json)
      -p PREPROC_DIR, --preproc-dir PREPROC_DIR
                            location of preprocessed data (if relative path, search
                            bids_dir/derivatives, followed by output_dir)
      --derivative-label DERIVATIVE_LABEL
                            execution label to append to derivative directory name
      --space {MNI152NLin2009cAsym}
                            registered space of input datasets
      --include INCLUDE     regex pattern to include files
      --exclude EXCLUDE     regex pattern to exclude files

    Options to handle performance:
      --n-cpus N_CPUS       maximum number of threads across all processes
      --debug               run debug version of workflow

    Other options:
      -w WORK_DIR, --work-dir WORK_DIR
                            path where intermediate results should be stored

At present, FitLins does not support smoothing or operate in subject-native
space.
It is developed against `FMRIPREP`_-preprocessed datasets, but is intended to
work with any dataset following the `BIDS Derivatives`_ draft specification.

Models
------

By default, FitLins will look for a ``model.json`` in the root of the BIDS
directory.
A simple example model for `OpenFMRI dataset ds000030`_ is reproduced below::


	{
	    "name": "ds000030_bart",
	    "description": "model for balloon analog risk task",
	    "input": {
	        "task": "bart"
	    },
	    "blocks": [
	        {
	            "level": "run",
	            "transformations": [
	                {
	                    "name": "factor",
	                    "input": [
	                        "trial_type",
	                        "action"
	                    ]
	                },
	                {
	                    "name": "and",
	                    "input": [
	                        "trial_type.BALOON",
	                        "action.ACCEPT"
	                    ],
	                    "output": [
	                        "accept"
	                    ]
	                },
	                {
	                    "name": "and",
	                    "input": [
	                        "trial_type.BALOON",
	                        "action.EXPLODE"
	                    ],
	                    "output": [
	                        "explode"
	                    ]
	                }
	            ],
	            "model": {
	                "HRF_variables":[
	                    "accept",
	                    "explode"
	                ],
	                "variables": [
	                    "accept",
	                    "explode",
	                    "FramewiseDisplacement",
	                    "X",
	                    "Y",
	                    "Z",
	                    "RotX",
	                    "RotY",
	                    "RotZ"
	                ]
	            },
	            "contrasts": [
	                {
	                    "name": "accept_vs_explode",
	                    "condition_list": [
	                        "accept",
	                        "explode"
	                    ],
	                    "weights": [1, -1],
	                    "type": "T"
	                }
	            ]
	        },
	        {
	            "level": "dataset",
	            "model": {
	                "variables": [
	                    "accept_vs_explode"
	                ]
	            },
	            "contrasts": [
	                {
	                    "name": "group_accept_vs_explode",
	                    "condition_list":[
	                        "accept_vs_explode"
	                    ],
	                    "weights": [1],
	                    "type": "T"
	                }
	            ]
	        }
	    ]
	}

Additional examples can be found in the `models`_ branch of the main FitLins
repository.

.. note::

    The BIDS Model specification is a draft standard, and some details may
    change over time.

Warning
-------

FitLins is in Alpha-stage, and is not suitable for use as a library, as the
internal organization may change substantially without deprecation periods.
Similarly the outputs (or derivatives) are subject to change, as experience
and user feedback prompt.
The command-line interface outlined above should be fairly stable, however.

.. _`BIDS Model`: https://docs.google.com/document/d/1bq5eNDHTb6Nkx3WUiOBgKvLNnaa5OMcGtD0AZ9yms2M/
.. _`BIDS Derivatives`: https://docs.google.com/document/d/1Wwc4A6Mow4ZPPszDIWfCUCRNstn7d_zzaWPcfcHmgI4/
.. _BIDS-Apps: http://bids-apps.neuroimaging.io
.. _FMRIPREP: https://fmriprep.readthedocs.io
.. _`OpenFMRI dataset ds000030`: http://datasets.datalad.org/?dir=/openfmri/ds000030/
.. _models: https://github.com/poldracklab/fitlins/tree/models
