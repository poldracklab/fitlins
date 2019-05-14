.. include:: links.rst

BIDS Stats Models
=================

FitLins is a consumer of model specifications written according to the
`BIDS Stats Models`_ draft standard.
A model is a `JSON`_ document of the following layout:

.. code-block:: yaml

    {
      "Name": "two_condition",
      "Description": "A simple, two-condition contrast",
      "Input": {
        "task": "motor"
        },
      "Steps": [
        {
          "Level": "Run",
          ...
        },
        {
          "Level": "Session",
          ...
        },
        {
          "Level": "Subject",
          ...
        },
        {
          "Level": "Dataset",
          ...
        }
      ]
    }

The optional ``Input`` section is a series of *selectors*, or BIDS key/value pairs
that describe the BOLD files that are required for the model.
The heart of the model is the ``Steps`` section, which correspond to levels of analysis,
and can be specified at any BIDS-App level, *i.e.*, ``Run``, ``Session``, ``Subject``
or ``Dataset``.

The first step (typically ``Run``) implicitly ingests BOLD series as input images, along
with associated variables:

 * `Task events`_ with onsets and durations, defined in ``events.tsv`` files
 * `Physiological recordings`_ or stimuli taken during the scan, defined in
   ``physio.tsv.gz`` and ``stim.tsv.gz`` files, respectively
 * `Time series`_ with one data point per volume, such as confound regressors
   found in ``desc-confounds_regressors.tsv``

These variables can be transformed and combined into a design matrix.
For example, supposing you have an ``events.tsv`` with a ``trial_type`` column
that can take the values ``A`` and ``B``, and you want to contrast ``A - B`` with
24-parameter motion confounds:

.. code-block:: yaml

    {
      "Level": "Run",
      "Transformations": [
        {
          "Name": "Factor",
          "Inputs": ["trial_type"]
        },
        {
          "Name": "Convolve",
          "Model": "spm"
          "Inputs": ["trial_type.A", "trial_type.B"]
        },
        {
          "Name": "Lag",
          "Inputs": ["rot_x", "rot_y", "rot_z",
                     "trans_x", "trans_y", "trans_z"],
          "Outputs": ["d_rot_x", "d_rot_y", "d_rot_z",
                      "d_trans_x", "d_trans_y", "d_trans_z"]
        },
        {
          "Name": "Power",
          "Order": 2,
          "Inputs": ["rot_x", "rot_y", "rot_z",
                     "trans_x", "trans_y", "trans_z",
                     "d_rot_x", "d_rot_y", "d_rot_z",
                     "d_trans_x", "d_trans_y", "d_trans_z"],
          "Outputs": ["rot_x_2", "rot_y_2", "rot_z_2",
                      "trans_x_2", "trans_y_2", "trans_z_2",
                      "d_rot_x_2", "d_rot_y_2", "d_rot_z_2",
                      "d_trans_x_2", "d_trans_y_2", "d_trans_z_2"]
        }
      ],
      "X": [
        "trial_type.A", "trial_type.B",
        "rot_x", "rot_y", "rot_z",
        "trans_x", "trans_y", "trans_z",
        "d_rot_x", "d_rot_y", "d_rot_z",
        "d_trans_x", "d_trans_y", "d_trans_z",
        "rot_x_2", "rot_y_2", "rot_z_2",
        "trans_x_2", "trans_y_2", "trans_z_2",
        "d_rot_x_2", "d_rot_y_2", "d_rot_z_2",
        "d_trans_x_2", "d_trans_y_2", "d_trans_z_2"],
      "Contrasts": [
        {
          "Name": "a_vs_b",
          "ConditionList": ["trial_type.A", "trial_type.B"],
          "Type": "t",
          "Weights": [1, -1]
        }
      ]
    }

``X`` refers to the design matrix (in the sense of $\mathbf Y = \mathbf{XB} + \epsilon$),
and should include your *variables of interest* and your *nuisance regressors*.
While all variables found in your BIDS dataset will be available, only those explicitly
listed will be fit.
The output of this level will be statistical maps for the contrast ``a_vs_b``, which
will now be available to the next step.

For this contrast, we may want to simply run a basic $t$-test at the group level.
In this case, we can use an ``AutoContrast`` to make a very simple final step:

.. code-block:: yaml

    {
      "Level": "Dataset",
      "AutoContrasts": ["a_vs_b"]
    }

This is equivalent to the more verbose:

.. code-block:: yaml

    {
      "Level": "Dataset",
      "Contrasts": [
        {
          "Name": "a_vs_b",
          "ConditionList": ["a_vs_b"],
          "Type": "t",
          "Weights": [1]
        }
      ]
    }

The output of this level will again be a statistical map for the contrast ``a_vs_b``,
but summarized across the whole group.

.. _`Task events`: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
.. _`Physiological recordings`: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continous-recordings.html
.. _`Time series`: https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/04-functional-derivatives.html#time-series
