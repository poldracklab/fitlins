.. include:: /links.rst
.. include:: /../examples/models/README.rst

These models may be browsed in the examples_ directory on GitHub.

Word vs Pseudoword Contrast
---------------------------
Dataset: https://openneuro.org/datasets/ds000003/versions/00001

This model is translated from `model001
<https://legacy.openfmri.org/s3-browser/?prefix=ds000003/ds000003_R1.1.0/uncompressed/models/model001/>`_
in the original OpenFMRI dataset.

It demonstrates using the ``Factor`` transform to turn the ``trial_type``
column into a column for each trial type (*i.e.*, the ``trial_type.word``
column has ``1`` where ``trial_type`` was ``word``, ``0`` elsewhere, and so
on), as well as convolution.

The ``Model.X`` section demonstrates selection of regressors for the design
matrix, and ``Contrasts`` shows how to perform a simple contrast between
two conditions.

At the ``dataset`` level, the ``AutoContrasts`` option demonstrates taking
a simple mean at the group level.

.. raw:: html

    <details>
    <summary><code>ds000003/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000003/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>

Balloon Analog Risk Task
------------------------
Dataset: https://openneuro.org/datasets/ds000030/versions/00016

The balloon analog risk task (BART) is a risk-taking game where participants
decide whether to inflate a balloon, risking explosion, or cash out.
There are two trial types (``BALOON`` [*sic*] and ``CONTROL``), and three
possible actions (``ACCEPT``, ``CASHOUT``, ``EXPLODE``).

In this model, we contrast responses to ``ACCEPT`` and ``EXPLODE`` actions
in ``BALOON`` trials only.

This model is similar to the word-pseudoword model above, but also demonstrates
the use of the ``And`` transformation, that takes the logical and of two binary
(``0``/``1``) columns and assigns a new name to the result.

.. raw:: html

    <details>
    <summary><code>ds000030/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000030/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>

DS000114 Model
--------------

Dataset: `doi:10.18112/openneuro.ds000114.v1.0.1 <https://doi.org/10.18112/openneuro.ds000114.v1.0.1>`_

This model was written to demonstrate a model that specifies all levels of
analysis.

The ``finger_foot_lips`` task is a block-design motor task with interleaved
blocks of finger-tapping, foot-twitching and lip-pursing.

The ``Factor`` and ``Convolve`` transforms will be familiar from the above
models.
The contrast, however, shows a three-way contrast, testing for greater response
to finger than foot or lip actions.
Note that the negative values sum to ``-1`` and the positive to ``1``.

At the ``session`` level, no contrast is performed; rather the ``finger_vs_other``
contrasts are split across sessions, to avoid grouping them at the subject level.

The contrast at the subject level is a simple ``test - retest`` contrast, and
finally the dataset level again takes a simple mean across subjects.

.. note::

   This model can be run by FitLins, but it has a second-level contrast that
   Nistats_ cannot currently handle, so all group level stats will be ``NaN``.


.. raw:: html

    <details>
    <summary><code>ds000114/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000114/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>


DS000117 Model
--------------

Dataset: `doi:10.18112/openneuro.ds000117.v1.0.3 <https://doi.org/10.18112/openneuro.ds000117.v1.0.3>`_

This model is translated from `model001
<https://legacy.openfmri.org/s3-browser/?prefix=ds000117/ds000117_R0.1.1/uncompressed/models/model001/>`_
in the original OpenFMRI dataset.

This model is another basic contrast, mostly interesting because there are
several runs per subject to be averaged over before taking the group average.

FitLins does not currently support fixed effects models, but this will be
updated as we decide how to indicate that an analysis level should be a fixed
or random effects combination.

It also demonstrates the use of the logical ``Or`` transformation.


.. raw:: html

    <details>
    <summary><code>ds000117/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000117/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>


.. _///openfmri: http://datasets.datalad.org/?dir=/openfmri
.. _examples: https://github.com/poldracklab/fitlins/tree/master/examples
