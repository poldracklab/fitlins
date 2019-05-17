.. include:: /links.rst
.. include:: /../examples/models/README.rst

These models may be browsed in the examples_ directory on GitHub.

DS000003 Model
--------------

This model is translated from `model001`_ in the original OpenFMRI dataset.

.. raw:: html

    <details>
    <summary><code>ds000003/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000003/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>

DS000030 Model
--------------

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
This model can be run by FitLins, but it has a second-level contrast that
Nistats_ cannot currently handle, so all group level stats will be `NaN`.


.. raw:: html

    <details>
    <summary><code>ds000114/models/model-001_smdl.json</code></summary>

.. literalinclude:: /../examples/models/ds000114/models/model-001_smdl.json
   :language: json
   :linenos:

.. raw:: html

    </details>


.. _model001: https://openfmri.org/s3-browser/?prefix=ds000003/ds000003_R1.1.0/uncompressed/models/model001/
.. _///openfmri: http://datasets.datalad.org/?dir=/openfmri
.. _examples: https://github.com/poldracklab/fitlins/tree/master/examples
