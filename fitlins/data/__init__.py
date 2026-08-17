'''Data package

.. autofunction:: load_resource

.. automethod:: load_resource.readable

.. automethod:: load_resource.as_path

.. automethod:: load_resource.cached
'''

from acres import Loader

load_resource = Loader(__spec__.name)
