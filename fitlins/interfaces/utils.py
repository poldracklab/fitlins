from nipype.interfaces.io import IOBase, add_traits
from nipype.interfaces.base import DynamicTraitedSpec


class MergeAll(IOBase):
    input_spec = DynamicTraitedSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, fields=None):
        super(MergeAll, self).__init__()
        if not fields:
            raise ValueError("Fields must be a non-empty list")

        self._fields = fields
        add_traits(self.inputs, fields)

    def _add_output_traits(self, base):
        return add_traits(base, self._fields)

    def _list_outputs(self):
        outputs = self._outputs().get()
        lengths = None
        for key in self._fields:
            val = getattr(self.inputs, key)
            _lengths = list(map(len, val))
            if lengths is None:
                lengths = _lengths
            elif _lengths != lengths:
                raise ValueError("List lengths must be consistent across fields")
            outputs[key] = [elem for sublist in val for elem in sublist]

        return outputs
