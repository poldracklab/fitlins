from nipype.interfaces.io import IOBase, add_traits
from nipype.interfaces.base import SimpleInterface, DynamicTraitedSpec, TraitedSpec, traits


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


class CollateWithMetadataInputSpec(DynamicTraitedSpec):
    metadata = traits.List(traits.Dict)
    field_to_metadata_map = traits.Dict(traits.Str)


class CollateWithMetadataOutputSpec(TraitedSpec):
    metadata = traits.List(traits.Dict)
    out = traits.List(traits.Any)


class CollateWithMetadata(SimpleInterface):
    input_spec = CollateWithMetadataInputSpec
    output_spec = CollateWithMetadataOutputSpec

    def __init__(self, fields=None, **kwargs):
        super(CollateWithMetadata, self).__init__(**kwargs)
        if not fields:
            fields = self.inputs.field_to_metadata_map.keys()
            if not fields:
                raise ValueError("Fields must be a non-empty list")

        self._fields = fields
        add_traits(self.inputs, fields)

    def _run_interface(self, runtime):
        orig_metadata = self.inputs.metadata
        md_map = self.inputs.field_to_metadata_map
        n = len(orig_metadata)

        self._results.update({'metadata': [], 'out': []})
        for key in self._fields:
            val = getattr(self.inputs, key)
            if len(val) != n:
                raise ValueError(f"List lengths must match metadata. Failing list: {key}")
            for md, obj in zip(orig_metadata, val):
                metadata = md.copy()
                metadata.update(md_map.get(key, {}))
                self._results['metadata'].append(metadata)
                self._results['out'].append(obj)

        return runtime
