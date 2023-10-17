# ADR N022: Pluggable Runtime IO

Some users of `caikit` will require application-specific logic to fetch data from secure locations. These implementations will not be sufficiently generic to be maintained in the core of `caikit`, but for these security-minded applications, they are a must-have in order to satisfy security posture that fits the existing application.

## Decision

This proposal is to make the Input/Output mechanisms for `caikit.runtime` pluggable. Currently, this means updating `DataStreamSource` and the current `output_path` mechanism to use `oneof` semantics that can be extended by new implementations and bound at boot time with config options.

**Example Implementation**
```py
from caikit.core.data_model import DataBase, DataObjectBase, DataStream, dataobject
from caikit.runtime.service_generation.data_stream_source import DataStreamSourcePlugin, PluginFactory
from typing import List, Type

@dataobject
class NumericStreamInput(DataObjectBase):
    values: List[float]

@dataobject
class EncodedBytesStreamInput(DataObjectBase):
    value: bytes
    encoding: str

class MyDataStreamPlugin(DataStreamSourcePlugin):

    name = "MINE"

    def get_stream_message_type(self, element_type: type) -> Type[DataBase]:
        if element_type in [int, float]:
            return NumericStreamInput
        return EncodedBytesStreamInput

    def get_field_number(self) -> int:
        return 99 # Make this unique from all other plugins

    def to_data_stream(self, source_message: Type[DataBase], element_type: type) -> DataStream:
        if element_type in [int, float]:
            return DataStream.from_iterable(source_message.values).map(element_type)
        return DataStream.from_iterable(source_message.value.decode(source_message.encoding))


# Register the plugin with the factory so it can be configured
PluginFactory.register(MyDataStreamPlugin)
```

**Example Config**
```yaml
data_streams:
    source_plugins:
        inline:
            type: JsonData
        custom:
            type: MINE
```

## Status

choose one: Accepted

## Consequences

* New `config` section `data_streams.source_plugins` that will control which data stream source types are supported in the running `caikit` environment
* Equivalent section for `output_path` (naming TBD)
* An API breaking change to change `output_path` to something like `output_target` to be more generic and support a `oneof` for different output target representations
* New base classes and factories to register user-defined implementations of the plugins
