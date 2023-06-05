# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module holds common utilities for managing arbitrary JSON serializable
dicts as protobuf Struct objects
"""
# Standard
from typing import Dict, List, Optional, Union

# Third Party
from google.protobuf import descriptor, message_factory, struct_pb2

# Type hints for JSON serializable dicts
JsonDictValue = Union[
    int,
    float,
    str,
    bool,
    type(None),
    List["JsonDictValue"],
    "JsonDict",
]
JsonDict = Dict[str, JsonDictValue]


def dict_to_struct(
    dictionary: JsonDict,
    struct_class: Optional[type] = None,
    value_class: Optional[type] = None,
    list_value_class: Optional[type] = None,
) -> struct_pb2.Struct:
    """Convert a python dict to a protobuf Struct"""
    if struct_class is None:
        struct_class = struct_pb2.Struct
        value_class = struct_pb2.Value
        list_value_class = struct_pb2.ListValue
    else:
        if value_class is None:
            value_class = _get_message_class(
                struct_class.DESCRIPTOR.file.pool.FindMessageTypeByName(
                    "google.protobuf.Value"
                )
            )
        if list_value_class is None:
            list_value_class = _get_message_class(
                struct_class.DESCRIPTOR.file.pool.FindMessageTypeByName(
                    "google.protobuf.ListValue"
                )
            )

    return struct_class(
        fields={
            key: _value_to_struct_value(
                value,
                struct_class=struct_class,
                value_class=value_class,
                list_value_class=list_value_class,
            )
            for key, value in dictionary.items()
        }
    )


def struct_to_dict(struct: struct_pb2.Struct) -> JsonDict:
    """Convert a struct into the equivalent json dict"""
    return {key: _struct_value_to_py(val) for key, val in struct.fields.items()}


## Implementation Details ######################################################


def _value_to_struct_value(value, struct_class, value_class, list_value_class):
    """Recursive helper to convert python values to struct values"""
    if value is None:
        struct_value = value_class(null_value=struct_pb2.NullValue.NULL_VALUE)
    elif isinstance(value, dict):
        struct_value = value_class(
            struct_value=dict_to_struct(
                value,
                struct_class=struct_class,
                value_class=value_class,
                list_value_class=list_value_class,
            )
        )
    elif isinstance(value, list):
        struct_value = value_class(
            list_value=list_value_class(
                values=(
                    _value_to_struct_value(
                        item,
                        struct_class=struct_class,
                        value_class=value_class,
                        list_value_class=list_value_class,
                    )
                    for item in value
                )
            )
        )
    elif isinstance(value, bool):
        struct_value = value_class(bool_value=value)
    elif isinstance(value, int):
        struct_value = value_class(number_value=value)
    elif isinstance(value, float):
        struct_value = value_class(number_value=value)
    elif isinstance(value, str):
        struct_value = value_class(string_value=value)
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")
    return struct_value


def _struct_value_to_py(struct_value: struct_pb2.Value) -> JsonDictValue:
    """Recursive helper to convert struct values to python values"""
    which = struct_value.WhichOneof("kind")
    if which in [None, "null_value"]:
        return None
    if which == "number_value":
        val = struct_value.number_value
        if int(val) == val:
            return int(val)
        return val
    if which in ["string_value", "bool_value"]:
        return getattr(struct_value, which)
    if which == "struct_value":
        return struct_to_dict(struct_value.struct_value)
    if which == "list_value":
        return [_struct_value_to_py(item) for item in struct_value.list_value.values]


def _get_message_class(
    desc: descriptor.Descriptor,
) -> message_factory.message.Message:
    """Helper to get the concrete protobuf class from a descriptor. This
    supports compatibility between protobuf 3.X and 4.X
    """
    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(desc)  # pragma: no cover
    return message_factory.MessageFactory().GetPrototype(desc)  # pragma: no cover
