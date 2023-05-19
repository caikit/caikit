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
from typing import Dict, List, Union

# Third Party
from google.protobuf import struct_pb2  # import ListValue, NullValue, Struct, Value

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


def dict_to_struct(dictionary: JsonDict) -> struct_pb2.Struct:
    """Convert a python dict to a protobuf Struct"""
    return struct_pb2.Struct(
        fields={key: _value_to_struct_value(value) for key, value in dictionary.items()}
    )


def struct_to_dict(struct: struct_pb2.Struct) -> JsonDict:
    """Convert a struct into the equivalent json dict"""
    return {key: _struct_value_to_py(val) for key, val in struct.fields.items()}


## Implementation Details ######################################################


def _value_to_struct_value(value):
    """Recursive helper to convert python values to struct values"""
    if value is None:
        struct_value = struct_pb2.Value(null_value=struct_pb2.NullValue.NULL_VALUE)
    elif isinstance(value, dict):
        struct_value = struct_pb2.Value(struct_value=dict_to_struct(value))
    elif isinstance(value, list):
        struct_value = struct_pb2.Value(
            list_value=struct_pb2.ListValue(
                values=(_value_to_struct_value(item) for item in value)
            )
        )
    elif isinstance(value, bool):
        struct_value = struct_pb2.Value(bool_value=value)
    elif isinstance(value, int):
        struct_value = struct_pb2.Value(number_value=value)
    elif isinstance(value, float):
        struct_value = struct_pb2.Value(number_value=value)
    elif isinstance(value, str):
        struct_value = struct_pb2.Value(string_value=value)
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
