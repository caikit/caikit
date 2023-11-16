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

# Standard
from typing import List, Optional, Union

# Local
from caikit.core import DataObjectBase, dataobject
from caikit.interfaces.common.data_model import File
from caikit.runtime.http_server.pydantic_wrapper import dataobject_to_pydantic
from caikit.runtime.http_server.utils import (
    convert_json_schema_to_multipart,
    flatten_json_schema,
    update_dict_at_dot_path,
)

## Setup #########################################################################


@dataobject(package="caikit_data_model.test_utils")
class ComplexUtilBaseClass(DataObjectBase):
    bytes_type: bytes
    file_type: Optional[File]
    metadata: Union[int, str]
    list_file_type: List[File]


@dataobject(package="caikit_data_model.test_utils")
class ComplexUtilHttpServerInputs(DataObjectBase):
    inputs: ComplexUtilBaseClass


def _recursively_assert_no_refs(obj):
    if isinstance(obj, dict):
        assert "$ref" not in obj.keys()
        [_recursively_assert_no_refs(val) for val in obj.values()]
    elif isinstance(obj, list):
        [_recursively_assert_no_refs(val) for val in obj]


## Tests ########################################################################


### convert_json_schema_to_multipart #############################################################


def test_convert_json_schema_to_multipart():
    pydantic_model = dataobject_to_pydantic(ComplexUtilHttpServerInputs)
    parsed_schema = flatten_json_schema(pydantic_model.model_json_schema())
    converted_schema = convert_json_schema_to_multipart(parsed_schema)
    # Make sure the converted schema has the properly extracted fields
    assert "inputs" in converted_schema["properties"].keys()
    assert "inputs.bytes_type" in converted_schema["properties"].keys()
    assert "inputs.file_type" in converted_schema["properties"].keys()
    assert "inputs.list_file_type" in converted_schema["properties"].keys()
    assert converted_schema["properties"]["inputs.list_file_type"]["type"] == "array"
    _recursively_assert_no_refs(converted_schema)


### flatten_json_schema #############################################################


def test_flatten_json_schema():
    """If an invalid module is provided to caikit lib setup, it throws a ValueError"""
    pydantic_model = dataobject_to_pydantic(ComplexUtilHttpServerInputs)
    flattened_schema = flatten_json_schema(pydantic_model.model_json_schema())
    # Assert there are no longer defs in schema
    assert "$defs" not in flattened_schema
    _recursively_assert_no_refs(flattened_schema)


### update_dict_at_dot_path #############################################################


def test_update_dict_at_dot_path():
    source_object = {"nondict": 1}

    # Assert function can update dict
    assert update_dict_at_dot_path(source_object, "test.path", "value")
    assert source_object["test"]["path"] == "value"

    assert not update_dict_at_dot_path(source_object, "nondict.path", "value")
    assert source_object["nondict"] == 1
