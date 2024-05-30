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
"""
Tests for the pydantic wrapping for REST server
"""
# Standard
from typing import Dict, List, Union, get_args
import datetime
import enum

# Third Party
from fastapi.datastructures import FormData
import numpy as np
import pydantic
import pytest

# First Party
from py_to_proto.dataclass_to_proto import Annotated

# Local
from caikit.core import DataObjectBase, dataobject
from caikit.core.data_model.base import DataBase
from caikit.interfaces.common.data_model import File, FileReference
from caikit.interfaces.nlp.data_model.text_generation import (
    GeneratedTextStreamResult,
    GeneratedToken,
)
from caikit.runtime.http_server.pydantic_wrapper import (
    PYDANTIC_TO_DM_MAPPING,
    _from_base64,
    _get_pydantic_type,
    _parse_form_data_to_pydantic,
    dataobject_to_pydantic,
    pydantic_from_request,
    pydantic_to_dataobject,
)
from caikit.runtime.service_generation.data_stream_source import make_data_stream_source
from sample_lib.data_model.sample import (
    SampleInputType,
    SampleListInputType,
    SampleOutputType,
    SampleTrainingType,
)


def test_pydantic_to_dataobject_simple():
    """Test building our simple DM objects through pydantic objects"""
    # get our DM class
    sample_input_dm_class = DataBase.get_class_for_name("SampleInputType")
    # Create pydantic model for our DM class
    sample_input_pydantic_model = dataobject_to_pydantic(sample_input_dm_class)
    # build our DM object using a pydantic object
    sample_input_dm_obj = pydantic_to_dataobject(
        sample_input_pydantic_model(name="Hello world")
    )

    # assert it's our DM object, all fine and dandy
    assert isinstance(sample_input_dm_obj, DataBase)
    assert sample_input_dm_obj.to_json() == '{"name": "Hello world"}'


def test_pydantic_to_dataobject_documentation():
    """Test building a simple pydantic object retains the documentation information"""
    # get our DM class
    sample_input_dm_class = DataBase.get_class_for_name("SampleListInputType")
    # Create pydantic model for our DM class
    sample_input_pydantic_model = dataobject_to_pydantic(sample_input_dm_class)
    # Create openapi json from pydantic model to test descriptions
    json_schema = sample_input_pydantic_model.model_json_schema()

    # assert it's our DM object, all fine and dandy
    assert json_schema.get("description") == SampleListInputType.__doc__


def test_pydantic_to_dataobject_datastream_jsondata():
    """Test building our datastream DM objects through pydantic objects"""

    make_data_stream_source(SampleTrainingType)
    # get our DM class
    datastream_dm_class = DataBase.get_class_for_name(
        "DataStreamSourceSampleTrainingType"
    )
    # create pydantic model for our DM class
    datastream_pydantic_model = dataobject_to_pydantic(datastream_dm_class)
    # build our DM Datastream JsonData object using a pydantic object
    datastream_dm_obj = pydantic_to_dataobject(
        datastream_pydantic_model(
            data_stream={
                "data": [{"number": 1, "label": "foo"}, {"number": 2, "label": "bar"}]
            }
        )
    )

    # assert it's our DM object, all fine and dandy
    assert isinstance(datastream_dm_obj, DataBase)
    assert (
        datastream_dm_obj.to_json()
        == '{"jsondata": {"data": [{"number": 1, "label": "foo"}, {"number": 2, "label": "bar"}]}}'
    )


def test_pydantic_to_dataobject_datastream_file():
    make_data_stream_source(SampleTrainingType)
    # get our DM class
    datastream_dm_class = DataBase.get_class_for_name(
        "DataStreamSourceSampleTrainingType"
    )
    # create pydantic model for our DM class
    datastream_pydantic_model = dataobject_to_pydantic(datastream_dm_class)

    # build our DM Datastream File object using a pydantic object
    datastream_dm_obj = pydantic_to_dataobject(
        datastream_pydantic_model(data_stream={"filename": "hello"})
    )

    # assert it's our DM object, all fine and dandy
    assert isinstance(datastream_dm_obj, DataBase)
    assert isinstance(datastream_dm_obj.data_stream, FileReference)
    assert datastream_dm_obj.to_json() == '{"file": {"filename": "hello"}}'


@pytest.mark.parametrize(
    "input, output",
    [
        (np.integer, int),
        (np.floating, float),
        (int, int),
        (float, float),
        (bool, bool),
        (str, str),
        (
            bytes,
            Annotated[
                bytes, pydantic.functional_validators.BeforeValidator(_from_base64)
            ],
        ),
        (type(None), type(None)),
        (enum.Enum, enum.Enum),
        (Annotated[str, "blah"], str),
        (Union[str, int], Union[str, int]),
        (List[str], List[str]),
        (List[Annotated[str, "blah"]], List[str]),
        (Dict[str, int], Dict[str, int]),
        (Dict[Annotated[str, "blah"], int], Dict[str, int]),
        (datetime.datetime, datetime.datetime),
        (datetime.date, datetime.date),
        (datetime.time, datetime.time),
        (datetime.timedelta, datetime.timedelta),
    ],
)
def test_get_pydantic_type(input, output):
    assert _get_pydantic_type(input) == output


def test_get_pydantic_type_union():
    union_type = Union[SampleInputType, SampleOutputType]
    return_type = _get_pydantic_type(union_type)
    assert all(
        issubclass(ret_type, pydantic.BaseModel) for ret_type in get_args(return_type)
    )


def test_get_pydantic_type_DM():
    # DM case
    sample_input_dm_class = DataBase.get_class_for_name("SampleInputType")
    sample_input_pydantic_model = _get_pydantic_type(sample_input_dm_class)

    assert issubclass(sample_input_pydantic_model, pydantic.BaseModel)
    assert sample_input_pydantic_model in PYDANTIC_TO_DM_MAPPING
    assert sample_input_pydantic_model is PYDANTIC_TO_DM_MAPPING.get(
        sample_input_dm_class
    )


def test_get_pydantic_type_throws_random_type():
    # error case
    with pytest.raises(TypeError):
        _get_pydantic_type("some_random_type")


def test_pydantic_wrapping_with_enums():
    """Check that the pydantic wrapping works on our data models when they have enums"""
    # The NLP GeneratedTextStreamResult data model contains enums

    # Check that our data model is fine and dandy
    token = GeneratedToken(text="foo")
    assert token.text == "foo"

    # Wrap the containing data model in pydantic
    dataobject_to_pydantic(GeneratedTextStreamResult)

    # Check that our data model is _still_ fine and dandy
    token = GeneratedToken(text="foo")
    assert token.text == "foo"


def test_pydantic_wrapping_with_lists():
    """Check that pydantic wrapping works on data models with lists"""

    @dataobject(package="http")
    class BarTest(DataObjectBase):
        baz: int

    @dataobject(package="http")
    class FooTest(DataObjectBase):
        bars: List[BarTest]

    foo = FooTest(bars=[BarTest(1)])
    assert foo.bars[0].baz == 1

    dataobject_to_pydantic(FooTest)

    foo = FooTest(bars=[BarTest(1)])
    assert foo.bars[0].baz == 1


def test_dataobject_to_pydantic_simple_DM():
    """Test that we can create a pydantic model from a simple DM"""
    sample_input_dm_class = DataBase.get_class_for_name("SampleInputType")
    sample_input_pydantic_model = dataobject_to_pydantic(sample_input_dm_class)
    model_instance = sample_input_pydantic_model.model_validate_json(
        '{"name": "world"}'
    )
    assert {"name": str} == sample_input_pydantic_model.__annotations__
    assert issubclass(sample_input_pydantic_model, pydantic.BaseModel)
    assert sample_input_pydantic_model in PYDANTIC_TO_DM_MAPPING
    assert model_instance.name == "world"


def test_dataobject_to_pydantic_simple_DM_extra_forbidden_throws():
    """Test that if we forbid extra values, then we raise if we pass in extra values"""
    sample_input_dm_class = DataBase.get_class_for_name("SampleInputType")

    sample_input_pydantic_model = dataobject_to_pydantic(sample_input_dm_class)
    # sample_input_pydantic_model_extra_forbidden doesn't allow anything extra
    with pytest.raises(pydantic.ValidationError) as e1:
        sample_input_pydantic_model.model_validate_json('{"blah": "world"}')
    assert "Extra inputs are not permitted" in e1.value.errors()[0]["msg"]

    with pytest.raises(pydantic.ValidationError) as e2:
        sample_input_pydantic_model.model_validate_json(
            '{"name": "world", "blah": "world"}'
        )
    assert "Extra inputs are not permitted" in e2.value.errors()[0]["msg"]


def test_dataobject_to_pydantic_oneof():
    """Test that we can create a pydantic model from a DM with a Union"""
    make_data_stream_source(SampleTrainingType)
    sample_input_dm_class = DataBase.get_class_for_name(
        "DataStreamSourceSampleTrainingType"
    )
    data_stream_source_pydantic_model = dataobject_to_pydantic(sample_input_dm_class)

    assert issubclass(data_stream_source_pydantic_model, pydantic.BaseModel)
    assert {
        "data_stream": Union[
            PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.JsonData),
            PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.FileReference),
            PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.ListOfFileReferences),
            PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.Directory),
            PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.S3Files),
        ]
    } == data_stream_source_pydantic_model.__annotations__
    assert data_stream_source_pydantic_model in PYDANTIC_TO_DM_MAPPING

    assert issubclass(
        PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.JsonData),
        type(
            data_stream_source_pydantic_model.model_validate_json(
                '{"data_stream": {"data": [{"number": 1}]}}'
            ).data_stream
        ),
    )
    assert issubclass(
        PYDANTIC_TO_DM_MAPPING.get(sample_input_dm_class.FileReference),
        type(
            data_stream_source_pydantic_model.model_validate_json(
                '{"data_stream": {"filename": "file1"}}'
            ).data_stream
        ),
    )


def test_parse_form_data_to_pydantic():
    file_input_dm_class = DataBase.get_class_for_name("FileInputType")
    pydantic_model = dataobject_to_pydantic(file_input_dm_class)

    form = FormData({"file.data": b"raw_bytes_data", "metadata": '{"name":"test"}'})

    pydantic_instance = _parse_form_data_to_pydantic(pydantic_model, form)
    assert pydantic_instance.file.data == b"raw_bytes_data"
    assert pydantic_instance.metadata.name == "test"


def test_parse_form_data_to_pydantic_sub_field():
    file_input_dm_class = DataBase.get_class_for_name("FileInputType")
    pydantic_model = dataobject_to_pydantic(file_input_dm_class)

    form = FormData({"file.data": b"raw_bytes_data", "metadata.name": "test"})

    pydantic_instance = _parse_form_data_to_pydantic(pydantic_model, form)
    assert pydantic_instance.file.data == b"raw_bytes_data"
    assert pydantic_instance.metadata.name == "test"


def test_parse_form_data_to_pydantic_list():
    sample_list_input_dm_class = DataBase.get_class_for_name("SampleListInputType")
    pydantic_model = dataobject_to_pydantic(sample_list_input_dm_class)

    form = FormData(
        [
            ("inputs", '{"name":"testname"}'),
            ("inputs", '{"name":"anothername"}'),
        ]
    )

    pydantic_instance = _parse_form_data_to_pydantic(pydantic_model, form)
    assert isinstance(pydantic_instance.inputs, list)
    assert pydantic_instance.inputs[0].name == "testname"
    assert pydantic_instance.inputs[1].name == "anothername"
