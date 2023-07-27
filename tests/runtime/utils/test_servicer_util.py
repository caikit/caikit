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
import os
import tempfile

# Third Party
import pytest

# Local
from caikit.runtime.protobufs import model_runtime_pb2
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import get_data_model
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    build_proto_response,
    get_metadata,
    is_protobuf_primitive_field,
    validate_caikit_library_class_exists,
    validate_caikit_library_class_method_exists,
    validate_data_model,
)
from sample_lib import SamplePrimitiveModule
from sample_lib.data_model import SampleInputType
from tests.conftest import random_test_id
from tests.fixtures import Fixtures
import caikit.core
import caikit.interfaces
import sample_lib


# ---------------- Tests for validate_caikit_library_class_exists --------------------
def test_servicer_util_validate_caikit_library_class_exists_returns_caikit_class():
    """Test that validate_caikit_library_class_exists returns a caikit class object"""
    validated_class = validate_caikit_library_class_exists(
        get_data_model(), "TrainingStatusResponse"
    )

    assert issubclass(
        validated_class,
        caikit.interfaces.runtime.data_model.training_management.TrainingStatusResponse,
    )

    """
    We really wanted to avoid having concrete data model objects live in caikit.core,
    but because ModuleBase used ProducerId, we had to keep that there. That said,
    it is forwarded to caikit.interfaces.common, so from a user's perspective,
    we should consider it to be part of caikit.interfaces.common not
    caikit.core.data_model."""

    validated_class = validate_caikit_library_class_exists(
        get_data_model(), "ProducerId"
    )
    assert validated_class == caikit.interfaces.common.data_model.producer.ProducerId


def test_servicer_util_validate_caikit_library_class_exists_raises_for_garbage_input():
    """Test that validate_caikit_library_class_exists raises for garbage 'model'"""
    with pytest.raises(ValueError):
        validate_caikit_library_class_exists(get_data_model(), "NonExistentClass")


# ---------------- Tests for validate_caikit_library_class_method_exists --------------------
def test_service_util_validate_caikit_library_class_method_exists_doesnt_raise():
    """Test that validate_caikit_library_class_method_exists doesn't raise for valid methods"""
    try:
        validate_caikit_library_class_method_exists(
            caikit.interfaces.runtime.data_model.training_management.ModelPointer,
            "to_proto",
        )
        validate_caikit_library_class_method_exists(
            caikit.interfaces.runtime.data_model.training_management.ModelPointer,
            "from_proto",
        )
    except Exception as e:
        assert (
            False
        ), "validate_caikit_library_class_method_exists failed for ModelPointer!"

    try:
        validate_caikit_library_class_method_exists(
            caikit.interfaces.common.data_model.producer.ProducerId, "to_proto"
        )
        validate_caikit_library_class_method_exists(
            caikit.interfaces.common.data_model.producer.ProducerId, "from_proto"
        )
    except Exception as e:
        assert False, "validate_caikit_library_class_method_exists failed!"


def test_service_util_validate_caikit_library_class_method_exists_does_raise():
    """Test that validate_caikit_library_class_method_exists raises for garbage input method"""
    with pytest.raises(AttributeError):
        validate_caikit_library_class_method_exists(
            caikit.interfaces.runtime.data_model.ModelPointer, "non_existent_method"
        )


# ---------------- Tests for build_proto_response  --------------------------
def test_servicer_util_build_proto_response_raises_on_garbage_response_type():
    class FooResponse:
        def __init__(self, foo) -> None:
            self.foo = foo

    with pytest.raises(CaikitRuntimeException):
        build_proto_response(FooResponse(foo="This is a foo response"))


# ---------------- Tests for is_protobuf_primitive_field --------------------
def test_servicer_util_is_protobuf_primitive_returns_true_for_primitive_types(
    sample_inference_service,
):
    """Test that is_protobuf_primitive_field is True when considering primitive types"""
    assert is_protobuf_primitive_field(
        sample_inference_service.messages.SampleTaskRequest().DESCRIPTOR.fields_by_name[
            "int_type"
        ]
    )


def test_servicer_util_is_protobuf_primitive_returns_false_for_custom_types(
    sample_inference_service,
):
    """Test that is_protobuf_primitive_field is False when considering message and
    enum types. This is essential for handling Caikit library CDM objects, which are
    generally defined in terms of messages"""
    assert not is_protobuf_primitive_field(
        sample_inference_service.messages.SampleTaskRequest().DESCRIPTOR.fields_by_name[
            "sample_input"
        ]
    )


def test_servicer_util_is_protobuf_primitive_returns_false_for_instance_not_in_FieldDescriptor():
    field = caikit.interfaces.runtime.data_model.ModelPointer
    assert False == (is_protobuf_primitive_field(field))


# ---------------- Tests for get_metadata --------------------
def test_servicer_util_get_metadata_raises_on_no_metadata_in_context_but_key_required():
    """Test that get_metadata raises on no metadata context, but the key is required"""
    with pytest.raises(CaikitRuntimeException):
        get_metadata(context=None, key="some-key", required=True)


def test_servicer_util_get_metadata_returns_none_for_absent_key():
    """Test that get_metadata returns None for absent invocation metadata key"""
    assert (
        get_metadata(
            Fixtures.build_context("syntax_izumo_v0-0-1_en"), "some-missing-key"
        )
        == None
    )


def test_servicer_util_get_metadata_should_get_mmmodelid():
    """Test that get_metadata returns value for present invocation metadata key"""
    assert (
        get_metadata(Fixtures.build_context("syntax_izumo_v0-0-1_en"), "mm-model-id")
        == "syntax_izumo_v0-0-1_en"
    )


# ---------------- Tests for validate_data_model --------------------
def test_servicer_util_validates_caikit_core_data_model(
    sample_inference_service, sample_train_service
):
    """Test that validate_data_model validates Caikit library CDM on initialization"""

    try:
        validate_data_model(sample_train_service.descriptor)
    except Exception as e:
        assert False, "Validation of interfaces failed for train!"

    try:
        validate_data_model(sample_inference_service.descriptor)
    except Exception as e:
        assert False, "Validation of interfaces failed for Predict!"


def test_servicer_util_will_not_validate_arbitrary_service_descriptor():
    """Test that validate_data_model raises exception validating arbitrary ServiceDescriptor"""
    with pytest.raises(ValueError):
        validate_data_model(model_runtime_pb2._MODELRUNTIME)


# ---------------- Tests for build_caikit_library_request_dict --------------------
HAPPY_PATH_INPUT = SampleInputType(name="Gabe").to_proto()


def test_global_predict_build_caikit_library_request_dict_creates_caikit_core_run_kwargs(
    sample_inference_service,
):
    """Test that build_caikit_library_request_dict creates module run kwargs from RPC msg"""
    request_dict = build_caikit_library_request_dict(
        sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        ),
        sample_lib.modules.sample_task.SampleModule.RUN_SIGNATURE,
    )

    # No self or "throw", throw was not set and the throw parameter contains a default value
    expected_arguments = {"sample_input"}

    assert expected_arguments == set(request_dict.keys())
    assert isinstance(request_dict["sample_input"], SampleInputType)


def test_global_predict_build_caikit_library_request_dict_strips_invalid_run_kwargs_from_request(
    sample_inference_service,
):
    """Global predict build_caikit_library_request_dict strips invalid run kwargs from request"""
    # Sample module doesn't take the `int_type` or `bool_type` params
    request_dict = build_caikit_library_request_dict(
        sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT,
            int_type=5,
            bool_type=True,
        ),
        sample_lib.modules.sample_task.SampleModule.RUN_SIGNATURE,
    )

    expected_arguments = {"sample_input"}
    assert expected_arguments == set(request_dict.keys())
    assert "int_type" not in request_dict.keys()


def test_global_predict_build_caikit_library_request_dict_strips_empty_list_from_request(
    sample_inference_service,
):
    """Global predict build_caikit_library_request_dict strips empty list from request"""
    request_dict = build_caikit_library_request_dict(
        sample_inference_service.messages.SampleTaskRequest(int_type=5, list_type=[]),
        sample_lib.modules.sample_task.SamplePrimitiveModule.RUN_SIGNATURE,
    )

    assert "list_type" not in request_dict.keys()
    assert "int_type" in request_dict.keys()


def test_global_predict_build_caikit_library_request_dict_works_for_unset_primitives(
    sample_inference_service,
):
    """Global predict build_caikit_library_request_dict works for primitives"""
    request = sample_inference_service.messages.SampleTaskRequest()

    request_dict = build_caikit_library_request_dict(
        request, SamplePrimitiveModule.RUN_SIGNATURE
    )
    # None of the primitive args should be there
    assert len(request_dict.keys()) == 0


def test_global_predict_build_caikit_library_request_dict_works_for_set_primitives(
    sample_inference_service,
):
    """Global predict build_caikit_library_request_dict works for primitives"""
    request = sample_inference_service.messages.SampleTaskRequest(
        int_type=5,
        float_type=4.2,
        str_type="moose",
        bytes_type=b"foo",
        list_type=["1", "2", "3"],
    )

    request_dict = build_caikit_library_request_dict(
        request, SamplePrimitiveModule.RUN_SIGNATURE
    )
    assert request_dict["int_type"] == 5
    assert request_dict["float_type"] == 4.2
    assert request_dict["str_type"] == "moose"
    assert request_dict["bytes_type"] == b"foo"
    assert request_dict["list_type"] == ["1", "2", "3"]


def test_global_train_build_caikit_library_request_dict_strips_empty_list_from_request(
    sample_train_service,
):
    """Global train build_caikit_library_request_dict strips empty list from request"""
    # NOTE: not sure this test is relevant anymore, since nothing effectively gets removed?
    # the datastream is empty but it's not removed from request, which is expected
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(jsondata=stream_type.JsonData(data=[])).to_proto()
    train_request = sample_train_service.messages.SampleTaskSampleModuleTrainRequest(
        model_name=random_test_id(), training_data=training_data
    )

    caikit.core_request = build_caikit_library_request_dict(
        train_request,
        sample_lib.modules.sample_task.SampleModule.TRAIN_SIGNATURE,
    )

    # model_name is not expected to be passed through
    expected_arguments = {"training_data", "union_list"}

    assert expected_arguments == set(caikit.core_request.keys())


def test_global_train_build_caikit_library_request_dict_works_for_repeated_fields(
    sample_train_service,
):
    """Global train build_caikit_library_request_dict works for repeated fields"""

    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(jsondata=stream_type.JsonData(data=[])).to_proto()
    train_request = sample_train_service.messages.SampleTaskListModuleTrainRequest(
        model_name=random_test_id(),
        training_data=training_data,
        poison_pills=["Bob Marley", "Bunny Livingston"],
    )

    caikit.core_request = build_caikit_library_request_dict(
        train_request, sample_lib.modules.sample_task.ListModule.TRAIN_SIGNATURE
    )

    # model_name is not expected to be passed through
    expected_arguments = {"training_data", "poison_pills"}

    assert expected_arguments == set(caikit.core_request.keys())
    assert len(caikit.core_request.keys()) == 2
    assert "poison_pills" in caikit.core_request
    assert isinstance(caikit.core_request["poison_pills"], list)


def test_global_train_build_caikit_library_request_dict_ok_with_DataStreamSourceInt_with_inline_json(
    sample_train_service,
):
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[100, 120])
    ).to_proto()

    train_request = sample_train_service.messages.OtherTaskOtherModuleTrainRequest(
        model_name="Bar Training",
        sample_input_sampleinputtype=SampleInputType(name="Gabe").to_proto(),
        batch_size=100,
        training_data=training_data,
    )
    caikit.core_request = build_caikit_library_request_dict(
        train_request,
        sample_lib.modules.other_task.OtherModule.TRAIN_SIGNATURE,
    )

    expected_arguments = {"batch_size", "sample_input", "training_data"}

    assert expected_arguments == set(caikit.core_request.keys())


def test_global_train_build_caikit_library_request_dict_ok_with_data_stream_file_type_csv(
    sample_train_service, sample_csv_file
):
    """Global train build_caikit_library_request_dict works for csv training data file"""

    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        file=stream_type.File(filename=sample_csv_file)
    ).to_proto()
    train_request = sample_train_service.messages.SampleTaskSampleModuleTrainRequest(
        model_name=random_test_id(),
        training_data=training_data,
    )

    caikit.core_request = build_caikit_library_request_dict(
        train_request,
        sample_lib.modules.sample_task.SampleModule.TRAIN_SIGNATURE,
    )

    # model_name is not expected to be passed through
    expected_arguments = {"training_data", "union_list"}

    assert expected_arguments == set(caikit.core_request.keys())


def test_global_train_build_caikit_library_request_dict_ok_with_training_data_as_list_of_files(
    sample_train_service, sample_csv_file, sample_json_file
):
    """Global train build_caikit_library_request_dict works for list of data files"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        listoffiles=stream_type.ListOfFiles(files=[sample_csv_file, sample_json_file])
    ).to_proto()
    train_request = sample_train_service.messages.SampleTaskListModuleTrainRequest(
        model_name=random_test_id(),
        training_data=training_data,
        poison_pills=["Bob Marley", "Bunny Livingston"],
    )

    caikit.core_request = build_caikit_library_request_dict(
        train_request,
        sample_lib.modules.sample_task.ListModule.TRAIN_SIGNATURE,
    )

    # model_name is not expected to be passed through
    expected_arguments = {"training_data", "poison_pills"}

    assert expected_arguments == set(caikit.core_request.keys())
    assert len(caikit.core_request.keys()) == 2
    assert "training_data" in caikit.core_request


def test_build_caikit_library_request_dict_works_when_data_stream_directory_includes_mixed_types(
    sample_train_service,
):
    """Global train build_caikit_library_request_dict works when mixed supported types are present"""

    with tempfile.TemporaryDirectory() as tempdir:
        fname1 = os.path.join(tempdir, "justacsv.csv")
        with open(fname1, "w") as handle:
            handle.writelines("valid_csv,1,2,3")
        fname2 = os.path.join(tempdir, "justajson.json")
        with open(fname2, "w") as handle:
            handle.writelines('{"number": 1}')

        stream_type = (
            caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
        )
        training_data = stream_type(
            directory=stream_type.Directory(dirname=tempdir, extension="json")
        ).to_proto()
        train_request = (
            sample_train_service.messages.SampleTaskSampleModuleTrainRequest(
                model_name=random_test_id(),
                training_data=training_data,
            )
        )

        # no error because at least 1 json file exists within the provided dir
        caikit.core_request = build_caikit_library_request_dict(
            train_request,
            sample_lib.modules.sample_task.SampleModule.TRAIN_SIGNATURE,
        )
