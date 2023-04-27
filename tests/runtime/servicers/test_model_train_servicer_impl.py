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
from contextlib import contextmanager
import json
import os
import tempfile
import uuid

# Third Party
import grpc
import pytest

# Local
from caikit.interfaces.runtime.data_model import TrainingStatus
from caikit.runtime.protobufs import process_pb2
from caikit.runtime.servicers.model_train_servicer import ModelTrainServicerImpl
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import temp_config
import sample_lib


@pytest.fixture
def sample_model_train_servicer(sample_train_service) -> ModelTrainServicerImpl:
    servicer = ModelTrainServicerImpl(training_service=sample_train_service)
    yield servicer


@pytest.fixture
def output_dir() -> str:
    with tempfile.TemporaryDirectory() as workdir:
        yield workdir


# temporarily clear all messages from servicer
@contextmanager
def clear_messages_from_servicer(servicer):
    messages = servicer._training_service.messages
    servicer._training_service.messages = []
    try:
        yield
    finally:
        servicer._training_service.messages = messages


@pytest.fixture(autouse=True, params=[True, False])
def set_train_location(request):
    """This fixture ensures that all tests in this file will be run with both
    subprocess and local training styles
    """
    with temp_config({"training": {"use_subprocess": request.param}}):
        yield


#####################################################################
# Error tests


def test_model_train_no_training_params_raises(sample_model_train_servicer, output_dir):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        sample_model_train_servicer.Run(model_train_request)


def test_model_train_no_train_module_raises(sample_model_train_servicer, output_dir):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "training_params": '{"model_name": "abc", "training_data": [1]}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        sample_model_train_servicer.Run(model_train_request)


def test_model_train_incorrect_train_params_raises(
    sample_model_train_servicer, output_dir
):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": '{"model_name": "abc", "training_data": "blah"}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        sample_model_train_servicer.Run(model_train_request)


def test_model_train_incorrect_train_module_raises(
    sample_model_train_servicer, output_dir
):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "random_id",
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException) as e:
        sample_model_train_servicer.Run(model_train_request)
    assert "Model Train not able to parse module for this Train Request" in str(e.value)
    assert e.value.status_code == grpc.StatusCode.INVALID_ARGUMENT


def test_model_train_incorrect_train_request_raises(
    sample_model_train_servicer, output_dir
):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": '{"model_name": "abc", "training_data": [1]}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with clear_messages_from_servicer(sample_model_train_servicer):
        with pytest.raises(CaikitRuntimeException) as e:
            sample_model_train_servicer.Run(model_train_request)
        assert "Model Train not able to create a request" in str(e.value)
        assert e.value.status_code == grpc.StatusCode.INTERNAL


#####################################################################
# Normal tests
def test_model_train_sample_widget(sample_model_train_servicer, output_dir):
    training_id = str(uuid.uuid4())

    training_output_dir = os.path.join(output_dir, training_id)
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": json.dumps(
                {
                    "model_name": "abc",
                    "training_data": {
                        "jsondata": {
                            "data": [
                                sample_lib.data_model.SampleTrainingType(
                                    number=1
                                ).to_dict()
                            ]
                        },
                    },
                }
            ),
        },
        training_input_dir="training_input_dir",
        training_output_dir=training_output_dir,
    )
    training_response = sample_model_train_servicer.Run(model_train_request)
    assert os.path.isdir(training_output_dir)

    # Make sure that the request completed synchronously
    assert (
        sample_model_train_servicer._gts.training_manager.get_training_status(
            training_id
        )
        == TrainingStatus.COMPLETED
    )
    assert sample_model_train_servicer._gts.training_map[training_id].done()

    # Make sure that the return object looks right
    assert isinstance(training_response, process_pb2.ProcessResponse)
    assert training_response == process_pb2.ProcessResponse(
        trainingID=model_train_request.trainingID,
        customTrainingID=model_train_request.customTrainingID,
    )
