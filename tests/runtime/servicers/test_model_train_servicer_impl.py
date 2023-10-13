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
"""General model train servicer tests - run for both subprocess and local"""
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
from caikit.core import MODEL_MANAGER
from caikit.core.data_model import TrainingStatus
from caikit.runtime.protobufs import process_pb2
from caikit.runtime.servicers.model_train_servicer import ModelTrainServicerImpl
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import set_use_subprocess, temp_config
from tests.fixtures import Fixtures
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


# We need to add reset_model_manager to make sure all trainers use the config that we provide.
# Without this, it didn't actually work since the get_trainer function always fetches a trainer
# that was initialized before this config comes into play. Hence we’re never actually checking
# a training in a sub_process. I verified this by running a failing test and seeing that we always
# hit destroyable_thread.py in the stacktrace instead of destroyable_process in both scenarios.


@pytest.fixture(autouse=True, params=[True, False])
def set_train_location(request, reset_model_manager):
    """This fixture ensures that all tests in this file will be run with both
    subprocess and local training styles
    """
    with set_use_subprocess(request.param):
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
        context = Fixtures.build_context("foo")
        sample_model_train_servicer.Run(model_train_request, context)


def test_model_train_no_train_module_raises(sample_model_train_servicer, output_dir):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "training_params": '{"model_name": "abc", "parameters": {"training_data": [1]}}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        context = Fixtures.build_context("foo")
        sample_model_train_servicer.Run(model_train_request, context)


def test_model_train_incorrect_train_params_raises(
    sample_model_train_servicer, output_dir
):
    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": '{"model_name": "abc", "parameters": {"training_data": "blah"}}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        context = Fixtures.build_context("foo")
        sample_model_train_servicer.Run(model_train_request, context)


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
        context = Fixtures.build_context("foo")
        sample_model_train_servicer.Run(model_train_request, context)

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
            "training_params": '{"model_name": "abc", "parameters": {"training_data": [1]}}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with clear_messages_from_servicer(sample_model_train_servicer):
        with pytest.raises(CaikitRuntimeException) as e:
            context = Fixtures.build_context("foo")
            sample_model_train_servicer.Run(model_train_request, context)

        assert "Model Train not able to create a request" in str(e.value)
        assert e.value.status_code == grpc.StatusCode.INTERNAL


def test_model_train_validation_error_raises(sample_model_train_servicer, output_dir):
    """Test that if there is validation error we are able to raise it"""

    training_id = str(uuid.uuid4())
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        customTrainingID=str(uuid.uuid4()),
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": '{"model_name": "abc", "parameters": {"training_data": [1], "batch_size": 999}}',
        },
        training_input_dir="training_input_dir",
        training_output_dir=os.path.join(output_dir, training_id),
    )
    with pytest.raises(CaikitRuntimeException):
        context = Fixtures.build_context("foo")
        sample_model_train_servicer.Run(model_train_request, context)


def test_model_train_surfaces_caikit_errors(sample_model_train_servicer, output_dir):
    """Test whether model train surfaces errors from Caikit using both sub-process and thread"""
    training_id = str(uuid.uuid4())

    training_output_dir = os.path.join(output_dir, training_id)
    training_input_dir = os.path.join(output_dir, training_id, "inputs")
    # we don't support .txt files yet, hence this should throw an error
    input_file_name = "data.txt"

    os.makedirs(training_input_dir, exist_ok=True)
    with open(os.path.join(training_input_dir, input_file_name), "w") as f:
        json.dump([sample_lib.data_model.SampleTrainingType(number=1).to_dict()], f)

    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": json.dumps(
                {
                    "model_name": "abc",
                    "parameters": {
                        "training_data": {
                            "file": {
                                "filename": input_file_name  # This is relative to training_input_dir
                            },
                        },
                    },
                }
            ),
        },
        training_input_dir=training_input_dir,
        training_output_dir=training_output_dir,
    )
    context = Fixtures.build_context("test-any-unresponsive-model")
    with pytest.raises(CaikitRuntimeException) as e:
        sample_model_train_servicer.Run(model_train_request, context)
    assert isinstance(e.value, CaikitRuntimeException)
    assert "Extension not supported" in e.value.message


#####################################################################
# Normal tests
@pytest.mark.parametrize("output_in_config", (True, False))
@pytest.mark.parametrize("save_with_id", (True, False))
@pytest.mark.parametrize("use_custom_id", (True, False))
def test_model_train_sample_widget(
    sample_model_train_servicer,
    output_dir,
    output_in_config,
    save_with_id,
    use_custom_id,
):
    """This test tests end-to-end training. It includes verifying that the model
    is saved in the right place. The place where the model is saved comes from:

    1. get_config().runtime.training.output_dir
    2. request.training_output_dir

    IN THAT ORDER!
    """
    training_id = str(uuid.uuid4())
    custom_id = f"custom-{uuid.uuid4()}" if use_custom_id else None
    model_name = "abc"

    # If getting the output path from config, use a bogus dir in the request to
    # ensure that the config is preferred
    if output_in_config:
        config_output_dir = output_dir
        req_output_dir = "/some/bad/path"
    # Otherwise the config must be set to an empty string so the request is used
    else:
        config_output_dir = ""
        req_output_dir = output_dir

    with temp_config(
        {
            "runtime": {
                "training": {
                    "output_dir": config_output_dir,
                    "save_with_id": save_with_id,
                },
            },
        },
        "merge",
    ):
        model_train_request = process_pb2.ProcessRequest(
            trainingID=training_id,
            customTrainingID=custom_id,
            request_dict={
                "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
                "training_params": json.dumps(
                    {
                        "model_name": model_name,
                        "parameters": {
                            "training_data": {
                                "jsondata": {
                                    "data": [
                                        sample_lib.data_model.SampleTrainingType(
                                            number=1
                                        ).to_dict()
                                    ]
                                },
                            },
                        },
                    }
                ),
            },
            training_input_dir="training_input_dir",
            training_output_dir=req_output_dir,
        )
        context = Fixtures.build_context("test-any-unresponsive-model")
        training_response = sample_model_train_servicer.Run(
            model_train_request, context
        )
        if save_with_id:
            if use_custom_id:
                model_save_path = os.path.join(output_dir, custom_id, model_name)
            else:
                model_save_path = os.path.join(output_dir, training_id, model_name)
        else:
            model_save_path = os.path.join(output_dir, model_name)
        assert os.path.isdir(model_save_path)

    # Make sure that the request completed synchronously
    model_future = MODEL_MANAGER.get_model_future(custom_id or training_id)
    assert model_future.get_info().status == TrainingStatus.COMPLETED

    # Make sure that the return object looks right
    assert isinstance(training_response, process_pb2.ProcessResponse)
    assert training_response == process_pb2.ProcessResponse(
        trainingID=training_id,
        customTrainingID=model_train_request.customTrainingID,
    )


def test_files_from_training_input_dir_are_used(
    sample_model_train_servicer, output_dir
):
    """The MT request comes with a `training_input_dir` field.
    If datastreams reference files, they may exist inside this dynamically-created directory.
    The servicer should detect this and direct datastreams to that directory.
    """
    training_id = str(uuid.uuid4())

    training_output_dir = os.path.join(output_dir, training_id)
    training_input_dir = os.path.join(output_dir, training_id, "inputs")
    input_file_name = "data.json"

    os.makedirs(training_input_dir, exist_ok=True)
    with open(os.path.join(training_input_dir, input_file_name), "w") as f:
        json.dump([sample_lib.data_model.SampleTrainingType(number=1).to_dict()], f)

    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": json.dumps(
                {
                    "model_name": "abc",
                    "parameters": {
                        "training_data": {
                            "file": {
                                "filename": input_file_name  # This is relative to training_input_dir
                            },
                        },
                    },
                }
            ),
        },
        training_input_dir=training_input_dir,
        training_output_dir=training_output_dir,
    )
    context = Fixtures.build_context("test-any-unresponsive-model")
    training_response = sample_model_train_servicer.Run(model_train_request, context)
    assert os.path.isdir(training_output_dir)

    # Make sure that the training succeeded
    model_future = MODEL_MANAGER.get_model_future(training_response.trainingID)
    assert model_future.get_info().status == TrainingStatus.COMPLETED
