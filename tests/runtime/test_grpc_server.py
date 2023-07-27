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

# Have pylint ignore Class XXXX has no YYYY member so that we can use gRPC enums.
# pylint: disable=E1101
# Standard
from dataclasses import dataclass
from unittest import mock
import json
import os
import signal
import tempfile
import threading
import time
import uuid

# Third Party
from google.protobuf.descriptor_pool import DescriptorPool
from grpc._utilities import RpcMethodHandler
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)
import grpc
import pytest
import tls_test_tools

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import MODEL_MANAGER
from caikit.core.data_model.base import DataBase
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingJob,
    TrainingStatus,
    TrainingStatusResponse,
)
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.protobufs import (
    model_runtime_pb2,
    model_runtime_pb2_grpc,
    process_pb2,
    process_pb2_grpc,
)
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from sample_lib import InnerModule, SamplePrimitiveModule
from sample_lib.data_model import (
    OtherOutputType,
    SampleInputType,
    SampleOutputType,
    SampleTrainingType,
)
from tests.conftest import random_test_id
from tests.core.helpers import *
from tests.fixtures import Fixtures
from tests.runtime.conftest import (
    ModuleSubproc,
    register_trained_model,
    runtime_grpc_test_server,
)
import caikit.interfaces.common
import sample_lib

## Helpers #####################################################################

log = alog.use_channel("TEST-SERVE-I")

HAPPY_PATH_INPUT = SampleInputType(name="Gabe").to_proto()
HAPPY_PATH_RESPONSE = SampleOutputType(greeting="Hello Gabe").to_proto()
HAPPY_PATH_TRAIN_RESPONSE = TrainingJob(
    model_name="dummy name", training_id="dummy id"
).to_proto()


def is_good_train_response(
    actual_response, expected, model_name, training_management_stub
):
    assert dir(actual_response) == dir(expected)
    assert actual_response.training_id is not None
    assert isinstance(actual_response.training_id, str)
    assert actual_response.model_name == model_name

    status = TrainingStatus.RUNNING.value
    i = 0
    while status == TrainingStatus.RUNNING.value:
        training_info_request = TrainingInfoRequest(
            training_id=actual_response.training_id
        )
        training_management_response: TrainingStatusResponse = (
            TrainingStatusResponse.from_proto(
                training_management_stub.GetTrainingStatus(
                    training_info_request.to_proto()
                )
            )
        )
        status = training_management_response.state
        assert status != TrainingStatus.ERRORED.value
        i += 1
        assert i < 100, "Waited too long for training to complete"

    assert status == TrainingStatus.COMPLETED.value


## Tests #######################################################################


def test_model_train(runtime_grpc_server):
    """Test model train's RUN function"""
    model_train_stub = process_pb2_grpc.ProcessStub(
        runtime_grpc_server.make_local_channel()
    )
    training_id = str(uuid.uuid4())
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
        training_output_dir=os.path.join(
            runtime_grpc_server.workdir, "training_output", training_id
        ),
    )
    training_response = model_train_stub.Run(model_train_request)
    assert isinstance(training_response, process_pb2.ProcessResponse)
    assert training_response == process_pb2.ProcessResponse(
        trainingID=model_train_request.trainingID,
        customTrainingID=model_train_request.customTrainingID,
    )

    # check result of training

    # construct training request for training manager
    training_info_request = TrainingInfoRequest(training_id=training_id).to_proto()

    training_management_service: ServicePackage = (
        ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.TRAINING_MANAGEMENT,
        )
    )

    training_management_stub = training_management_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )

    # MT training is sync, so it should be COMPLETE immediately
    response: TrainingStatusResponse = TrainingStatusResponse.from_proto(
        training_management_stub.GetTrainingStatus(training_info_request)
    )
    assert response.state == TrainingStatus.COMPLETED.value

    # Make sure we wait for training to finish
    result = MODEL_MANAGER.get_model_future(response.training_id).load()

    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.sample_task.sample_implementation.SampleModule"
    )
    # Fields with defaults have expected values
    assert result.batch_size == 64
    assert result.learning_rate == 0.0015


def test_components_preinitialized(
    reset_globals, open_port, sample_inference_service, sample_train_service
):
    """Test that all model management components get pre-initialized when the
    server is instantiated
    """
    assert not MODEL_MANAGER._trainers
    assert not MODEL_MANAGER._finders
    assert not MODEL_MANAGER._initializers
    with runtime_grpc_test_server(
        open_port,
        inference_service=sample_inference_service,
        training_service=sample_train_service,
    ):
        assert MODEL_MANAGER._trainers
        assert MODEL_MANAGER._finders
        assert MODEL_MANAGER._initializers


def test_predict_sample_module_ok_response(
    sample_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.SampleTaskPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    actual_response = stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", sample_task_model_id)]
    )
    assert actual_response == HAPPY_PATH_RESPONSE


def test_predict_streaming_module(
    streaming_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.StreamingTaskPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_request = (
        sample_inference_service.messages.ServerStreamingStreamingTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
    )
    stream = stub.ServerStreamingStreamingTaskPredict(
        predict_request, metadata=[("mm-model-id", streaming_task_model_id)]
    )

    count = 0
    for response in stream:
        assert response == HAPPY_PATH_RESPONSE
        count += 1
    assert count == 10


def test_predict_sample_module_error_response(
    runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.SampleTaskPredict error response"""
    with pytest.raises(grpc.RpcError) as context:
        stub = sample_inference_service.stub_class(
            runtime_grpc_server.make_local_channel()
        )
        predict_request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
        stub.SampleTaskPredict(
            predict_request, metadata=[("mm-model-id", "random_model_id")]
        )
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.skip("Skipping for now since we're doing streaming stuff")
def test_rpc_validation_on_predict(
    sample_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Check that the server catches models sent to the wrong task RPCs"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_request = sample_inference_service.messages.OtherTaskRequest(
        sample_input_sampleinputtype=HAPPY_PATH_INPUT
    )
    with pytest.raises(
        grpc.RpcError,
        match="Wrong inference RPC invoked for model class .* Use SampleTaskPredict instead of OtherTaskPredict",
    ) as context:
        stub.OtherTaskPredict(
            predict_request, metadata=[("mm-model-id", sample_task_model_id)]
        )
    assert context.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_rpc_validation_on_predict_for_unsupported_model(
    runtime_grpc_server: RuntimeGRPCServer, sample_inference_service, tmp_path
):
    """Check that the server catches models that have no supported inference rpc"""
    unsupported_model = InnerModule()
    tmpdir = str(tmp_path)
    unsupported_model.save(tmpdir)
    model_id = random_test_id()
    try:
        runtime_grpc_server._global_predict_servicer._model_manager.load_model(
            model_id, tmpdir, "foo"
        )

        stub = sample_inference_service.stub_class(
            runtime_grpc_server.make_local_channel()
        )
        predict_request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
        with pytest.raises(grpc.RpcError) as context:
            stub.SampleTaskPredict(
                predict_request, metadata=[("mm-model-id", model_id)]
            )
        assert context.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Inference for model class" in str(context.value)
        assert "not supported by this runtime" in str(context.value)

    finally:
        runtime_grpc_server._global_predict_servicer._model_manager.unload_model(
            model_id
        )


def test_rpc_validation_on_predict_for_wrong_streaming_flavor(
    runtime_grpc_server: RuntimeGRPCServer, sample_inference_service, tmp_path
):
    """Check that the server catches models that have no supported inference rpc"""
    unary_only_model = SamplePrimitiveModule()
    tmpdir = str(tmp_path)
    unary_only_model.save(tmpdir)
    model_id = random_test_id()
    try:
        runtime_grpc_server._global_predict_servicer._model_manager.load_model(
            model_id, tmpdir, "foo"
        )

        stub = sample_inference_service.stub_class(
            runtime_grpc_server.make_local_channel()
        )
        predict_request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
        with pytest.raises(grpc.RpcError) as context:
            response = stub.ServerStreamingSampleTaskPredict(
                predict_request, metadata=[("mm-model-id", model_id)]
            )
            for r in response:
                # try to read off the stream
                pass

        assert context.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "Model class" in str(context.value)
        assert "does not support ServerStreamingSampleTaskPredict" in str(context.value)

    finally:
        runtime_grpc_server._global_predict_servicer._model_manager.unload_model(
            model_id
        )


####### End-to-end tests for train a model and then predict with it
def test_train_fake_module_ok_response_and_can_predict_with_trained_model(
    train_stub,
    inference_stub,
    runtime_grpc_server,
    sample_train_service,
    sample_inference_service,
    training_management_stub,
):
    """Test RPC CaikitRuntime.SampleTaskSampleModuleTrain successful response"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    )
    model_name = random_test_id()
    train_request_class = DataBase.get_class_for_name(
        "SampleTaskSampleModuleTrainRequest"
    )
    train_request = train_request_class(
        model_name=model_name,
        training_data=training_data,
        union_list=caikit.interfaces.common.data_model.StrSequence(
            values=["str", "sequence"]
        ),
    ).to_proto()

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)

    is_good_train_response(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_ok_response_with_loaded_model_can_predict_with_trained_model(
    sample_task_model_id,
    runtime_grpc_server,
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
    training_management_stub,
):
    """Test RPC CaikitRuntime.WorkflowsSampleTaskSampleWorkflowTrain successful response with a loaded model"""
    sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
        model_id=sample_task_model_id
    ).to_proto()
    model_name = random_test_id()
    train_request = sample_train_service.messages.SampleTaskCompositeModuleTrainRequest(
        model_name=model_name, sample_block=sample_model
    )
    actual_response = train_stub.SampleTaskCompositeModuleTrain(train_request)
    is_good_train_response(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_does_not_change_another_instance_model_of_block(
    other_task_model_id,
    runtime_grpc_server,
    sample_int_file,
    train_stub,
    inference_stub,
    training_management_stub,
    sample_train_service,
    sample_inference_service,
):
    """This test: original "stock" OtherModule model has batch size 42
    (See fixtures/models/bar/config.yml).
    We then train a custom OtherModule model with batch size 100,
    then we make a predict to each, they should have the correct batch size"""

    # Train an OtherModule with batch size 100
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    training_data = stream_type(
        file=stream_type.File(filename=sample_int_file)
    ).to_proto()

    train_request = sample_train_service.messages.OtherTaskOtherModuleTrainRequest(
        model_name="Bar Training",
        sample_input_sampleinputtype=SampleInputType(name="Gabe").to_proto(),
        batch_size=100,
        training_data=training_data,
    )
    actual_response = train_stub.OtherTaskOtherModuleTrain(train_request)
    is_good_train_response(
        actual_response,
        HAPPY_PATH_TRAIN_RESPONSE,
        "Bar Training",
        training_management_stub,
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference, and the batch size 100 was used
    predict_request = sample_inference_service.messages.OtherTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    trained_inference_response = inference_stub.OtherTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    expected_trained_inference_response = OtherOutputType(
        farewell="goodbye: Gabe 100 times"
    ).to_proto()
    assert trained_inference_response == expected_trained_inference_response

    # make sure the previously loaded OtherModule model still has batch size 42
    original_inference_response = inference_stub.OtherTaskPredict(
        predict_request, metadata=[("mm-model-id", other_task_model_id)]
    )
    expected_original_inference_response = OtherOutputType(
        farewell="goodbye: Gabe 42 times"
    ).to_proto()
    assert original_inference_response == expected_original_inference_response


def test_train_primitive_model(
    runtime_grpc_server,
    train_stub,
    inference_stub,
    training_management_stub,
    sample_train_service,
    sample_inference_service,
):
    """Test that we can make a successful training and inference call to the primitive module using primitive inputs"""

    model_name = "primitive_trained_model"
    train_request_class = DataBase.get_class_for_name(
        "SampleTaskSamplePrimitiveModuleTrainRequest"
    )
    union_list_str_dm = caikit.interfaces.common.data_model.StrSequence
    union_list_int_dm = caikit.interfaces.common.data_model.IntSequence
    union_list_bool_dm = caikit.interfaces.common.data_model.BoolSequence

    train_request = train_request_class(
        model_name=model_name,
        sample_input=SampleInputType(name="Gabe"),
        simple_list=["hello", "world"],
        union_list=union_list_str_dm(values=["str", "sequence"]),
        union_list2=union_list_int_dm(values=[1, 2]),
        union_list3=union_list_bool_dm(values=[True, False]),
        union_list4=123,
        training_params_json_dict={"foo": {"bar": [1, 2, 3]}},
        training_params_json_dict_list=[{"foo": {"bar": [1, 2, 3]}}],
        training_params_dict={"layer_sizes": 100, "window_scaling": 200},
        training_params_dict_int={1: 0.1, 2: 0.01},
    ).to_proto()

    training_response = train_stub.SampleTaskSamplePrimitiveModuleTrain(train_request)
    is_good_train_response(
        training_response,
        HAPPY_PATH_TRAIN_RESPONSE,
        model_name,
        training_management_stub,
    )
    register_trained_model(
        runtime_grpc_server, training_response.model_name, training_response.training_id
    )

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )

    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", training_response.model_name)]
    )
    expected_inference_response = SampleOutputType(
        greeting="hello: primitives! [1, 2, 3] 100"
    ).to_proto()
    assert inference_response == expected_inference_response


##### Test different datastream types #####
def test_train_fake_module_ok_response_with_datastream_jsondata(
    runtime_grpc_server,
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
    training_management_stub,
):
    """Test RPC CaikitRuntime.SampleTaskSampleModuleTrainRequest successful response with training data json type"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    ).to_proto()
    model_name = random_test_id()
    train_request = sample_train_service.messages.SampleTaskSampleModuleTrainRequest(
        model_name=model_name,
        batch_size=42,
        training_data=training_data,
    )

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)
    is_good_train_response(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_ok_response_with_datastream_csv_file(
    runtime_grpc_server,
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
    sample_csv_file,
    training_management_stub,
):
    """Test RPC CaikitRuntime.SampleTaskSampleModuleTrainRequest successful response with training data file type"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        file=stream_type.File(filename=sample_csv_file)
    ).to_proto()
    model_name = random_test_id()
    train_request = sample_train_service.messages.SampleTaskSampleModuleTrainRequest(
        model_name=model_name,
        training_data=training_data,
    )

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)
    is_good_train_response(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


#### Error cases for train tests #####
def test_train_fake_module_error_response_with_unloaded_model(
    train_stub, sample_train_service
):
    """Test RPC CaikitRuntime.SampleTaskCompositeModuleTrain error response because sample model is not loaded"""
    with pytest.raises(grpc.RpcError) as context:
        sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
            model_id=random_test_id()
        ).to_proto()

        train_request = (
            sample_train_service.messages.SampleTaskCompositeModuleTrainRequest(
                model_name=random_test_id(), sample_block=sample_model
            )
        )
        train_stub.SampleTaskCompositeModuleTrain(train_request)
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


#### ModelRuntime tests ####
def test_load_model_ok_response(runtime_grpc_server):
    """Test load model's successful response"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    load_model_request = model_runtime_pb2.LoadModelRequest(
        modelId="real",
        modelType="foo",
        modelPath=Fixtures.get_good_model_path(),
        modelKey="bar",
    )
    actual_response = stub.loadModel(load_model_request)
    assert actual_response.sizeInBytes > 0


def test_load_model_notfound_error_response(runtime_grpc_server):
    """Test load model's error response - model does not exist"""
    with pytest.raises(grpc.RpcError) as context:
        stub = model_runtime_pb2_grpc.ModelRuntimeStub(
            runtime_grpc_server.make_local_channel()
        )
        load_model_request = model_runtime_pb2.LoadModelRequest(
            modelId="foo",
            modelType="bar",
            modelPath="test/some/invalid/path",
            modelKey="baz",
        )
        stub.loadModel(load_model_request)
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


def test_load_model_badmodel_error_response(runtime_grpc_server):
    """Test load model's error response - invalid model"""
    with pytest.raises(grpc.RpcError) as context:
        stub = model_runtime_pb2_grpc.ModelRuntimeStub(
            runtime_grpc_server.make_local_channel()
        )
        load_model_request = model_runtime_pb2.LoadModelRequest(
            modelId="foo",
            modelType="bar",
            modelPath=Fixtures.get_bad_model_archive_path(),
            modelKey="baz",
        )
        stub.loadModel(load_model_request)
    assert context.value.code() == grpc.StatusCode.INTERNAL


def test_unload_model_ok_response(sample_task_model_id, runtime_grpc_server):
    """Test unload model's successful response"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    unload_model_request = model_runtime_pb2.UnloadModelRequest(
        modelId=sample_task_model_id
    )
    # Unload model throws on failure (response message has no fields)
    stub.unloadModel(unload_model_request)


def test_unload_model_does_not_throw_if_model_does_not_exist(runtime_grpc_server):
    """Unloading a model that does not exist (or has already been deleted) is totally cool"""
    try:
        stub = model_runtime_pb2_grpc.ModelRuntimeStub(
            runtime_grpc_server.make_local_channel()
        )
        unload_model_request = model_runtime_pb2.UnloadModelRequest(modelId="new_model")
        stub.unloadModel(unload_model_request)
    except grpc.RpcError:
        assert False, "Unload for a model that does not exist threw an error!"


def test_predict_model_size_ok_response(runtime_grpc_server):
    """Test predict model size successful response on model that is not loaded before"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    predict_model_size_request = model_runtime_pb2.PredictModelSizeRequest(
        modelId="foo", modelType="bar", modelPath=Fixtures.get_good_model_path()
    )
    actual_response = stub.predictModelSize(predict_model_size_request)
    assert 0 < actual_response.sizeInBytes


def test_predict_model_size_on_loaded_model_ok_response(
    sample_task_model_id, runtime_grpc_server
):
    """Test predict model size successful response on a model that has been loaded"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    predict_model_size_request = model_runtime_pb2.PredictModelSizeRequest(
        modelId=sample_task_model_id, modelPath=Fixtures.get_good_model_path()
    )
    actual_response = stub.predictModelSize(predict_model_size_request)
    assert 0 < actual_response.sizeInBytes


def test_predict_model_size_model_notfound_error_response(runtime_grpc_server):
    """Test predict model size with unknown model path error response"""
    with pytest.raises(grpc.RpcError) as context:
        stub = model_runtime_pb2_grpc.ModelRuntimeStub(
            runtime_grpc_server.make_local_channel()
        )
        predict_model_size_request = model_runtime_pb2.PredictModelSizeRequest(
            modelId=str(uuid.uuid4()), modelPath="/test/Does/Not/Exist"
        )
        stub.predictModelSize(predict_model_size_request)
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


def test_model_size_ok_response(sample_task_model_id, runtime_grpc_server):
    """Test model size successful response"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )

    model_size_request = model_runtime_pb2.ModelSizeRequest(
        modelId=sample_task_model_id
    )
    actual_response = stub.modelSize(model_size_request)
    # Mar. 14, 23
    # The size of the directory pointed to by Fixtures.get_good_model_path() is 355 now.
    expected_size = (
        355
        * get_config().inference_plugin.model_mesh.model_size_multipliers[
            Fixtures.get_good_model_type()
        ]
    )
    assert abs(actual_response.sizeInBytes - expected_size) < 100


def test_model_size_model_notfound_error_response(runtime_grpc_server):
    """Test model size with model not found error response"""
    with pytest.raises(grpc.RpcError) as context:
        stub = model_runtime_pb2_grpc.ModelRuntimeStub(
            runtime_grpc_server.make_local_channel()
        )
        model_size_request = model_runtime_pb2.ModelSizeRequest(
            modelId="unknown_model_size_test"
        )
        stub.modelSize(model_size_request)
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


def test_runtime_status_ok_response(runtime_grpc_server):
    """Test run-time status successful response."""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    runtime_status_request = model_runtime_pb2.RuntimeStatusRequest()
    actual_response = stub.runtimeStatus(runtime_status_request)
    assert actual_response.status == model_runtime_pb2.RuntimeStatusResponse.READY
    assert (
        actual_response.capacityInBytes
        == get_config().inference_plugin.model_mesh.capacity
    )
    assert actual_response.maxLoadingConcurrency == 2
    assert (
        actual_response.modelLoadingTimeoutMs
        == get_config().inference_plugin.model_mesh.model_loading_timeout_ms
    )
    assert actual_response.defaultModelSizeInBytes == 18874368
    assert actual_response.numericRuntimeVersion == 0


#### Health Probe tests ####
def test_grpc_health_probe_ok_response(runtime_grpc_server):
    """Test health check successful response"""
    stub = health_pb2_grpc.HealthStub(runtime_grpc_server.make_local_channel())
    health_check_request = health_pb2.HealthCheckRequest()
    actual_response = stub.Check(health_check_request)
    assert actual_response.status == 1


def test_grpc_server_can_render_all_necessary_protobufs(
    runtime_grpc_server, sample_inference_service, sample_train_service, tmp_path
):
    """Test service protobufs can be rendered"""
    tempdir = str(tmp_path)
    runtime_grpc_server.render_protos(tempdir)
    rendered_protos = os.listdir(tempdir)

    assert (
        f"{sample_inference_service.service.__name__.lower()}.proto" in rendered_protos
    )
    assert f"{sample_train_service.service.__name__.lower()}.proto" in rendered_protos


# Test related to handling load aborts for sad situations :(
def test_canceling_model_loads_causes_exceptions(runtime_grpc_server):
    """Test to make sure that if we cancel a load, an exception is thrown inside our loader."""

    request_received = threading.Event()
    request_finished = threading.Event()

    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )

    load_model_request = model_runtime_pb2.LoadModelRequest(
        modelId="model_load_abort_test",
        modelType="foo",
        modelPath=Fixtures.get_good_model_path(),
        modelKey="bar",
    )

    def never_return(*args, **kwargs):
        request_received.set()
        try:
            while True:
                time.sleep(0.01)
        except Exception as e:
            request_finished.set()
            raise e

    manager_instance = ModelManager.get_instance()
    with mock.patch.object(manager_instance, "load_model", never_return):
        # Call the stub - note that .future makes this thing return a future object that
        # we can wait for (or cancel) that will eventually have a meaningful result.
        load_model_future = stub.loadModel.future(load_model_request)
        request_received.wait()
        # Request has been received; now cancel it
        load_model_future.cancel()

        # Wait for an exception to be raised in our mock, and assert it was
        request_finished.wait(10)
        assert request_finished.is_set()


def test_tls(sample_inference_service, open_port):
    """Boot up a server with TLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert="", key="")
    )
    with runtime_grpc_test_server(
        open_port,
        inference_service=sample_inference_service,
        training_service=None,
        tls_config_override=tls_config,
    ) as server:
        _assert_connection(_make_secure_channel(server, ca_cert))


def test_mtls(sample_inference_service, open_port):
    """Boot up a server with mTLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert=ca_cert, key="")
    )
    with runtime_grpc_test_server(
        open_port,
        inference_service=sample_inference_service,
        training_service=None,
        tls_config_override=tls_config,
    ) as server:
        client_key, client_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)
        _assert_connection(
            _make_secure_channel(server, ca_cert, client_key, client_cert)
        )

        # Once we know we can connect with mTLS, assert we cannot with plain TLS
        bad_channel = _make_secure_channel(server, ca_cert)
        with pytest.raises(grpc.RpcError):
            stub = health_pb2_grpc.HealthStub(bad_channel)
            health_check_request = health_pb2.HealthCheckRequest()
            stub.Check(health_check_request)


def test_certs_can_be_loaded_as_files(sample_inference_service, tmp_path, open_port):
    """mTLS test with all tls configs loaded from files"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tmpdir = str(tmp_path)

    tls_cert_path = os.path.join(tmpdir, "tls.crt")
    tls_key_path = os.path.join(tmpdir, "tls.key")
    ca_cert_path = os.path.join(tmpdir, "ca.crt")
    with open(tls_cert_path, "w") as f:
        f.write(tls_cert)
    with open(tls_key_path, "w") as f:
        f.write(tls_key)
    with open(ca_cert_path, "w") as f:
        f.write(ca_cert)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert_path, key=tls_key_path),
        client=KeyPair(cert=ca_cert_path, key=""),
    )
    with runtime_grpc_test_server(
        open_port,
        inference_service=sample_inference_service,
        training_service=None,
        tls_config_override=tls_config,
    ) as server:
        client_key, client_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)
        _assert_connection(
            _make_secure_channel(server, ca_cert, client_key, client_cert)
        )


def test_metrics_stored_after_server_interrupt(
    sample_task_model_id, sample_inference_service, open_port
):
    """This tests the gRPC server's behaviour when interrupted"""

    with runtime_grpc_test_server(
        open_port,
        inference_service=sample_inference_service,
        training_service=None,
    ) as server:
        stub = sample_inference_service.stub_class(server.make_local_channel())
        predict_request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
        _ = stub.SampleTaskPredict(
            predict_request, metadata=[("mm-model-id", sample_task_model_id)]
        )

        # Interrupt server
        server.interrupt(None, None)

        # Assertions on the created metrics file
        with open(server._global_predict_servicer.rpc_meter.file_path) as f:
            data = [json.loads(line) for line in f]

            assert len(data) == 1
            assert list(data[0].keys()) == [
                "timestamp",
                "batch_size",
                "model_type_counters",
                "container_id",
            ]
            assert data[0]["batch_size"] == 1
            assert len(data[0]["model_type_counters"]) == 1
            assert data[0]["model_type_counters"] == {
                "<class 'sample_lib.modules.sample_task.sample_implementation.SampleModule'>": 1
            }


def test_reflection_enabled(runtime_grpc_server):
    """This pings the reflection API to ensure we can list the caikit services that are running"""
    # See https://github.com/grpc/grpc/blob/master/doc/python/server_reflection.md
    channel = runtime_grpc_server.make_local_channel()
    reflection_db = ProtoReflectionDescriptorDatabase(channel)

    desc_pool = DescriptorPool(reflection_db)
    service_desc = desc_pool.FindServiceByName(
        "caikit.runtime.SampleLib.SampleLibService"
    )
    method_desc = service_desc.FindMethodByName("SampleTaskPredict")
    assert method_desc is not None


def test_streaming_responses_work(runtime_grpc_server):
    """This test uses the health check's Watch rpc to ensure that a unary->stream RPC functions
    as expected."""
    stub = health_pb2_grpc.HealthStub(runtime_grpc_server.make_local_channel())
    req = health_pb2.HealthCheckRequest()
    for response in stub.Watch(req):
        assert response is not None
        break


def test_streaming_handlers_are_built_correctly(runtime_grpc_server):
    """This is a very cheat-y test of a private method to check that we build our internal
    handlers using the correct handler function"""

    class FakeHandler:
        pass

    # NB: the unary_stream case is tested via health check watch in `test_streaming_responses_work`
    # (unary_unary cases are checked in every other test)

    stream_unary_handler = RpcMethodHandler(
        request_streaming=True,
        response_streaming=False,
        request_deserializer="foo",
        response_serializer="bar",
        unary_unary=None,
        stream_unary=FakeHandler,
        unary_stream=None,
        stream_stream=None,
    )
    new_handler = runtime_grpc_server.server._make_new_handler(stream_unary_handler)
    assert new_handler.stream_unary is not None
    assert new_handler.stream_unary.__name__ == "safe_rpc_call"

    stream_stream_handler = RpcMethodHandler(
        request_streaming=True,
        response_streaming=True,
        request_deserializer="foo",
        response_serializer="bar",
        unary_unary=None,
        stream_unary=None,
        unary_stream=None,
        stream_stream=FakeHandler,
    )
    new_handler = runtime_grpc_server.server._make_new_handler(stream_stream_handler)
    assert new_handler.stream_stream is not None
    assert new_handler.stream_stream.__name__ == "safe_rpc_call"


def test_grpc_sever_shutdown_with_model_poll(open_port):
    """Test that a SIGINT successfully shuts down the running server"""
    with tempfile.TemporaryDirectory() as workdir:
        server_proc = ModuleSubproc(
            "caikit.runtime.grpc_server",
            kill_timeout=30.0,
            RUNTIME_GRPC_PORT=str(open_port),
            RUNTIME_LOCAL_MODELS_DIR=workdir,
            RUNTIME_LAZY_LOAD_LOCAL_MODELS="true",
            RUNTIME_LAZY_LOAD_POLL_PERIOD_SECONDS="0.1",
            RUNTIME_METRICS_ENABLED="false",
        )
        with server_proc as proc:

            # Wait for the server to be up
            _assert_connection(
                grpc.insecure_channel(f"localhost:{open_port}"), max_failures=500
            )

            # Signal the server to shut down
            proc.send_signal(signal.SIGINT)

        # Make sure the process was not killed
        assert not server_proc.killed


# Test implementation details #########################
@dataclass
class KeyPair:
    cert: str
    key: str


@dataclass
class TLSConfig:
    server: KeyPair
    client: KeyPair


def _make_secure_channel(
    server: RuntimeGRPCServer,
    ca_cert: str,
    client_key: str = None,
    client_cert: str = None,
):
    if client_key and client_cert:
        # mTLS
        credentials = grpc.ssl_channel_credentials(
            root_certificates=bytes(ca_cert, "utf-8"),
            private_key=bytes(client_key, "utf-8"),
            certificate_chain=bytes(client_cert, "utf-8"),
        )
    else:
        # TLS
        credentials = grpc.ssl_channel_credentials(
            root_certificates=bytes(ca_cert, "utf-8")
        )
    return grpc.secure_channel(f"localhost:{server.port}", credentials=credentials)


def _assert_connection(channel, max_failures=20):
    """Check that we can ping the server on this channel.

    Assumes that it will come up, but it's hard to distinguish between a failure to boot and a
    failure of TLS communication.
    """
    done = False
    failures = 0
    while not done:
        try:
            health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest())
            done = True
        except grpc.RpcError as e:
            log.debug(
                f"[RpcError] {e}; will try to reconnect to test server in 0.1 second."
            )

            time.sleep(0.1)
            failures += 1
            if failures > max_failures:
                raise e
