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
import signal
import tempfile
import threading
import time

# Third Party
from google.protobuf.descriptor_pool import DescriptorPool
from grpc._utilities import RpcMethodHandler
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha.proto_reflection_descriptor_database import (
    ProtoReflectionDescriptorDatabase,
)
import grpc
import pytest
import requests
import tls_test_tools

# First Party
import alog

# Local
from caikit import get_config
from caikit.core.data_model.producer import ProducerId
from caikit.interfaces.common.data_model import File
from caikit.interfaces.runtime.data_model import (
    DeployModelRequest,
    ModelInfo,
    ModelInfoRequest,
    ModelInfoResponse,
    RuntimeInfoRequest,
    RuntimeInfoResponse,
    TrainingInfoRequest,
    TrainingJob,
    TrainingStatusResponse,
    UndeployModelRequest,
)
from caikit.runtime import (
    get_inference_request,
    get_train_params,
    get_train_request,
    http_server,
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
from caikit.runtime.utils.servicer_util import build_caikit_library_request_dict
from sample_lib import CompositeModule, InnerModule, OtherModule, SamplePrimitiveModule
from sample_lib.data_model import (
    OtherOutputType,
    SampleInputType,
    SampleOutputType,
    SampleTrainingType,
)
from sample_lib.data_model.sample import OtherTask, SampleTask, StreamingTask
from sample_lib.modules import FirstTask
from tests.conftest import ARM_ARCH, PROTOBUF_VERSION, random_test_id, temp_config
from tests.core.helpers import *
from tests.fixtures import Fixtures
from tests.runtime.conftest import (
    KeyPair,
    ModuleSubproc,
    TLSConfig,
    deploy_good_model_files,
    get_open_port,
    register_trained_model,
    runtime_grpc_test_server,
)
from tests.runtime.model_management.test_model_manager import (
    non_singleton_model_managers,
)
import caikit.interfaces.common

## Helpers #####################################################################

log = alog.use_channel("TEST-SERVE-I")

HAPPY_PATH_INPUT_DM = SampleInputType(name="Gabe")
HAPPY_PATH_INPUT = HAPPY_PATH_INPUT_DM.to_proto()
HAPPY_PATH_RESPONSE = SampleOutputType(greeting="Hello Gabe").to_proto()
HAPPY_PATH_TRAIN_RESPONSE = TrainingJob(
    model_name="dummy name", training_id="dummy id"
).to_proto()


def assert_training_successful(
    actual_response, expected, model_name, training_management_stub
):
    assert dir(actual_response) == dir(expected)
    assert actual_response.training_id is not None
    assert isinstance(actual_response.training_id, str)
    assert actual_response.model_name == model_name

    MODEL_MANAGER.get_model_future(actual_response.training_id).wait()

    training_info_request = TrainingInfoRequest(training_id=actual_response.training_id)
    training_management_response: TrainingStatusResponse = (
        TrainingStatusResponse.from_proto(
            training_management_stub.GetTrainingStatus(training_info_request.to_proto())
        )
    )
    status = training_management_response.state
    assert status == TrainingStatus.COMPLETED.value


## Tests #######################################################################


def test_model_train(runtime_grpc_server):
    """Test model train's RUN function"""
    model_train_stub = process_pb2_grpc.ProcessStub(
        runtime_grpc_server.make_local_channel()
    )
    training_id = str(uuid.uuid4())
    model_name = "abc"
    model_train_request = process_pb2.ProcessRequest(
        trainingID=training_id,
        request_dict={
            "train_module": "00110203-0405-0607-0809-0a0b02dd0e0f",
            "training_params": json.dumps(
                {
                    "model_name": model_name,
                    "parameters": {
                        "training_data": {
                            "jsondata": {
                                "data": [SampleTrainingType(number=1).to_dict()]
                            },
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
    model_future = MODEL_MANAGER.get_model_future(response.training_id)
    result = model_future.load()

    # Make sure we put the path bits in the right order: base/training_id/model_name
    training_id_and_model_name_path_bit = os.path.join(training_id, model_name)
    assert training_id_and_model_name_path_bit in model_future.save_path

    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.sample_task.sample_implementation.SampleModule"
    )
    # Fields with defaults have expected values
    assert result.batch_size == 64
    assert result.learning_rate == 0.0015


def test_components_preinitialized(reset_globals, open_port):
    """Test that all model management components get pre-initialized when the
    server is instantiated
    """
    assert not MODEL_MANAGER._trainers
    assert not MODEL_MANAGER._finders
    assert not MODEL_MANAGER._initializers
    with runtime_grpc_test_server(
        open_port,
    ):
        assert MODEL_MANAGER._trainers
        assert MODEL_MANAGER._finders
        assert MODEL_MANAGER._initializers


def test_predict_sample_module_ok_response(
    sample_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.SampleTaskPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_request = get_inference_request(SampleTask)(
        sample_input=HAPPY_PATH_INPUT_DM
    ).to_proto()

    actual_response = stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", sample_task_model_id)]
    )

    assert actual_response == HAPPY_PATH_RESPONSE


def test_predict_sample_module_multi_task_response(
    multi_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.SampleTaskPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_class = get_inference_request(FirstTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()

    actual_response = stub.FirstTaskPredict(
        predict_request, metadata=[("mm-model-id", multi_task_model_id)]
    )

    assert actual_response == SampleOutputType("Hello from FirstTask").to_proto()


def test_global_predict_build_caikit_library_request_dict_creates_caikit_core_run_kwargs(
    sample_inference_service,
):
    """Test using proto versus pythonic data model for inference requests to compare diffs"""
    # protobuf request
    proto_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT_DM.to_proto(),
    )
    proto_request_dict = build_caikit_library_request_dict(
        proto_request,
        SampleModule.RUN_SIGNATURE,
    )

    # unset fields not included
    proto_expected_arguments = {"sample_input"}
    assert proto_request.HasField("throw") is False
    assert proto_expected_arguments == set(proto_request_dict.keys())

    # pythonic data model request
    predict_class = get_inference_request(SampleTask)
    python_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()

    python_sample_module_request_dict = build_caikit_library_request_dict(
        python_request,
        SampleModule.RUN_SIGNATURE,
    )

    # unset fields are included if they have defaults set
    python_expected_arguments = {"sample_input", "throw"}
    assert python_request.HasField("throw") is True
    assert python_expected_arguments == set(python_sample_module_request_dict.keys())


def test_predict_streaming_module(
    streaming_task_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.StreamingTaskPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_class = get_inference_request(
        StreamingTask, input_streaming=False, output_streaming=True
    )
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()

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
        predict_class = get_inference_request(SampleTask)
        predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()

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
    predict_class = get_inference_request(OtherTask)
    predict_request = predict_class(
        sample_input_sampleinputtype=HAPPY_PATH_INPUT_DM
    ).to_proto()

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
        predict_class = get_inference_request(SampleTask)
        predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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

        predict_class = get_inference_request(
            SampleTask,
        )
        predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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
    train_request = get_train_request(SampleModule)(
        model_name=model_name,
        parameters=get_train_params(SampleModule)(
            training_data=training_data,
            union_list=["str", "sequence"],
        ),
    ).to_proto()

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)

    assert_training_successful(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_class = get_inference_request(SampleTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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
    )
    model_name = random_test_id()
    train_request = get_train_request(CompositeModule)(
        model_name=model_name,
        parameters=get_train_params(CompositeModule)(sample_block=sample_model),
    ).to_proto()

    actual_response = train_stub.SampleTaskCompositeModuleTrain(train_request)
    assert_training_successful(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_class = get_inference_request(SampleTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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
        file=stream_type.FileReference(filename=sample_int_file)
    )

    train_request = get_train_request(OtherModule)(
        model_name="Bar Training",
        parameters=get_train_params(OtherModule)(
            sample_input_sampleinputtype=HAPPY_PATH_INPUT_DM,
            batch_size=100,
            training_data=training_data,
        ),
    ).to_proto()
    actual_response = train_stub.OtherTaskOtherModuleTrain(train_request)
    assert_training_successful(
        actual_response,
        HAPPY_PATH_TRAIN_RESPONSE,
        "Bar Training",
        training_management_stub,
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference, and the batch size 100 was used
    predict_class = get_inference_request(OtherTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
    trained_inference_response = inference_stub.OtherTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    expected_trained_inference_response = OtherOutputType(
        farewell="goodbye: Gabe 100 times", producer_id=ProducerId("other_mod", "1.1.1")
    ).to_proto()
    assert trained_inference_response == expected_trained_inference_response

    # make sure the previously loaded OtherModule model still has batch size 42
    original_inference_response = inference_stub.OtherTaskPredict(
        predict_request, metadata=[("mm-model-id", other_task_model_id)]
    )
    expected_original_inference_response = OtherOutputType(
        farewell="goodbye: Gabe 42 times", producer_id=ProducerId("other_mod", "1.1.1")
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

    train_request = get_train_request(SamplePrimitiveModule)(
        model_name=model_name,
        parameters=get_train_params(SamplePrimitiveModule)(
            sample_input=HAPPY_PATH_INPUT_DM,
            simple_list=["hello", "world"],
            union_list=["str", "sequence"],
            union_list2=[1, 2],
            union_list3=[True, False],
            union_list4=123,
            training_params_json_dict={"foo": {"bar": [1, 2, 3]}},
            training_params_json_dict_list=[{"foo": {"bar": [1, 2, 3]}}],
            training_params_dict={"layer_sizes": 100, "window_scaling": 200},
            training_params_dict_int={1: 0.1, 2: 0.01},
        ),
    ).to_proto()

    training_response = train_stub.SampleTaskSamplePrimitiveModuleTrain(train_request)
    assert_training_successful(
        training_response,
        HAPPY_PATH_TRAIN_RESPONSE,
        model_name,
        training_management_stub,
    )
    register_trained_model(
        runtime_grpc_server, training_response.model_name, training_response.training_id
    )

    # make sure the trained model can run inference
    predict_class = get_inference_request(SampleTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()

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
    )
    model_name = random_test_id()

    train_request = get_train_request(SampleModule)(
        model_name=model_name,
        parameters=get_train_params(SampleModule)(
            batch_size=42,
            training_data=training_data,
        ),
    ).to_proto()

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)
    assert_training_successful(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_class = get_inference_request(SampleTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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
        file=stream_type.FileReference(filename=sample_csv_file)
    )
    model_name = random_test_id()

    train_request = get_train_request(SampleModule)(
        model_name=model_name,
        parameters=get_train_params(SampleModule)(
            training_data=training_data,
        ),
    ).to_proto()

    actual_response = train_stub.SampleTaskSampleModuleTrain(train_request)
    assert_training_successful(
        actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name, training_management_stub
    )
    register_trained_model(
        runtime_grpc_server, actual_response.model_name, actual_response.training_id
    )

    # make sure the trained model can run inference
    predict_class = get_inference_request(SampleTask)
    predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_and_successfully_cancel_training(
    train_stub, sample_train_service, training_management_stub
):
    # train a model, make sure training doesn't have error
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    )
    model_name = random_test_id()
    # start a training that sleeps for a long time, so I can cancel

    train_request = get_train_request(SampleModule)(
        model_name=model_name,
        parameters=get_train_params(SampleModule)(
            training_data=training_data, sleep_time=10
        ),
    ).to_proto()
    train_response = train_stub.SampleTaskSampleModuleTrain(train_request)

    training_id = train_response.training_id
    training_info_request = TrainingInfoRequest(training_id=training_id)
    training_management_response: TrainingStatusResponse = (
        TrainingStatusResponse.from_proto(
            training_management_stub.GetTrainingStatus(training_info_request.to_proto())
        )
    )
    assert (
        training_management_response.state == TrainingStatus.RUNNING.value
    ), "Could not cancel this training within 10s"
    # cancel the training
    canceled_response = training_management_stub.CancelTraining(
        training_info_request.to_proto()
    )
    assert canceled_response.state == TrainingStatus.CANCELED.value


def test_cancel_does_not_affect_other_models(
    train_stub, sample_train_service, training_management_stub
):
    # train a model, make sure training is running
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    )
    model_name = random_test_id()
    # start a training that sleeps for a long time, so I can cancel
    train_request = get_train_request(SampleModule)(
        model_name=model_name,
        parameters=get_train_params(SampleModule)(
            training_data=training_data, sleep_time=10
        ),
    ).to_proto()
    train_response = train_stub.SampleTaskSampleModuleTrain(train_request)

    assert dir(train_response) == dir(HAPPY_PATH_TRAIN_RESPONSE)
    assert train_response.training_id is not None
    assert isinstance(train_response.training_id, str)
    assert train_response.model_name == model_name

    training_id = train_response.training_id
    training_info_request = TrainingInfoRequest(training_id=training_id)
    training_management_response: TrainingStatusResponse = (
        TrainingStatusResponse.from_proto(
            training_management_stub.GetTrainingStatus(training_info_request.to_proto())
        )
    )

    # train another model
    model_name2 = random_test_id()
    train_request2 = get_train_request(SampleModule)(
        model_name=model_name2,
        parameters=get_train_params(SampleModule)(training_data=training_data),
    ).to_proto()
    train_response2 = train_stub.SampleTaskSampleModuleTrain(train_request2)

    # cancel the first training
    assert (
        training_management_response.state == TrainingStatus.RUNNING.value
    ), "Could not cancel this training within 10s"
    canceled_response = training_management_stub.CancelTraining(
        training_info_request.to_proto()
    )
    assert canceled_response.state == TrainingStatus.CANCELED.value

    # second training should be COMPLETED
    assert_training_successful(
        train_response2,
        HAPPY_PATH_TRAIN_RESPONSE,
        model_name2,
        training_management_stub,
    )


#### Error cases for train tests #####
def test_train_fake_module_error_response_with_unloaded_model(
    train_stub, sample_train_service
):
    """Test RPC CaikitRuntime.SampleTaskCompositeModuleTrain error response because sample model is not loaded"""
    with pytest.raises(grpc.RpcError) as context:
        sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
            model_id=random_test_id()
        )
        train_request = get_train_request(CompositeModule)(
            model_name=random_test_id(),
            parameters=get_train_params(CompositeModule)(sample_block=sample_model),
        ).to_proto()

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
    assert context.value.code() == grpc.StatusCode.NOT_FOUND


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


def test_runtime_info_ok_response(runtime_grpc_server):
    runtime_info_service: ServicePackage = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFO,
    )

    runtime_info_stub = runtime_info_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )

    runtime_request = RuntimeInfoRequest()
    runtime_info_response: RuntimeInfoResponse = RuntimeInfoResponse.from_proto(
        runtime_info_stub.GetRuntimeInfo(runtime_request.to_proto())
    )

    assert "caikit" in runtime_info_response.python_packages
    # runtime_version not added if not set
    assert runtime_info_response.runtime_version == ""
    # dependent libraries not added if all packages not set to true
    assert "py_to_proto" not in runtime_info_response.python_packages


def test_runtime_info_ok_response_all_packages(runtime_grpc_server):
    with temp_config(
        {
            "runtime": {
                "version_info": {
                    "python_packages": {
                        "all": True,
                    },
                    "runtime_image": "1.2.3",
                }
            },
        },
        "merge",
    ):
        runtime_info_service: ServicePackage = (
            ServicePackageFactory().get_service_package(
                ServicePackageFactory.ServiceType.INFO,
            )
        )

        runtime_info_stub = runtime_info_service.stub_class(
            runtime_grpc_server.make_local_channel()
        )

        runtime_request = RuntimeInfoRequest()
        runtime_info_response: RuntimeInfoResponse = RuntimeInfoResponse.from_proto(
            runtime_info_stub.GetRuntimeInfo(runtime_request.to_proto())
        )

        assert "caikit" in runtime_info_response.python_packages
        assert runtime_info_response.runtime_version == "1.2.3"
        # dependent libraries versions added
        assert "alog" in runtime_info_response.python_packages
        assert "py_to_proto" in runtime_info_response.python_packages


def test_all_model_info_ok_response(runtime_grpc_server, sample_task_model_id):
    info_service: ServicePackage = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFO,
    )

    model_info_stub = info_service.stub_class(runtime_grpc_server.make_local_channel())

    model_request = ModelInfoRequest()
    model_info_response: ModelInfoResponse = ModelInfoResponse.from_proto(
        model_info_stub.GetModelsInfo(model_request.to_proto())
    )

    assert len(model_info_response.models) > 0

    found_sample_task = False
    for model in model_info_response.models:
        # Assert name and id exist
        assert model.name and model.module_id
        # Assert loaded is set (could be True or False)
        assert model.loaded is not None
        # Assert metadata module_name matches expected
        if model.name == sample_task_model_id:
            assert model.module_metadata.get("name") == "SampleModule"
            found_sample_task = True

    assert found_sample_task, "Unable to find sample_task model in models list"


def test_single_model_info_ok_response(runtime_grpc_server, sample_task_model_id):
    info_service: ServicePackage = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFO,
    )

    model_info_stub = info_service.stub_class(runtime_grpc_server.make_local_channel())

    model_request = ModelInfoRequest(model_ids=[sample_task_model_id])
    model_info_response: ModelInfoResponse = ModelInfoResponse.from_proto(
        model_info_stub.GetModelsInfo(model_request.to_proto())
    )

    # Assert only one model was returned
    assert len(model_info_response.models) == 1
    model = model_info_response.models[0]
    # Assert name and id exist
    assert model.name and model.module_id
    # Assert metadata module_name matches expected
    assert model.module_metadata.get("name") == "SampleModule"


#### Health Probe tests ####
def test_grpc_health_probe_ok_response(runtime_grpc_server):
    """Test health check successful response"""
    stub = health_pb2_grpc.HealthStub(runtime_grpc_server.make_local_channel())
    health_check_request = health_pb2.HealthCheckRequest()
    actual_response = stub.Check(health_check_request)
    assert actual_response.status == 1


@pytest.mark.skipif(
    PROTOBUF_VERSION < 4 and ARM_ARCH, reason="protobuf 3 serialization bug"
)
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
        try:
            request_received.set()
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
        request_finished.wait(2)
        assert request_finished.is_set()


def test_tls(open_port):
    """Boot up a server with TLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert="", key="")
    )
    with runtime_grpc_test_server(
        open_port,
        tls_config_override=tls_config,
    ) as server:
        _assert_connection(_make_secure_channel(server, ca_cert))


def test_mtls(open_port):
    """Boot up a server with mTLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert=ca_cert, key="")
    )
    with runtime_grpc_test_server(
        open_port,
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


def test_mtls_different_root(open_port):
    """Make sure mtls communication works when the CA for the client is not the
    same as the CA for the server (including health checks using the server's
    CA)
    """
    # Server TLS Infra
    server_ca_key = tls_test_tools.generate_key()[0]
    server_ca_cert = tls_test_tools.generate_ca_cert(server_ca_key)
    server_tls_key, server_tls_cert = tls_test_tools.generate_derived_key_cert_pair(
        server_ca_key
    )

    # Client TLS Infra
    client_ca_key = tls_test_tools.generate_key()[0]
    client_ca_cert = tls_test_tools.generate_ca_cert(
        client_ca_key, common_name="my.client"
    )
    client_tls_key, client_tls_cert = tls_test_tools.generate_derived_key_cert_pair(
        client_ca_key, common_name="my.client"
    )

    server_tls_config = TLSConfig(
        server=KeyPair(cert=server_tls_cert, key=server_tls_key),
        client=KeyPair(cert=client_ca_cert, key=""),
    )
    with runtime_grpc_test_server(
        open_port,
        tls_config_override=server_tls_config,
    ) as server:
        # Connect using the client's creds
        _assert_connection(
            _make_secure_channel(
                server, server_ca_cert, client_tls_key, client_tls_cert
            )
        )
        # Connect using the server's creds
        _assert_connection(
            _make_secure_channel(
                server, server_ca_cert, server_tls_key, server_tls_cert
            )
        )


@pytest.mark.parametrize(
    ["enable_inference", "enable_training"],
    [(True, False), (False, True), (False, False)],
)
def test_services_disabled(open_port, enable_inference, enable_training):
    """Boot up a server with different combinations of services disabled"""
    with temp_config(
        {
            "runtime": {
                "service_generation": {
                    "enable_inference": enable_inference,
                    "enable_training": enable_training,
                }
            },
        },
        "merge",
    ):
        with runtime_grpc_test_server(open_port) as server:
            _assert_connection(server.make_local_channel())
            assert server.enable_inference == enable_inference
            assert (
                server._global_predict_servicer
                and server.model_management_service
                and enable_inference
            ) or (
                server._global_predict_servicer is None
                and server.model_management_service is None
                and not enable_inference
            )
            assert server.enable_training == enable_training
            assert (
                server.training_service
                and server.training_management_service
                and enable_training
            ) or (
                server.training_service is None
                and server.training_management_service is None
                and not enable_training
            )


def test_certs_can_be_loaded_as_files(tmp_path, open_port):
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

    with temp_config({"runtime": {"metering": {"enabled": True}}}, "merge"):
        with runtime_grpc_test_server(open_port) as server:
            stub = sample_inference_service.stub_class(server.make_local_channel())
            predict_class = get_inference_request(SampleTask)
            predict_request = predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto()
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


def test_all_signal_handlers_invoked(open_port):
    """Test that a SIGINT successfully shuts down all running servers"""

    # whoops, need 2 ports. Try to find another open one that isn't the one we already have
    other_open_port = get_open_port()

    with tempfile.TemporaryDirectory() as workdir:
        server_proc = ModuleSubproc(
            "caikit.runtime",
            kill_timeout=30.0,
            RUNTIME_GRPC_PORT=str(open_port),
            RUNTIME_HTTP_PORT=str(other_open_port),
            RUNTIME_LOCAL_MODELS_DIR=workdir,
            RUNTIME_LAZY_LOAD_LOCAL_MODELS="true",
            RUNTIME_LAZY_LOAD_POLL_PERIOD_SECONDS="0.1",
            RUNTIME_METRICS_ENABLED="false",
            RUNTIME_GRPC_ENABLED="true",
            RUNTIME_HTTP_ENABLED="true",
            LOG_LEVEL="info",
        )
        with server_proc as proc:
            # Wait for the grpc server to be up:
            _assert_connection(
                grpc.insecure_channel(f"localhost:{open_port}"), max_failures=500
            )

            # Then wait for the http server as well:
            http_failures = 0
            while http_failures < 500:
                try:
                    resp = requests.get(
                        f"http://localhost:{other_open_port}{http_server.HEALTH_ENDPOINT}",
                        timeout=0.1,
                    )
                    resp.raise_for_status()
                    break
                except (
                    requests.HTTPError,
                    requests.ConnectionError,
                    requests.ConnectTimeout,
                ):
                    http_failures += 1
                    # tiny sleep because a connection refused won't hit the full `0.1`s timeout
                    time.sleep(0.001)

            # Signal the server to shut down
            proc.send_signal(signal.SIGINT)

        # Make sure the process was not killed
        assert not server_proc.killed
        # Check the logs (barf) to see if both grpc and http signal handlers called
        # communicate returns (stdout, stderr) in bytes
        logs = server_proc.proc.communicate()[1].decode("utf-8")
        assert "Shutting down gRPC server" in logs
        assert "Shutting down http server" in logs


def test_construct_with_options(open_port, sample_train_service, sample_int_file):
    """Make sure that the server can be booted with config options"""
    with temp_config(
        {
            "runtime": {
                "grpc": {
                    "port": open_port,
                    "options": {
                        "grpc.max_receive_message_length": 1,
                    },
                }
            }
        },
        "merge",
    ):
        server = RuntimeGRPCServer()
        with server:
            # Send a request with a payload that's too big and make sure it gets
            # rejected
            stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
            training_data = stream_type(
                file=stream_type.FileReference(filename=sample_int_file)
            ).to_proto()
            train_request = sample_train_service.messages.OtherTaskOtherModuleTrainRequest(
                model_name="Bar Training",
                parameters=sample_train_service.messages.OtherTaskOtherModuleTrainParameters(
                    sample_input_sampleinputtype=SampleInputType(
                        name="Gabe"
                    ).to_proto(),
                    batch_size=100,
                    training_data=training_data,
                ),
            )
            train_stub = sample_train_service.stub_class(server.make_local_channel())
            with pytest.raises(grpc.RpcError) as context:
                train_stub.OtherTaskOtherModuleTrain(train_request)
            assert context.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_grpc_server_socket_listen():
    """Make sure that the server correctly listen on a unix socket"""
    with tempfile.TemporaryDirectory() as socket_dir:
        with temp_config(
            {"runtime": {"grpc": {"unix_socket_path": socket_dir + "/grpc.sock"}}},
            "merge",
        ):
            with RuntimeGRPCServer():
                stub = model_runtime_pb2_grpc.ModelRuntimeStub(
                    grpc.insecure_channel(f"unix://{socket_dir}/grpc.sock")
                )
                runtime_status_request = model_runtime_pb2.RuntimeStatusRequest()
                actual_response = stub.runtimeStatus(runtime_status_request)
                assert (
                    actual_response.status
                    == model_runtime_pb2.RuntimeStatusResponse.READY
                )


def test_grpc_server_model_management_lifecycle(
    open_port, sample_inference_service, deploy_good_model_files
):
    """Test that models can be deployed/undeployed and reflect in the
    local_models_dir
    """
    info_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.INFO,
    )
    model_management_service = ServicePackageFactory().get_service_package(
        ServicePackageFactory.ServiceType.MODEL_MANAGEMENT,
    )
    with tempfile.TemporaryDirectory() as workdir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": workdir,
                    "lazy_load_local_models": True,
                },
            },
            "merge",
        ):
            with runtime_grpc_test_server(open_port) as server:
                local_channel = server.make_local_channel()
                _assert_connection(local_channel)
                info_stub = info_service.stub_class(local_channel)
                mm_stub = model_management_service.stub_class(local_channel)
                inf_stub = sample_inference_service.stub_class(local_channel)

                # Make sure no models loaded initially
                resp = ModelInfoResponse.from_proto(
                    info_stub.GetModelsInfo(ModelInfoRequest().to_proto())
                )
                assert len(resp.models) == 0

                # Do the deploy
                model_id = "my-model"
                deploy_req = DeployModelRequest(
                    model_id=model_id,
                    model_files=[
                        File(filename=fname, data=data)
                        for fname, data in deploy_good_model_files.items()
                    ],
                )
                deploy_resp = ModelInfo.from_proto(
                    mm_stub.DeployModel(deploy_req.to_proto())
                )
                assert deploy_resp.name == model_id
                model_path = os.path.join(workdir, model_id)
                assert deploy_resp.model_path == model_path
                assert os.path.isdir(model_path)

                # Call inference on the model
                inf_resp = inf_stub.SampleTaskPredict(
                    get_inference_request(SampleTask)(
                        sample_input=HAPPY_PATH_INPUT_DM
                    ).to_proto(),
                    metadata=[("mm-model-id", model_id)],
                )
                assert inf_resp == HAPPY_PATH_RESPONSE

                # Make sure model shows as loaded
                resp = ModelInfoResponse.from_proto(
                    info_stub.GetModelsInfo(ModelInfoRequest().to_proto())
                )
                assert len(resp.models) == 1
                assert resp.models[0].name == model_id

                # Make sure an appropriate error is raised for trying to deploy
                # the same model again
                with pytest.raises(grpc.RpcError) as excinfo:
                    mm_stub.DeployModel(deploy_req.to_proto())
                assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS

                # Undeploy the model
                undeploy_req = UndeployModelRequest(model_id).to_proto()
                resp = mm_stub.UndeployModel(undeploy_req)
                assert resp.model_id

                # Make sure undeploying a second time is NOT_FOUND
                with pytest.raises(grpc.RpcError) as excinfo:
                    mm_stub.UndeployModel(undeploy_req)
                assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


# Test implementation details #########################


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
