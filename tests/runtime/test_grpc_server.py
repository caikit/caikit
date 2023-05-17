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
import threading
import time
import uuid

# Third Party
from grpc_health.v1 import health_pb2, health_pb2_grpc
import grpc
import pytest
import tls_test_tools

# First Party
import alog

# Local
from caikit import get_config
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingInfoResponse,
    TrainingJob,
    TrainingStatus,
)
from caikit.runtime.grpc_server import RuntimeGRPCServer
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.model_management.training_manager import TrainingManager
from caikit.runtime.protobufs import (
    model_runtime_pb2,
    model_runtime_pb2_grpc,
    process_pb2,
    process_pb2_grpc,
)
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from sample_lib.data_model import (
    OtherOutputType,
    SampleInputType,
    SampleOutputType,
    SampleTrainingType,
)
from tests.conftest import random_test_id, temp_config
from tests.fixtures import Fixtures
import caikit
import sample_lib

log = alog.use_channel("TEST-SERVE-I")

HAPPY_PATH_INPUT = SampleInputType(name="Gabe").to_proto()
HAPPY_PATH_RESPONSE = SampleOutputType(greeting="Hello Gabe").to_proto()
HAPPY_PATH_TRAIN_RESPONSE = TrainingJob(
    model_name="dummy name", training_id="dummy id"
).to_proto()


def is_good_train_response(actual_response, expected, model_name):
    assert dir(actual_response) == dir(expected)
    assert actual_response.training_id is not None
    assert isinstance(actual_response.training_id, str)
    assert actual_response.model_name == model_name


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
        training_output_dir=os.path.join("test", "training_output", training_id),
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
            ServicePackageFactory.ServiceSource.GENERATED,
        )
    )

    training_management_stub = training_management_service.stub_class(
        runtime_grpc_server.make_local_channel()
    )

    # MT training is sync, so it should be COMPLETE immediately
    response: TrainingInfoResponse = TrainingInfoResponse.from_proto(
        training_management_stub.GetTrainingStatus(training_info_request)
    )
    assert response.status == TrainingStatus.COMPLETED.value

    # Make sure we wait for training to finish
    result = TrainingManager.get_instance().training_futures[training_id].result()

    assert (
        result.MODULE_CLASS
        == "sample_lib.modules.sample_task.sample_implementation.SampleModule"
    )
    # Fields with defaults have expected values
    assert result.batch_size == 64
    assert result.learning_rate == 0.0015


def test_predict_fake_module_ok_response(
    loaded_model_id, runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.WidgetPredict successful response"""
    stub = sample_inference_service.stub_class(runtime_grpc_server.make_local_channel())
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    actual_response = stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", loaded_model_id)]
    )
    assert actual_response == HAPPY_PATH_RESPONSE


def test_predict_fake_module_error_response(
    runtime_grpc_server, sample_inference_service
):
    """Test RPC CaikitRuntime.WidgetPredict error response"""
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


####### End-to-end tests for train a model and then predict with it
def test_train_fake_module_ok_response_and_can_predict_with_trained_model(
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
):
    """Test RPC CaikitRuntime.ModulesSampleTaskSampleModuleTrain successful response"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    ).to_proto()
    model_name = random_test_id()
    train_request = (
        sample_train_service.messages.ModulesSampleTaskSampleModuleTrainRequest(
            model_name=model_name, training_data=training_data
        )
    )
    actual_response = train_stub.ModulesSampleTaskSampleModuleTrain(train_request)
    is_good_train_response(actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name)

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_ok_response_with_loaded_model_can_predict_with_trained_model(
    loaded_model_id,
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
):
    """Test RPC CaikitRuntime.WorkflowsSampleTaskSampleWorkflowTrain successful response with a loaded model"""
    sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
        model_id=loaded_model_id
    ).to_proto()
    model_name = random_test_id()
    train_request = (
        sample_train_service.messages.ModulesSampleTaskCompositeModuleTrainRequest(
            model_name=model_name, sample_block=sample_model
        )
    )
    actual_response = train_stub.ModulesSampleTaskCompositeModuleTrain(train_request)
    is_good_train_response(actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name)

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_does_not_change_another_instance_model_of_block(
    other_loaded_model_id,
    sample_int_file,
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
):
    """This test: original "stock" OtherModule model has batch size 42 (see fixtures/models/bar.yml),
    we then train a custom OtherModule model with batch size 100,
    then we make a predict to each, they should have the correct batch size"""

    # Train an OtherModule with batch size 100
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceInt
    training_data = stream_type(
        file=stream_type.File(filename=sample_int_file)
    ).to_proto()

    train_request = (
        sample_train_service.messages.ModulesOtherTaskOtherModuleTrainRequest(
            model_name="Bar Training", batch_size=100, training_data=training_data
        )
    )
    actual_response = train_stub.ModulesOtherTaskOtherModuleTrain(train_request)
    is_good_train_response(actual_response, HAPPY_PATH_TRAIN_RESPONSE, "Bar Training")

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

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
        predict_request, metadata=[("mm-model-id", other_loaded_model_id)]
    )
    expected_original_inference_response = OtherOutputType(
        farewell="goodbye: Gabe 42 times"
    ).to_proto()
    assert original_inference_response == expected_original_inference_response


##### Test different datastream types #####
def test_train_fake_module_ok_response_with_datastream_jsondata(
    train_stub, inference_stub, sample_train_service, sample_inference_service
):
    """Test RPC CaikitRuntime.ModulesSampleTaskSampleModuleTrainRequest successful response with training data json type"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(
            data=[SampleTrainingType(1), SampleTrainingType(2)]
        )
    ).to_proto()
    model_name = random_test_id()
    train_request = (
        sample_train_service.messages.ModulesSampleTaskSampleModuleTrainRequest(
            model_name=model_name,
            batch_size=42,
            training_data=training_data,
        )
    )

    actual_response = train_stub.ModulesSampleTaskSampleModuleTrain(train_request)
    is_good_train_response(actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name)

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

    # make sure the trained model can run inference
    predict_request = sample_inference_service.messages.SampleTaskRequest(
        sample_input=HAPPY_PATH_INPUT
    )
    inference_response = inference_stub.SampleTaskPredict(
        predict_request, metadata=[("mm-model-id", actual_response.model_name)]
    )
    assert inference_response == HAPPY_PATH_RESPONSE


def test_train_fake_module_ok_response_with_datastream_csv_file(
    train_stub,
    inference_stub,
    sample_train_service,
    sample_inference_service,
    sample_csv_file,
):
    """Test RPC CaikitRuntime.ModulesSampleTaskSampleModuleTrainRequest successful response with training data file type"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        file=stream_type.File(filename=sample_csv_file)
    ).to_proto()
    model_name = random_test_id()
    train_request = (
        sample_train_service.messages.ModulesSampleTaskSampleModuleTrainRequest(
            model_name=model_name,
            training_data=training_data,
        )
    )

    actual_response = train_stub.ModulesSampleTaskSampleModuleTrain(train_request)
    is_good_train_response(actual_response, HAPPY_PATH_TRAIN_RESPONSE, model_name)

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

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
    """Test RPC CaikitRuntime.WorkflowsSampleTaskSampleWorkflowTrain error response because sample model is not loaded"""
    with pytest.raises(grpc.RpcError) as context:
        sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
            model_id=random_test_id()
        ).to_proto()

        train_request = (
            sample_train_service.messages.ModulesSampleTaskCompositeModuleTrainRequest(
                model_name=random_test_id(), sample_block=sample_model
            )
        )
        train_stub.ModulesSampleTaskCompositeModuleTrain(train_request)
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


def test_unload_model_ok_response(loaded_model_id, runtime_grpc_server):
    """Test unload model's successful response"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    unload_model_request = model_runtime_pb2.UnloadModelRequest(modelId=loaded_model_id)
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
    loaded_model_id, runtime_grpc_server
):
    """Test predict model size successful response on a model that has been loaded"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )
    predict_model_size_request = model_runtime_pb2.PredictModelSizeRequest(
        modelId=loaded_model_id, modelPath=Fixtures.get_good_model_path()
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


def test_model_size_ok_response(loaded_model_id, runtime_grpc_server):
    """Test model size successful response"""
    stub = model_runtime_pb2_grpc.ModelRuntimeStub(
        runtime_grpc_server.make_local_channel()
    )

    model_size_request = model_runtime_pb2.ModelSizeRequest(modelId=loaded_model_id)
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

    def never_return(*args):
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


def test_tls(sample_inference_service):
    """Boot up a server with TLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert="", key="")
    )
    with RuntimeGRPCServer(
        inference_service=sample_inference_service,
        training_service=None,
        tls_config_override=tls_config,
    ) as server:
        _assert_connection(_make_secure_channel(server, ca_cert))


def test_mtls(sample_inference_service):
    """Boot up a server with mTLS enabled and ping it on a secure channel"""
    ca_key = tls_test_tools.generate_key()[0]
    ca_cert = tls_test_tools.generate_ca_cert(ca_key)
    tls_key, tls_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)

    tls_config = TLSConfig(
        server=KeyPair(cert=tls_cert, key=tls_key), client=KeyPair(cert=ca_cert, key="")
    )
    with RuntimeGRPCServer(
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


def test_certs_can_be_loaded_as_files(sample_inference_service, tmp_path):
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
    with RuntimeGRPCServer(
        inference_service=sample_inference_service,
        training_service=None,
        tls_config_override=tls_config,
    ) as server:
        client_key, client_cert = tls_test_tools.generate_derived_key_cert_pair(ca_key)
        _assert_connection(
            _make_secure_channel(server, ca_cert, client_key, client_cert)
        )


def test_metrics_stored_after_server_interrupt(
    loaded_model_id, sample_inference_service
):
    """This tests the gRPC server's behaviour when interrupted"""

    with RuntimeGRPCServer(
        inference_service=sample_inference_service,
        training_service=None,
    ) as server:
        stub = sample_inference_service.stub_class(server.make_local_channel())
        predict_request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        )
        _ = stub.SampleTaskPredict(
            predict_request, metadata=[("mm-model-id", loaded_model_id)]
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


def test_out_of_range_port(sample_inference_service):
    """Test that the server can use a port outside of the default 8888-9000
    range
    """
    free_high_port = RuntimeGRPCServer._find_port(50000, 60000)
    with temp_config(
        {
            "runtime": {
                "port": free_high_port,
                "find_available_port": False,
            }
        },
        merge_strategy="merge",
    ):
        with RuntimeGRPCServer(
            inference_service=sample_inference_service,
            training_service=None,
        ) as server:
            _assert_connection(grpc.insecure_channel(f"localhost:{free_high_port}"))


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


def _assert_connection(channel):
    """Check that we can ping the server on this channel.

    Assumes that it will come up, but it's hard to distinguish between a failure to boot and a
    failure of TLS communication.
    """
    done = False
    failures = 0
    while not done:
        try:
            stub = health_pb2_grpc.HealthStub(channel)
            health_check_request = health_pb2.HealthCheckRequest()
            stub.Check(health_check_request)
            done = True
        except grpc.RpcError as e:
            log.debug(
                f"[RpcError] {e}; will try to reconnect to test server in 0.1 second."
            )

            time.sleep(0.1)
            failures += 1
            if failures > 20:
                raise e
