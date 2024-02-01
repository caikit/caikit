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
from typing import Iterator
from unittest.mock import MagicMock, patch

# Local
from caikit.core.data_model import DataStream, ProducerId
from caikit.runtime.service_factory import get_inference_request
from sample_lib.data_model.sample import GeoSpatialTask
from sample_lib.modules import MultiTaskModule, SecondTask
from sample_lib.modules.geospatial import GeoStreamingModule

try:
    # Standard
    from test.support.threading_helper import catch_threading_exception
except (NameError, ModuleNotFoundError):
    from tests.base import catch_threading_exception

# Standard
import json
import threading
import time

# Third Party
import grpc
import pytest

# Local
from caikit.interfaces.common.data_model import File
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.types.aborted_exception import AbortedException
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from sample_lib.data_model import SampleInputType, SampleOutputType
from sample_lib.data_model.sample import OtherOutputType, SampleTask
from sample_lib.modules.sample_task import SampleModule
from tests.conftest import temp_config
from tests.fixtures import Fixtures

HAPPY_PATH_INPUT_DM = SampleInputType(name="Gabe")
HAPPY_PATH_RESPONSE_DM = SampleOutputType(greeting="Hello Gabe")
HAPPY_PATH_INPUT = HAPPY_PATH_INPUT_DM.to_proto()
HAPPY_PATH_RESPONSE = HAPPY_PATH_RESPONSE_DM.to_proto()


def test_calling_predict_should_raise_if_module_raises(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    with pytest.raises(CaikitRuntimeException) as context:
        # SampleModules will raise a RuntimeError if the throw flag is set
        predict_class = get_inference_request(SampleTask)
        request = predict_class(sample_input=HAPPY_PATH_INPUT_DM, throw=True).to_proto()
        sample_predict_servicer.Predict(
            request,
            Fixtures.build_context(sample_task_model_id),
            caikit_rpc=sample_task_unary_rpc,
        )
    assert context.value.status_code == grpc.StatusCode.INTERNAL


def test_predict_raises_with_grpc_errors(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    with pytest.raises(CaikitRuntimeException) as context:
        # SampleModules will raise a RuntimeError if the throw flag is set
        predict_class = get_inference_request(SampleTask)
        request = predict_class(
            sample_input=HAPPY_PATH_INPUT_DM,
            throw=True,
            error="GRPC_RESOURCE_EXHAUSTED",
        ).to_proto()
        sample_predict_servicer.Predict(
            request,
            Fixtures.build_context(sample_task_model_id),
            caikit_rpc=sample_task_unary_rpc,
        )
    assert context.value.status_code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert "Model is overloaded" in context.value.message


def test_invalid_input_to_a_valid_caikit_core_class_method_raises(
    sample_task_model_id,
    sample_inference_service,
    sample_predict_servicer,
    sample_task_unary_rpc,
):
    """Test that a caikit.core module that gets an unexpected input value provides an error in an expected way"""
    with pytest.raises(CaikitRuntimeException) as context:
        # SampleModules will raise a ValueError if the poison pill name is given
        predict_class = get_inference_request(SampleTask)
        request = predict_class(
            sample_input=SampleInputType(name=SampleModule.POISON_PILL_NAME)
        ).to_proto()
        sample_predict_servicer.Predict(
            request,
            Fixtures.build_context(sample_task_model_id),
            caikit_rpc=sample_task_unary_rpc,
        )
    assert context.value.status_code == grpc.StatusCode.INVALID_ARGUMENT


def test_global_predict_works_for_unary_rpcs(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    """Global predict of SampleTaskRequest returns a prediction"""
    predict_class = get_inference_request(SampleTask)
    response = sample_predict_servicer.Predict(
        predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto(),
        Fixtures.build_context(sample_task_model_id),
        caikit_rpc=sample_task_unary_rpc,
    )
    assert response == HAPPY_PATH_RESPONSE


def test_global_predict_explicit_inference_function(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    """Calling predict_model can explicitly select the inference function"""
    predict_class = get_inference_request(SampleTask)
    request_name = sample_task_unary_rpc.request.name
    response = sample_predict_servicer.predict_model(
        request_name=request_name,
        model_id=sample_task_model_id,
        inference_func_name="run_stream_out",
        sample_input=HAPPY_PATH_INPUT_DM,
    )
    assert isinstance(response, DataStream)


def test_global_predict_works_on_bidirectional_streaming_rpcs(
    sample_inference_service, sample_predict_servicer, sample_task_model_id
):
    """Simple test that our SampleModule's bidirectional stream inference fn is supported"""

    predict_class = get_inference_request(
        SampleTask, input_streaming=True, output_streaming=True
    )

    def req_iterator() -> Iterator[predict_class]:
        for i in range(100):
            yield predict_class(sample_inputs=HAPPY_PATH_INPUT_DM).to_proto()

    response_stream = sample_predict_servicer.Predict(
        req_iterator(),
        Fixtures.build_context(sample_task_model_id),
        caikit_rpc=sample_inference_service.caikit_rpcs[
            "BidiStreamingSampleTaskPredict"
        ],
    )
    count = 0
    for response in response_stream:
        assert response == HAPPY_PATH_RESPONSE
        count += 1
    assert count == 100


def test_global_predict_works_on_bidirectional_streaming_rpcs_with_multiple_streaming_parameters(
    sample_inference_service, sample_predict_servicer, sample_task_model_id
):
    """Test that our little geospatial model that takes multiple streams is supported"""

    mock_manager = MagicMock()
    mock_manager.retrieve_model.return_value = GeoStreamingModule()

    predict_class = get_inference_request(
        GeoSpatialTask, input_streaming=True, output_streaming=True
    )

    def req_iterator() -> Iterator[predict_class]:
        for i in range(100):
            yield predict_class(lats=i, lons=100 - i, name="Gabe").to_proto()

    with patch.object(sample_predict_servicer, "_model_manager", mock_manager):
        response_stream = sample_predict_servicer.Predict(
            req_iterator(),
            Fixtures.build_context(sample_task_model_id),
            caikit_rpc=sample_inference_service.caikit_rpcs[
                "BidiStreamingGeoSpatialTaskPredict"
            ],
        )
        count = 0
        for i, response in enumerate(response_stream):
            assert response.greeting == f"Hello from Gabe at {i}.0°, {100-i}.0°"
            count += 1
        assert count == 100


def test_global_predict_works_for_multitask_model(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_model_id,
):
    mock_manager = MagicMock()
    mock_manager.retrieve_model.return_value = MultiTaskModule()

    predict_class = get_inference_request(
        SecondTask, input_streaming=False, output_streaming=False
    )
    with patch.object(sample_predict_servicer, "_model_manager", mock_manager):
        response = sample_predict_servicer.Predict(
            predict_class(
                file_input=File(filename="foo", data=bytes("bar", "utf-8"))
            ).to_proto(),
            Fixtures.build_context(sample_task_model_id),
            caikit_rpc=sample_inference_service.caikit_rpcs["SecondTaskPredict"],
        )

    assert (
        response
        == OtherOutputType(
            "Goodbye from SecondTask", ProducerId("MultiTaskModule", "0.0.1")
        ).to_proto()
    )


def test_global_predict_predict_model_direct(
    sample_inference_service, sample_predict_servicer, sample_task_model_id
):
    """Test that the direct predict_model function can be called to bypass the
    gRPC handler
    """
    response = sample_predict_servicer.predict_model(
        request_name="SampleTaskRequest",
        model_id=sample_task_model_id,
        sample_input=HAPPY_PATH_INPUT_DM,
    )
    assert response == HAPPY_PATH_RESPONSE_DM


def test_global_predict_aborts_long_running_predicts(
    sample_inference_service,
    sample_predict_servicer,
    sample_task_unary_rpc,
):
    mock_manager = MagicMock()

    # return a dummy model from the mock model manager
    class UnresponsiveModel(SampleModule):
        started = threading.Event()

        def run(self, *args, **kwargs):
            self.started.set()
            while True:
                time.sleep(0.001)

    dummy_model = UnresponsiveModel()
    mock_manager.retrieve_model.return_value = dummy_model

    context = Fixtures.build_context("test-any-unresponsive-model")
    predict_class = get_inference_request(SampleTask)
    predict_thread = threading.Thread(
        target=sample_predict_servicer.Predict,
        args=(
            predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto(),
            context,
        ),
        kwargs={"caikit_rpc": sample_task_unary_rpc},
    )

    with catch_threading_exception() as cm:
        # Patch in the mock manager and start the prediction
        with patch.object(sample_predict_servicer, "_model_manager", mock_manager):
            predict_thread.start()
            assert dummy_model.started.wait(2)
            # Simulate a timeout or client abort
            context.cancel()
            predict_thread.join(2)

        # Make sure the prediction actually stopped
        assert not predict_thread.is_alive()

        # Make sure the correct exception was raised
        assert cm.exc_type == AbortedException


def test_metering_ignore_unsuccessful_calls(
    sample_inference_service,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    with temp_config({"runtime": {"metering": {"enabled": True}}}, "merge"):
        gps = GlobalPredictServicer(sample_inference_service)
        try:
            with patch.object(gps.rpc_meter, "update_metrics") as mock_update_func:
                predict_class = get_inference_request(SampleTask)
                request = predict_class(
                    sample_input=HAPPY_PATH_INPUT_DM, throw=True
                ).to_proto()
                with pytest.raises(CaikitRuntimeException):
                    gps.Predict(
                        request,
                        Fixtures.build_context(sample_task_model_id),
                        caikit_rpc=sample_task_unary_rpc,
                    )

                mock_update_func.assert_not_called()
        finally:
            gps.stop_metering()


def test_metering_predict_rpc_counter(
    sample_inference_service, sample_task_model_id, sample_task_unary_rpc
):
    # need a new servicer to get a fresh new RPC meter
    with temp_config({"runtime": {"metering": {"enabled": True}}}, "merge"):
        sample_predict_servicer = GlobalPredictServicer(sample_inference_service)
        try:
            # Making 20 requests
            predict_class = get_inference_request(SampleTask)
            for i in range(20):
                sample_predict_servicer.Predict(
                    predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto(),
                    Fixtures.build_context(sample_task_model_id),
                    caikit_rpc=sample_task_unary_rpc,
                )

            # Force meter to write
            sample_predict_servicer.rpc_meter.flush_metrics()

            # Assertions on the created metrics file
            with open(sample_predict_servicer.rpc_meter.file_path) as f:
                data = [json.loads(line) for line in f]

            assert len(data) == 1
            assert list(data[0].keys()) == [
                "timestamp",
                "batch_size",
                "model_type_counters",
                "container_id",
            ]
            assert data[0]["batch_size"] == 20
            assert len(data[0]["model_type_counters"]) == 1
            assert data[0]["model_type_counters"] == {
                "<class 'sample_lib.modules.sample_task.sample_implementation.SampleModule'>": 20
            }
        finally:
            sample_predict_servicer.stop_metering()


def test_metering_write_to_metrics_file_twice(
    sample_inference_service,
    sample_task_model_id,
    sample_task_unary_rpc,
):
    """Make sure subsequent metering events append to file"""
    with temp_config({"runtime": {"metering": {"enabled": True}}}, "merge"):
        # need a new servicer to get a fresh new RPC meter
        sample_predict_servicer = GlobalPredictServicer(sample_inference_service)
        try:
            predict_class = get_inference_request(SampleTask)
            sample_predict_servicer.Predict(
                predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto(),
                Fixtures.build_context(sample_task_model_id),
                caikit_rpc=sample_task_unary_rpc,
            )

            # Force write
            sample_predict_servicer.rpc_meter.flush_metrics()

            sample_predict_servicer.Predict(
                predict_class(sample_input=HAPPY_PATH_INPUT_DM).to_proto(),
                Fixtures.build_context(sample_task_model_id),
                caikit_rpc=sample_task_unary_rpc,
            )

            # Force write
            sample_predict_servicer.rpc_meter.flush_metrics()

            with open(sample_predict_servicer.rpc_meter.file_path) as f:
                data = [json.loads(line) for line in f]

            assert len(data) == 2
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
        finally:
            sample_predict_servicer.stop_metering()
