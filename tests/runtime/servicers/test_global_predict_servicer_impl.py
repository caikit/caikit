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
from unittest.mock import MagicMock, patch
import json
import os
import tempfile
import threading
import time
import uuid

# Third Party
from sample_lib.blocks.sample_task import SampleBlock
from sample_lib.data_model import SampleInputType, SampleOutputType
import grpc
import pytest
import sample_lib

# Local
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.fixtures import Fixtures
from tests.fixtures.protobufs import primitive_party_pb2


def _random_test_id():
    return "test-any-model-" + str(uuid.uuid4())


HAPPY_PATH_INPUT = SampleInputType(name="Gabe").to_proto()
HAPPY_PATH_RESPONSE = SampleOutputType(greeting="Hello Gabe").to_proto()


@pytest.mark.skip("We'd like to just nix the whole strict_rpc_mode check from caikit")
def test_calling_the_wrong_rpc_for_a_model_raises(
    sample_inference_service, sample_predict_servicer, loaded_model_id
):
    request = sample_inference_service.messages.BobbitRequest(
        producer_id=TEST_PRODUCER_ID
    )

    with pytest.raises(CaikitRuntimeException) as context:
        sample_predict_servicer.Predict(
            request, Fixtures.build_context(loaded_model_id)
        )

    assert context.value.status_code == grpc.StatusCode.INVALID_ARGUMENT

    assert "Wrong return type from model" in context.value.message


def test_calling_predict_should_raise_if_block_raises(
    sample_inference_service, sample_predict_servicer, loaded_model_id
):
    with pytest.raises(CaikitRuntimeException) as context:
        # SampleBlocks will raise a RuntimeError if the throw flag is set
        request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT, throw=True
        )
        sample_predict_servicer.Predict(
            request, Fixtures.build_context(loaded_model_id)
        )
    assert context.value.status_code == grpc.StatusCode.INTERNAL
    assert "Unhandled exception during prediction" in context.value.message


@pytest.mark.skip("We'd like to just nix the strict_rpc_mode checking from caikit")
def test_wrong_rpc_edge_case(sample_inference_service, sample_predict_servicer):
    """Test that a model that returns a type with a name like ${some_bad_prefix}${expected_type_name} will still
    raise an invalid argument error"""
    model_id = _random_test_id()
    model = DeprecatedWidget()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, model_id)
        model.save(model_path)
        Fixtures.load_model(model_id, model_path, Fixtures().get_good_model_type())
        with pytest.raises(CaikitRuntimeException) as context:
            sample_predict_servicer.Predict(
                sample_inference_service.messages.WidgetRequest(
                    producer_id=TEST_PRODUCER_ID
                ),
                Fixtures.build_context(model_id),
            )
        assert context.value.status_code == grpc.StatusCode.INVALID_ARGUMENT
        assert "Wrong return type from model" in context.value.message


def test_invalid_input_to_a_valid_caikit_core_class_method_raises(
    loaded_model_id, sample_inference_service, sample_predict_servicer
):
    """Test that a caikit.core block that gets an unexpected input value errors in an expected way"""
    with pytest.raises(CaikitRuntimeException) as context:
        # SampleBlocks will raise a ValueError if the poison pill name is given
        request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=SampleInputType(name=SampleBlock.POISON_PILL_NAME).to_proto(),
        )
        sample_predict_servicer.Predict(
            request, Fixtures.build_context(loaded_model_id)
        )
    assert context.value.status_code == grpc.StatusCode.INVALID_ARGUMENT
    assert "problem with your input" in context.value.message


def test_global_predict_works_on_good_inputs(
    sample_inference_service, sample_predict_servicer, loaded_model_id
):
    """Global predict of CategoriesEsaPredict returns a categories prediction"""
    response = sample_predict_servicer.Predict(
        sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT
        ),
        Fixtures.build_context(loaded_model_id),
    )
    assert response == HAPPY_PATH_RESPONSE


def test_global_predict_aborts_long_running_predicts(
    sample_inference_service, sample_predict_servicer
):
    mock_manager = MagicMock()

    # return a dummy model from the mock model manager
    class UnresponsiveModel:
        started = threading.Event()

        def run(self, *args, **kwargs):
            self.started.set()
            while True:
                time.sleep(0.01)

    dummy_model = UnresponsiveModel()
    mock_manager.retrieve_model.return_value = dummy_model

    context = Fixtures.build_context("test-any-unresponsive-model")
    predict_thread = threading.Thread(
        target=sample_predict_servicer.Predict,
        args=(
            sample_inference_service.messages.SampleTaskRequest(
                sample_input=HAPPY_PATH_INPUT
            ),
            context,
        ),
    )

    # Patch in the mock manager and start the prediction
    with patch.object(sample_predict_servicer, "_model_manager", mock_manager):
        predict_thread.start()
        dummy_model.started.wait()
        # Simulate a timeout or client abort
        context.cancel()
        predict_thread.join(10)

    # Make sure the prediction actually stopped
    assert not predict_thread.is_alive()


def test_metering_ignore_unsuccessful_calls(
    sample_inference_service, sample_predict_servicer, loaded_model_id
):
    with patch.object(
        sample_predict_servicer.rpc_meter, "update_metrics"
    ) as mock_update_func:
        request = sample_inference_service.messages.SampleTaskRequest(
            sample_input=HAPPY_PATH_INPUT, throw=True
        )
        with pytest.raises(CaikitRuntimeException):
            sample_predict_servicer.Predict(
                request, Fixtures.build_context(loaded_model_id)
            )

        mock_update_func.assert_not_called()


def test_metering_predict_rpc_counter(sample_inference_service, loaded_model_id):
    # need a new servicer to get a fresh new RPC meter
    sample_predict_servicer = GlobalPredictServicer(sample_inference_service)
    try:
        # Making 20 requests
        for i in range(20):
            sample_predict_servicer.Predict(
                sample_inference_service.messages.SampleTaskRequest(
                    sample_input=HAPPY_PATH_INPUT
                ),
                Fixtures.build_context(loaded_model_id),
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
            "<class 'sample_lib.blocks.sample_task.sample_implementation.SampleBlock'>": 20
        }
    finally:
        sample_predict_servicer.rpc_meter.end_writer_thread()


def test_metering_write_to_metrics_file_twice(
    sample_inference_service, loaded_model_id
):
    """Make sure subsequent metering events append to file"""
    # need a new servicer to get a fresh new RPC meter
    sample_predict_servicer = GlobalPredictServicer(sample_inference_service)
    try:
        sample_predict_servicer.Predict(
            sample_inference_service.messages.SampleTaskRequest(
                sample_input=HAPPY_PATH_INPUT
            ),
            Fixtures.build_context(loaded_model_id),
        )

        # Force write
        sample_predict_servicer.rpc_meter.flush_metrics()

        sample_predict_servicer.Predict(
            sample_inference_service.messages.SampleTaskRequest(
                sample_input=HAPPY_PATH_INPUT
            ),
            Fixtures.build_context(loaded_model_id),
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
            "<class 'sample_lib.blocks.sample_task.sample_implementation.SampleBlock'>": 1
        }
    finally:
        sample_predict_servicer.rpc_meter.end_writer_thread()
