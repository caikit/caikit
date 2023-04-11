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
from tempfile import TemporaryDirectory
import time
import uuid

# Third Party
from sample_lib.blocks.sample_task.sample_implementation import SampleBlock
from sample_lib.data_model.sample import (
    SampleInputType,
    SampleOutputType,
    SampleTrainingType,
)
import pytest

# Local
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import temp_config_parser
from tests.fixtures import Fixtures
from tests.fixtures.protobufs import primitive_party_pb2
import caikit.core

## Helpers #####################################################################


def _random_test_id():
    return "test-any-model-" + str(uuid.uuid4())


def _random_training_id():
    return "training-" + str(uuid.uuid4())


def _primitives_function(
    self,
    double_field,
    float_field,
    int64_field,
    uint64_field,
    int32_field,
    fixed64_field,
    fixed32_field,
    bool_field,
    string_field,
    bytes_field,
    uint32_field,
    sfixed32_field,
    sfixed64_field,
    sint32_field,
    sint64_field,
):
    pass


HAPPY_PATH_RESPONSE = caikit.interfaces.runtime.data_model.TrainingJob(
    model_name="Foo Bar Training",
    training_id=_random_training_id(),
)


@pytest.fixture(autouse=True, params=[True, False])
def set_train_location(request):
    """This fixture ensures that all tests in this file will be run with both
    subprocess and local training styles
    """
    with temp_config_parser({"training": {"use_subprocess": request.param}}):
        yield


# Train tests for the GlobalTrainServicer class ############################################################
##############
# Normal cases
##############


def test_global_train_simple_Sample_Widget(
    sample_train_service,
    sample_train_servicer,
    sample_inference_service,
    sample_predict_servicer,
):
    """Global train of TrainRequest returns a training job with the correct
    model name, and some training id for a basic train function that doesn't
    require any loaded model
    """
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    ).to_proto()
    train_request = (
        sample_train_service.messages.BlocksSampleTaskSampleBlockTrainRequest(
            model_name="Foo Bar Training",
            batch_size=42,
            training_data=training_data,
        )
    )

    training_response = sample_train_servicer.Train(train_request)
    assert training_response.model_name == "Foo Bar Training"
    assert training_response.training_id is not None
    assert isinstance(training_response.training_id, str)

    result = sample_train_servicer.training_map.get(
        training_response.training_id
    ).result()
    assert result.batch_size == 42
    assert (
        result.BLOCK_CLASS
        == "sample_lib.blocks.sample_task.sample_implementation.SampleBlock"
    )

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

    inference_response = sample_predict_servicer.Predict(
        sample_inference_service.messages.SampleTaskRequest(
            sample_input=SampleInputType(name="Gabe").to_proto()
        ),
        Fixtures.build_context(training_response.model_name),
    )
    assert (
        inference_response
        == SampleOutputType(
            greeting="Hello Gabe",
        ).to_proto()
    )


def test_global_train_Another_Widget_that_requires_SampleWidget_loaded_should_not_raise(
    loaded_model_id,
    sample_train_service,
    sample_train_servicer,
    sample_inference_service,
    sample_predict_servicer,
):
    """Global train of TrainRequest returns a training job with the correct model name, and some training id for a train function that requires another loaded model"""
    sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
        model_id=loaded_model_id
    ).to_proto()

    training_request = (
        sample_train_service.messages.WorkflowsSampleTaskSampleWorkflowTrainRequest(
            model_name="AnotherWidget_Training",
            sample_block=sample_model,
        )
    )

    training_response = sample_train_servicer.Train(training_request)

    assert training_response.model_name == "AnotherWidget_Training"
    assert training_response.training_id is not None
    assert isinstance(training_response.training_id, str)

    training_result = sample_train_servicer.training_map.get(
        training_response.training_id
    ).result()

    assert (
        training_result.WORKFLOW_CLASS
        == "sample_lib.workflows.sample_task.sample_implementation.SampleWorkflow"
    )

    # give the trained model time to load
    # TODO: no sleeps in tests!
    time.sleep(1)

    # make sure the trained model can run inference
    inference_response = sample_predict_servicer.Predict(
        sample_inference_service.messages.SampleTaskRequest(
            sample_input=SampleInputType(name="Gabe").to_proto()
        ),
        Fixtures.build_context(training_response.model_name),
    )
    assert (
        inference_response
        == SampleOutputType(
            greeting="Hello Gabe",
        ).to_proto()
    )


def test_run_train_job_works_with_wait(
    sample_train_service, sample_inference_service, sample_predict_servicer
):
    """Check if run_train_job works as expected for syncronous requests"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    ).to_proto()
    train_request = (
        sample_train_service.messages.BlocksSampleTaskSampleBlockTrainRequest(
            model_name="Foo Bar Training",
            batch_size=42,
            training_data=training_data,
        )
    )
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    with TemporaryDirectory() as tmp_dir:
        training_response = servicer.run_training_job(
            train_request,
            SampleBlock,
            training_id="dummy-training-id",
            training_output_dir=tmp_dir,
            wait=True,
        )

        assert training_response.training_id == "dummy-training-id"

        inference_response = sample_predict_servicer.Predict(
            sample_inference_service.messages.SampleTaskRequest(
                sample_input=SampleInputType(name="Test").to_proto()
            ),
            Fixtures.build_context(training_response.model_name),
        )
        assert (
            inference_response
            == SampleOutputType(
                greeting="Hello Test",
            ).to_proto()
        )


def test_run_train_job_works_with_no_autoload(sample_train_service):
    """Check if run_train_job works with no auto-load doesn't load model"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    ).to_proto()
    train_request = (
        sample_train_service.messages.BlocksSampleTaskSampleBlockTrainRequest(
            model_name=str(uuid.uuid4()),
            batch_size=42,
            training_data=training_data,
        )
    )
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    servicer.auto_load_trained_model = False
    init_loaded_models_count = len(servicer._model_manager.loaded_models)

    with TemporaryDirectory() as tmp_dir:
        training_response = servicer.run_training_job(
            train_request,
            SampleBlock,
            training_id="dummy-training-id",
            training_output_dir=tmp_dir,
            wait=True,
        )
        assert training_response.training_id == "dummy-training-id"
        assert init_loaded_models_count == len(servicer._model_manager.loaded_models)


def test_run_train_job_works_with_autoload(sample_train_service):
    """Check if run_train_job works with auto-load loading the model"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    ).to_proto()
    train_request = (
        sample_train_service.messages.BlocksSampleTaskSampleBlockTrainRequest(
            model_name=str(uuid.uuid4()),
            batch_size=42,
            training_data=training_data,
        )
    )
    servicer = GlobalTrainServicer(training_service=sample_train_service)
    servicer.auto_load_trained_model = True
    init_loaded_models_count = len(servicer._model_manager.loaded_models)

    with TemporaryDirectory() as tmp_dir:
        training_response = servicer.run_training_job(
            train_request,
            SampleBlock,
            training_id="dummy-training-id-2",
            training_output_dir=tmp_dir,
            wait=True,
        )
        assert training_response.training_id == "dummy-training-id-2"
        assert init_loaded_models_count < len(servicer._model_manager.loaded_models)


#############
# Error cases
#############


def test_global_train_Another_Widget_that_requires_SampleWidget_but_not_loaded_should_raise(
    sample_train_service, sample_train_servicer
):
    """Global train of TrainRequest raises when calling a train function that requires another loaded model, but model is not loaded"""
    model_id = _random_test_id()

    sample_model = caikit.interfaces.runtime.data_model.ModelPointer(
        model_id=model_id
    ).to_proto()
    request = (
        sample_train_service.messages.WorkflowsSampleTaskSampleWorkflowTrainRequest(
            model_name="AnotherWidget_Training",
            sample_block=sample_model,
        )
    )

    with pytest.raises(CaikitRuntimeException) as context:
        sample_train_servicer.Train(request)

    assert f"Model '{model_id}' not loaded" == context.value.message


def test_global_train_Edge_Case_Widget_should_raise_when_error_surfaces_from_block(
    sample_train_service, sample_train_servicer
):
    """Test that if a block raises a ValueError, we should surface it to the user in a helpful way"""
    stream_type = caikit.interfaces.common.data_model.DataStreamSourceSampleTrainingType
    training_data = stream_type(
        jsondata=stream_type.JsonData(data=[SampleTrainingType(1)])
    ).to_proto()
    train_request = (
        sample_train_service.messages.BlocksSampleTaskSampleBlockTrainRequest(
            model_name="Foo Bar Training",
            batch_size=999,
            training_data=training_data,
        )
    )
    with pytest.raises(CaikitRuntimeException) as context:
        training_response = sample_train_servicer.Train(train_request)

        training_result = sample_train_servicer.training_map.get(
            training_response.training_id
        ).result()

    assert f"This may be a problem with your input" in str(context.value.message)


#####################################################################

# NOTE: This test was commented out in the original unittest.TestCase impl - leaving as is
# def test_global_train_aborts_long_running_trains(self):
#     mock_manager = MagicMock()

#     # return a dummy model from the mock model manager
#     class UnresponsiveModel:
#         started = threading.Event()

#         def run(self, *args, **kwargs):
#             self.started.set()
#             while True:
#                 time.sleep(0.01)

#     dummy_model = UnresponsiveModel()
#     mock_manager.retrieve_model.return_value = dummy_model

#     context = Fixtures.build_context("test-any-unresponsive-model")
#     train_thread = threading.Thread(
#         target=self.train_servicer.Train,
#         args=(self.HAPPY_PATH_FAKE_BLOCK_REQUEST, context),
#     )

#     # Patch in the mock manager and start the training
#     with patch.object(self.train_servicer, "_model_manager", mock_manager):
#         train_thread.start()
#         dummy_model.started.wait()
#         # Simulate a timeout or client abort
#         context.cancel()
#         train_thread.join(10)

#     # Make sure the training job actually stopped
#     self.assertFalse(train_thread.is_alive())
