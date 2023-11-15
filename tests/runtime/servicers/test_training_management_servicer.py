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
from unittest import mock
import concurrent.futures
import datetime
import re
import threading
import time

# Third Party
import grpc
import pytest

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model import DataStream, TrainingStatus
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.model_management.model_trainer_base import TrainingInfo
from caikit.interfaces.runtime.data_model import (
    TrainingInfoRequest,
    TrainingStatusResponse,
)
from caikit.runtime.servicers.training_management_servicer import (
    TrainingManagementServicerImpl,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from sample_lib import SampleModule


@pytest.fixture
def training_pool():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        yield executor


@pytest.fixture
def training_management_servicer():
    return TrainingManagementServicerImpl()


class MockModelFuture:
    """Mocks up a model future that will return a 404 error after being cancelled"""

    def __init__(self) -> None:
        self._canceled = False

    def get_info(self) -> TrainingInfo:
        if not self._canceled:
            return TrainingInfo(status=TrainingStatus.RUNNING)
        else:
            raise CaikitCoreException(
                status_code=CaikitCoreStatusCode.NOT_FOUND, message="Training not found"
            )

    def cancel(self):
        self._canceled = True


def test_training_runs(training_management_servicer, training_pool):

    # Create a future and set it in the training manager
    event = threading.Event()
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        wait_event=event,
    )

    # send a request, check it's running
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.state == TrainingStatus.RUNNING.value

    event.set()
    model_future.wait()

    # Ensure it's now done
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.state == TrainingStatus.COMPLETED.value


def test_training_timestamps(training_management_servicer, training_pool):
    """Check that the submission and completion timestamps are set correctly from the model
    future."""
    # Create a future and set it in the training manager
    event = threading.Event()
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        wait_event=event,
    )

    # send a request, check that the submission timestamp was set
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    response_dm: TrainingStatusResponse = TrainingStatusResponse.from_proto(response)
    assert response_dm.submission_timestamp is not None
    assert (
        datetime.datetime.now() - response_dm.submission_timestamp
        < datetime.timedelta(milliseconds=50)
    )
    assert response_dm.completion_timestamp is None

    event.set()
    # Wait for training to complete, but _don't_ interact with the future.
    # This simulates async training on the runtime
    for i in range(1000):
        response = training_management_servicer.GetTrainingStatus(request, context=None)
        if response.state == TrainingStatus.COMPLETED.value:
            break

    # Ensure completion timestamp is set
    response_dm: TrainingStatusResponse = TrainingStatusResponse.from_proto(response)
    assert response_dm.completion_timestamp is not None
    assert (
        datetime.datetime.now() - response_dm.completion_timestamp
        < datetime.timedelta(milliseconds=50)
    )
    assert response_dm.completion_timestamp > response_dm.submission_timestamp


def test_training_cannot_cancel_on_completed_training(training_management_servicer):
    # Create a future and set it in the training manager
    event = threading.Event()
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        wait_event=event,
    )

    # send a request, check it's not errored
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.state != TrainingStatus.ERRORED.value

    event.set()
    model_future.wait()

    # cancel the training request, but this won't change its status as it was finished
    training_management_servicer.CancelTraining(request, context=None)
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.state == TrainingStatus.COMPLETED.value


def test_training_cancel_on_correct_id(training_management_servicer):
    # Create a training future for first model with a long runtime
    # NOTE: We cannot use a single log time.sleep or a threading.Event().wait
    #   here because those are not actually destroyable via DestroyableThread.
    #   Instead, we use the sleep_time argument that runs a loop of short sleeps
    #   and then we verify below that the training duration was far less than
    #   this configured sleep time.
    start_event = threading.Event()
    model_future_1 = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        sleep_time=100,
        start_event=start_event,
    )

    # Wait until the training has started to ensure it is interrupted in flight
    start_event.wait()

    request_1 = TrainingInfoRequest(training_id=model_future_1.id).to_proto()
    response_1 = training_management_servicer.GetTrainingStatus(request_1, context=None)
    assert response_1.state == TrainingStatus.RUNNING.value

    # Model 2 has no wait event, should proceed to complete training
    model_future_2 = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([1, 2, 3]),
    )

    # Cancel first training
    request_1 = TrainingInfoRequest(training_id=model_future_1.id).to_proto()
    training_management_servicer.CancelTraining(request_1, context=None)

    response_1 = training_management_servicer.GetTrainingStatus(request_1, context=None)
    assert response_1.state == TrainingStatus.CANCELED.value

    # Make sure the model future completes without the blocking event being set
    start_time = time.time()
    model_future_1.wait()
    # Sanity check that the model did not wait for anywhere close to the full
    # 100 seconds
    wait_time = time.time() - start_time
    assert wait_time < 5
    assert (
        training_management_servicer.GetTrainingStatus(request_1, context=None).state
        == TrainingStatus.CANCELED.value
    )

    # training number 2 should still complete
    model_future_2.wait()
    request_2 = TrainingInfoRequest(training_id=model_future_2.id).to_proto()
    response_2 = training_management_servicer.GetTrainingStatus(request_2, context=None)
    assert response_2.state == TrainingStatus.COMPLETED.value


def test_training_cancel_on_mock_model_future(training_management_servicer):
    # Patch in our mock model future
    with mock.patch.object(MODEL_MANAGER, "get_model_future") as mock_gmf:
        mock_gmf.return_value = MockModelFuture()

        # Check that we get "running" status
        info_request = TrainingInfoRequest(training_id="anything").to_proto()
        info_response = training_management_servicer.GetTrainingStatus(
            info_request, context=None
        )
        assert info_response.state == TrainingStatus.RUNNING.value

        # Make sure a cancel returns "canceled"
        cancel_response = training_management_servicer.CancelTraining(
            info_request, context=None
        )
        assert cancel_response.state == TrainingStatus.CANCELED.value


def test_training_complete_status(training_management_servicer, training_pool):

    # Create a future and set it in the training manager
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
    )

    # Block until train is done
    model_future.wait()

    # send a request
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)

    assert re.match("<class '.*TrainingStatusResponse'>", str(type(response)))
    assert response.state == TrainingStatus.COMPLETED.value


def test_training_status_incorrect_id(training_management_servicer):
    # send a request with a training id that doesn't exist
    request = TrainingInfoRequest(training_id="some_random_id").to_proto()
    with pytest.raises(CaikitRuntimeException) as context:
        training_management_servicer.GetTrainingStatus(request, context=None)

    assert context.value.status_code == grpc.StatusCode.NOT_FOUND
    assert (
        f"Unknown training_id: some_random_id" in context.value.message
    )  # message set by local_model_trainer.get_model_future


def test_training_raises_when_cancel_on_incorrect_id(training_management_servicer):
    # Create a future and set it in the training manager
    event = threading.Event()
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        wait_event=event,
    )

    # send a request, check it's not errored
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.state != TrainingStatus.ERRORED.value

    event.set()
    model_future.wait()

    # cancel a training with wrong training id
    cancel_request = TrainingInfoRequest(training_id="some_random_id").to_proto()
    with pytest.raises(CaikitRuntimeException) as context:
        training_management_servicer.CancelTraining(cancel_request, context=None)

    assert context.value.status_code == grpc.StatusCode.NOT_FOUND
    assert (
        f"Unknown training_id: some_random_id" in context.value.message
    )  # message set by local_model_trainer.get_model_future


def test_training_fails(training_management_servicer, training_pool):
    model_future = MODEL_MANAGER.train(
        SampleModule,
        DataStream.from_iterable([]),
        batch_size=SampleModule.POISON_PILL_BATCH_SIZE,
    )
    model_future.wait()

    # send a request
    request = TrainingInfoRequest(training_id=model_future.id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)

    assert response.state == TrainingStatus.ERRORED.value
