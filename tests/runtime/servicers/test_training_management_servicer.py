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
import concurrent.futures
import re
import threading
import uuid

# Third Party
import grpc
import pytest

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model import DataStream, TrainingStatus
from caikit.interfaces.runtime.data_model import TrainingInfoRequest
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


def _train(raise_: bool = False, wait_event: threading.Event = None):
    if raise_:
        raise RuntimeError()
    if wait_event is not None:
        wait_event.wait()
    return "done"


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
        "some_random_id not found in the list of currently running training jobs"
        in context.value.message
    )


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
