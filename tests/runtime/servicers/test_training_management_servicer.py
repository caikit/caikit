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
from caikit.interfaces.runtime.data_model import TrainingInfoRequest, TrainingStatus
from caikit.runtime.model_management.training_manager import TrainingManager
from caikit.runtime.servicers.training_management_servicer import (
    TrainingManagementServicerImpl,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException


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
    training_id = str(uuid.uuid4())
    training_manager = TrainingManager.get_instance()
    event = threading.Event()

    # Create a future and set it in the training manager
    future = training_pool.submit(_train, wait_event=event)
    training_manager.training_futures[training_id] = future

    # send a request, check it's processing
    request = TrainingInfoRequest(training_id=training_id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.status == TrainingStatus.PROCESSING.value

    event.set()
    future.result()

    # Ensure it's now done
    request = TrainingInfoRequest(training_id=training_id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)
    assert response.status == TrainingStatus.COMPLETED.value


def test_training_complete_status(training_management_servicer, training_pool):
    training_id = str(uuid.uuid4())
    training_manager = TrainingManager.get_instance()

    # Create a future and set it in the training manager
    future = training_pool.submit(_train)
    training_manager.training_futures[training_id] = future
    # Block until train is done
    future.result()

    # send a request
    request = TrainingInfoRequest(training_id=training_id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)

    assert re.match("<class '.*TrainingInfoResponse'>", str(type(response)))
    assert response.status == TrainingStatus.COMPLETED.value


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
    training_id = str(uuid.uuid4())
    training_manager = TrainingManager.get_instance()

    # Create a future and set it in the training manager
    future = training_pool.submit(_train, raise_=True)
    training_manager.training_futures[training_id] = future

    # send a request
    request = TrainingInfoRequest(training_id=training_id).to_proto()
    response = training_management_servicer.GetTrainingStatus(request, context=None)

    assert response.status == TrainingStatus.FAILED.value
