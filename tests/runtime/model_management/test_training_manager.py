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
from threading import Timer
import concurrent.futures as futures
import threading
import unittest
import uuid

# Local
from caikit.interfaces.runtime.data_model.training_management import TrainingStatus
from caikit.runtime.model_management.training_manager import TrainingManager


def _random_training_id():
    return "training-" + str(uuid.uuid4())


class TestTrainingManager(unittest.TestCase):
    def setUp(self):
        """This method runs before each test begins to run"""
        self.training_manager = TrainingManager.get_instance()

    def test_training_manager_is_a_singleton(self):
        with self.assertRaises(Exception) as context:
            another_training_manager = TrainingManager()
        self.assertIn("This class is a singleton!", str(context.exception))

    def test_get_training_status_not_started(self):
        """Make sure that if the future has not started, it doesn't raise an
        error and reports that it has not yet started
        """

        start_event = threading.Event()

        def block_start():
            start_event.wait()

        end_event = threading.Event()

        def train_task():
            end_event.set()

        # Start the pool and block it
        T = futures.ThreadPoolExecutor(1)  # Run at most 1 function concurrently
        T.submit(block_start)

        # Start the "training"
        train_future = T.submit(train_task)
        train_id = _random_training_id()
        self.training_manager.training_futures[train_id] = train_future

        # Make sure the training is NOT_STARTED
        assert (
            self.training_manager.get_training_status(train_id)
            == TrainingStatus.NOT_STARTED
        )

        # Let the "training" proceed and make sure it completes
        start_event.set()
        end_event.wait()
