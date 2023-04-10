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
import unittest
import uuid

# Local
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

    def test_get_training_status_raises_when_a_training_future_is_cancelled_due_to_shutdown(
        self,
    ):
        """Test that a waiting but cancelled training thread raises RuntimeError"""
        training_id = _random_training_id()
        training_id2 = _random_training_id()
        timer = Timer(2000, function=None)  # a Timer that just waits a while

        T = futures.ThreadPoolExecutor(1)  # Run at most 1 function concurrently

        def foo():
            timer.start()
            return

        thread = T.submit(foo)
        thread2 = T.submit(foo)
        T.shutdown(wait=False)
        self.training_manager.training_futures[training_id] = thread
        self.training_manager.training_futures[training_id2] = thread2

        with self.assertRaises(RuntimeError) as context:
            TrainingManager.get_training_status(self.training_manager, training_id2)

        self.assertIn("Unexpected error", str(context.exception))
        timer.cancel()
