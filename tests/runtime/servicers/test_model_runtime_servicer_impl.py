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
from threading import Event, Thread
from unittest.mock import MagicMock, patch

try:
    # Standard
    from test.support.threading_helper import catch_threading_exception
except (NameError, ModuleNotFoundError):
    from tests.base import catch_threading_exception

# Standard
import time
import unittest

# Local
from caikit import get_config
from caikit.runtime.protobufs import model_runtime_pb2
from caikit.runtime.servicers.model_runtime_servicer import ModelRuntimeServicerImpl
from caikit.runtime.types.aborted_exception import AbortedException
from tests.fixtures import Fixtures


class TestModelRuntimeServicerImpl(unittest.TestCase):
    """This test suite tests the ModelRuntimeServicerImpl class"""

    def setUp(self):
        """This method runs before each test begins to run"""
        self.servicer = ModelRuntimeServicerImpl()

    def test_model_load_sets_per_model_concurrency(self):
        model = "test-any-model-id"
        # Grab a model type that has some max concurrency set
        model_type = list(
            get_config().inference_plugin.model_mesh.max_model_concurrency_per_type.keys()
        )[0]
        request = model_runtime_pb2.LoadModelRequest(
            modelId=model, modelType=model_type
        )
        context = Fixtures.build_context(model)

        expected_concurrency = (
            get_config().inference_plugin.model_mesh.max_model_concurrency_per_type[
                model_type
            ]
        )
        mock_manager = MagicMock()
        mock_manager.load_model.return_value = 1

        with patch.object(self.servicer, "model_manager", mock_manager):
            response = self.servicer.loadModel(request, context)
        self.assertEqual(expected_concurrency, response.maxConcurrency)

    def test_model_load_sets_default_max_model_concurrency(self):
        model = "test-any-model-id"
        model_type = "some-fake-model-type"
        request = model_runtime_pb2.LoadModelRequest(
            modelId=model, modelType=model_type
        )
        context = Fixtures.build_context(model)

        expected_concurrency = (
            get_config().inference_plugin.model_mesh.max_model_concurrency
        )
        mock_manager = MagicMock()
        mock_manager.load_model.return_value = 1

        with patch.object(self.servicer, "model_manager", mock_manager):
            response = self.servicer.loadModel(request, context)
        self.assertEqual(expected_concurrency, response.maxConcurrency)

    def test_load_model_aborts(self):
        """ModelRuntimeServicer.loadModel will abort a long-running load"""
        model = "test-any-model-id"
        request = model_runtime_pb2.LoadModelRequest(modelId=model)
        context = Fixtures.build_context(model)

        mock_manager = MagicMock()
        started = Event()

        def never_return(*args, **kwargs):
            started.set()
            while True:
                time.sleep(0.01)

        mock_manager.load_model.side_effect = never_return
        load_thread = Thread(target=self.servicer.loadModel, args=(request, context))

        with catch_threading_exception() as cm:
            with patch.object(self.servicer, "model_manager", mock_manager):
                load_thread.start()
                started.wait()
                context.cancel()
                load_thread.join(10)

            self.assertFalse(load_thread.is_alive())

            # Make sure the correct exception was raised
            assert cm.exc_type == AbortedException
