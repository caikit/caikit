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
import threading
import time
import unittest

# Local
from caikit.runtime.types.aborted_exception import AbortedException
from caikit.runtime.work_management.abortable_action import AbortableAction
from caikit.runtime.work_management.rpc_aborter import RpcAborter
from tests.fixtures import Fixtures


class TestAbortableAction(unittest.TestCase):
    """This test suite tests the abortable action class"""

    def setUp(self):
        """This method runs before each test begins to run"""
        self.rpc_context = Fixtures.build_context()
        self.aborter = RpcAborter(self.rpc_context)

    def test_it_can_run_a_function(self):
        expected_result = "test-any-result"
        action = AbortableAction(self.aborter, lambda *args, **kwargs: expected_result)
        result = action.do()
        self.assertEqual(expected_result, result)

    def test_it_raises_if_the_rpc_has_already_terminated(self):
        action = AbortableAction(self.aborter, lambda *args, **kwargs: None)
        self.rpc_context.cancel()

        with self.assertRaises(AbortedException) as context:
            action.do()

    def test_it_raises_if_the_function_raises(self):
        expected_exception = ValueError("test-any-error")

        def thrower():
            raise expected_exception

        action = AbortableAction(self.aborter, thrower)
        with self.assertRaises(ValueError) as ctx:
            action.do()

        self.assertEqual(expected_exception, ctx.exception)

    def test_it_raises_if_the_rpc_is_terminated_mid_function(self):
        infinite_function_has_started = threading.Event()

        def infinite_function():
            infinite_function_has_started.set()
            while True:
                time.sleep(0.1)

        action = AbortableAction(self.aborter, infinite_function)

        def inner_test_thread():
            with self.assertRaises(AbortedException) as context:
                action.do()

        thread = threading.Thread(target=inner_test_thread)
        thread.start()
        infinite_function_has_started.wait()

        self.rpc_context.cancel()
        thread.join(5)
        self.assertFalse(thread.is_alive())
