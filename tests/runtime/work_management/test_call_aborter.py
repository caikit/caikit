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
import unittest

# Local
from caikit.runtime.work_management.rpc_aborter import RpcAborter
from tests.fixtures import Fixtures


class TestRpcAborter(unittest.TestCase):
    """This test suite tests the call aborter utility"""

    CHANNEL = None
    GRPC_THREAD = None

    def test_call_aborter_sets_event(self):
        ctx = Fixtures.build_context("call_aborter_event_party")
        # Create a new Call aborter
        aborter = RpcAborter(ctx)
        # Create a new threading event and add it to the call aborter
        event = threading.Event()
        aborter.add_event(event)
        # Cancel the call & wait for the threading event to be set by __rpc_terminated
        ctx.cancel()
        event.wait()
        self.assertTrue(aborter.must_abort())

    def test_call_aborter_sets_event_added_after_termination(self):
        ctx = Fixtures.build_context("call_aborter_event_party")
        # Create a new call aborter
        aborter = RpcAborter(ctx)
        # Cancel the call before creating the threading event and adding to the aborter
        ctx.cancel()
        event = threading.Event()
        aborter.add_event(event)
        event.wait()
        self.assertTrue(aborter.must_abort())


if __name__ == "__main__":
    unittest.main()
