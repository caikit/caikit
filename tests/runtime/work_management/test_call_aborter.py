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
import unittest

# Local
from caikit.runtime.work_management.rpc_aborter import RpcAborter
from tests.fixtures import Fixtures


class StubAbortableContext:
    """Test context, simply sets flag if `abort` was called"""

    def __init__(self):
        self.aborted = False

    def abort(self):
        self.aborted = True


def test_call_aborter_invokes_abortable_context():
    """The whole reason this class exists:
    If the grpc context is canceled, the abortable context should be aborted
    """
    grpc_ctx = Fixtures.build_context("call_aborter_event_party")
    abort_ctx = StubAbortableContext()

    # Create a new Call aborter
    aborter = RpcAborter(grpc_ctx)
    # Set its abort context
    aborter.set_context(abort_ctx)

    assert not abort_ctx.aborted

    # Cancel the call and check that context was aborted
    grpc_ctx.cancel()

    assert abort_ctx.aborted


def test_call_aborter_invokes_abortable_context_when_grpc_context_is_already_canceled():
    """Edge case: if the grpc context has already been canceled, the abortable context is immediately aborted as well"""
    grpc_ctx = Fixtures.build_context("call_aborter_event_party")
    abort_ctx = StubAbortableContext()

    # Prematurely cancel grpc context
    grpc_ctx.cancel()

    # Create a new Call aborter
    aborter = RpcAborter(grpc_ctx)
    # Set its abort context
    aborter.set_context(abort_ctx)
    # And it should immediately abort
    assert abort_ctx.aborted


if __name__ == "__main__":
    unittest.main()
