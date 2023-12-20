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

"""
This module helps us know when an rpc call is cancelled, and we need to stop or undo work
"""
# Third Party
import grpc

# First Party
import alog

# Local
from caikit.runtime.work_management.abortable_context import (
    AbortableContext,
    ActionAborter,
)

log = alog.use_channel("CALL-ABORTER")


class RpcAborter(ActionAborter):
    """
    This class registers a callback with a grpc context, to be called in the event of rpc
    termination. Termination could be nominal (we returned a response) but we should have
    relinquished control anyway. The interesting case is when a client cancels a call or a deadline
    is hit, which could trigger this callback but will not interrupt the thread doing work.

    In order to actually interrupt threads doing the work, abortable contexts can be registered
    with an instance of this class in order ton receive notification on RPC termination.

    IFF the RPC has been terminated, `must_abort` will return True.
    """

    def __init__(self, context: grpc.ServicerContext):
        self.is_terminated = False
        self.context = None

        callback_registered = context.add_callback(self.__rpc_terminated)

        # If we can't register the callback, it probably means the RPC has already been terminated.
        # We probably shouldn't be doing any more work.
        if not callback_registered:
            log.warning(
                "<RUN65620101W>",
                "Failed to register rpc termination callback, call has likely terminated",
            )
            self.is_terminated = True

    def must_abort(self):
        return self.is_terminated

    def set_context(self, context: AbortableContext):
        self.context = context
        if self.must_abort():
            self.context.abort()

    def unset_context(self):
        self.context = None

    def __rpc_terminated(self):
        self.is_terminated = True
        if self.context:
            self.context.abort()
