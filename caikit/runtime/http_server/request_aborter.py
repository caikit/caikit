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
This module helps us know when a HTTP client call is cancelled, and we need to stop or undo work
"""
# Standard
from typing import Optional
import asyncio
import threading

# Third Party
from fastapi import Request

# First Party
import alog

# Local
from caikit.runtime.work_management.abortable_context import (
    AbortableContext,
    ActionAborter,
)

log = alog.use_channel("REQUEST-ABORTER")


class HttpRequestAborter(ActionAborter):
    """
    In order to actually interrupt threads doing the work, abortable contexts can be registered
    with an instance of this class in order to receive notification on request disconnection.
    This allows work to be terminated when a client time's out or stops listening.

    IFF the client request has been terminated, `must_abort` will return True.
    """

    def __init__(
        self,
        context: Request,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        poll_time: Optional[float] = 0.25,
    ):
        """Initialize a Aborter and start the watch loop

        Args:
            context: starlette.Request
               The HTTP Request to be watched
            loop: Optional[asyncio.AbstractEventLoop]
               The asyncio loop to run tasks on. If not provided use the running loop
           poll_time: Optional[int]
               The time between disconnect checks
        """

        self.context = context
        self.event_loop = loop or asyncio.get_running_loop()
        self.poll_time = poll_time

        # Set initial
        self.is_terminated = threading.Event()
        self.abortable_context = None

        # Start request aborter task. Hold onto a reference of the task to ensure
        # it isn't garbage collected
        log.debug("<RUN81824293>", "Watching for request disconnect")
        self.task = self.event_loop.create_task(self.watch_for_disconnect())

    def __enter__(self):
        """Helper function to enable context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically abort aborter when exiting contextmanager"""
        self.abort()

    async def watch_for_disconnect(self):
        """Wait for a context to be disconnected"""

        while True:
            # Short circuit incase thread terminated externally
            if self.is_terminated.is_set():
                log.debug3(
                    "<RUN81824293>", "RequestAborter has already been terminated"
                )
                return

            is_disconnected = await self.context.is_disconnected()
            if is_disconnected:
                log.debug("<RUN81824293>", "Client disconnected, terminating action")
                self.is_terminated.set()
                if self.abortable_context:
                    self.abortable_context.abort()
                return
            log.debug4("<RUN81824293>", "Client still connected, sleeping aborter")
            await asyncio.sleep(self.poll_time)

    def abort(self):
        """Helper function to stop aborter before the request has terminated"""
        self.is_terminated.set()

    def must_abort(self):
        return self.is_terminated.is_set()

    def set_context(self, context: AbortableContext):
        self.abortable_context = context
        if self.must_abort():
            self.abortable_context.abort()

    def unset_context(self):
        self.abortable_context = None
