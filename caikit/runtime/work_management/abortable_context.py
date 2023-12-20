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
from queue import SimpleQueue
from typing import Dict, Optional
import abc
import ctypes
import threading
import uuid

# First Party
import alog

# Local
from caikit.runtime.types.aborted_exception import AbortedException

log = alog.use_channel("ABORT-ACTION")


class ActionAborter(abc.ABC):
    """Simple interface to wrap up a notification that an action must abort.

    Children of this class can bind to any notification tool (e.g. grpc context)
    """

    @abc.abstractmethod
    def must_abort(self) -> bool:
        """Indicate whether or not the action must be aborted"""

    @abc.abstractmethod
    def set_context(self, context: "AbortableContext"):
        """Set the abortable context that must be notified to abort work"""

    @abc.abstractmethod
    def unset_context(self):
        """Unset any abortable context already held. Do not notify it that work should abort"""


class ThreadInterrupter:
    """This class implements a listener which will observe all ongoing work in `AbortableContexts`
    and raise exceptions in the working threads if they need to be aborted.

    The implementation spawns a single extra thread to wait on any contexts to abort, and
    interrupt the thread that the context is running in. This keeps the total number of running
    threads much smaller than using a new thread to monitor each AbortableContext.
    """

    _SHUTDOWN_SIGNAL = -1

    def __init__(self):
        # Using a SimpleQueue because we don't need the Queue's task api
        self._queue = SimpleQueue()
        self._thread: Optional[threading.Thread] = None
        self._context_thread_map: Dict[uuid.UUID, int] = {}
        self._start_stop_lock = threading.Lock()

    def start(self):
        """Start the watch loop that will abort any registered contexts passed to .kill()"""
        with self._start_stop_lock:
            if self._thread and self._thread.is_alive():
                log.debug("ThreadInterrupter already started")
                return
            log.debug("Starting ThreadInterrupter")
            self._thread = threading.Thread(target=self._watch_loop)
            self._thread.start()

    def stop(self):
        """Stop the watch loop"""
        with self._start_stop_lock:
            if self._thread and not self._thread.is_alive():
                log.debug("ThreadInterrupter already shut down")
                return

            log.info("Stopping ThreadInterrupter")
            self._queue.put(self._SHUTDOWN_SIGNAL)
            self._thread.join(timeout=1)

    def register(self, context_id: uuid, thread: int) -> None:
        self._context_thread_map[context_id] = thread

    def unregister(self, context_id: uuid) -> None:
        self._context_thread_map.pop(context_id, None)

    def kill(self, context_id: uuid) -> None:
        # Put this context onto the queue for abortion and immediately return
        self._queue.put(context_id, block=False)

    def _watch_loop(self):
        while True:
            log.debug("Waiting on any work to abort")
            context_id = self._queue.get()

            if context_id == self._SHUTDOWN_SIGNAL:
                log.debug("Ending abort watch loop")
                return

            self._kill_thread(context_id)

            # Ensure this context/thread pair is unregistered
            self.unregister(context_id)

    def _kill_thread(self, context_id: uuid.UUID) -> bool:
        thread_id = self._context_thread_map.get(context_id, None)

        if thread_id:
            log.debug("Interrupting thread id: ", thread_id)
            # This raises an AbortedException asynchronously in the target thread. (We can't just
            # use raise, because this thread is the ThreadInterrupter's watch thread).
            # The exception will only be raised once the target thread regains control of the
            # python interpreter. This means that statements like `time.sleep(9999999)` cannot be
            # interrupted in this manner.
            async_exception_result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), ctypes.py_object(AbortedException)
            )
            if async_exception_result > 1:
                log.warning("Failed to abort thread")
                return False

            return True

        else:
            log.warning("AbortableWork context already unregistered")
            return False


class AbortableContext:
    """Context manager for running work inside a context where it's safe to abort.

    This is a class instead of a `@contextmanager` function because __exit__ needs to
    happen on exception.
    """

    def __init__(self, aborter: ActionAborter, interrupter: ThreadInterrupter):
        """Setup the context.
        The aborter is responsible for notifying this context if the work needs to be aborted.
        The interrupter watches all such events, and kills the thread running in this context
        if the aborter notifies it to abort."""
        self.aborter = aborter
        self.interrupter = interrupter

        self.id = uuid.uuid4()

    def __enter__(self):
        if self.aborter and self.interrupter:
            log.debug4("Entering abortable context %s", self.id)
            # Set this context on the aborter so that it can notify us when work should be aborted
            self.aborter.set_context(self)
            # Register this context with the interrupter so that it knows which thread to kill
            thread_id = threading.get_ident()
            self.interrupter.register(self.id, thread_id)
        else:
            log.debug4("Aborter or Interrupter was None, no abortable context created.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.aborter and self.interrupter:
            # On any exit, whether an exception or not, we unregister with the interrupter
            # This prevents the interrupter from aborting this thread once this context has ended
            self.interrupter.unregister(self.id)
            self.aborter.unset_context()

    def abort(self):
        """Called by the aborter when this context needs to be aborted"""
        if self.interrupter:
            self.interrupter.kill(self.id)
