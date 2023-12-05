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
from typing import Callable, Dict
import abc
import ctypes
import threading
import uuid

# First Party
import alog

# Local
from caikit.core.toolkit.concurrency.destroyable_thread import DestroyableThread
from caikit.runtime.types.aborted_exception import AbortedException

log = alog.use_channel("ABORT-ACTION")


class AbortableContextBase(abc.ABC):
    @abc.abstractmethod
    def abort(self):
        """Called to abort work in progress"""


class ActionAborter(abc.ABC):
    """Simple interface to wrap up a notification that an action must abort.

    Children of this class can bind to any notification tool (e.g. grpc context)
    """

    @abc.abstractmethod
    def must_abort(self) -> bool:
        """Indicate whether or not the action must be aborted"""

    @abc.abstractmethod
    def add_event(self, event: threading.Event):
        """Add an event to notify when abort happens"""

    @abc.abstractmethod
    def set_context(self, context: AbortableContextBase):
        """Set the abortable context that must be notified to abort work"""

    @abc.abstractmethod
    def unset_context(self):
        """Unset any abortable context already held. Do not notify it that work should abort"""


class AbortableAction:
    """A class for Abortable Actions. We want actions that are computationally heavy to be
    abortable by Model Mesh! Currently, we use this for the following operations.

    - Loading a model
    - Predicting with a model
    - Training a model

    In the future, this may include getting the size of a model, depending on how that we choose
    to implement that.

    How it works:
        Instances of this class create a threading.Event, which will be used to signal that either:
        - The RPC was terminated
        - The heavy work that we wanted to complete is done
        This is done by using a RpcAborter and a DestroyableThread.
        Registering the event with the RpcAborter will cause it to set when the RPC is
        terminated, and creating a DestroyableThread with the event will cause it to set when
        the thread terminates.

        The action will start the DestroyableThread and then wait on the event. When it wakes, it
        will check the reason and destroy the thread if it was woken by the RpcAborter or return
        the result if it was woken by the thread completing.
    """

    def __init__(
        self,
        call_aborter: ActionAborter,
        runnable_func: Callable,
        *args,
        **kwargs,
    ):
        """
        Args:
            call_aborter - call aborter capable of aborting the runnable_func
            runnable_func - the function to be run as an abortable action
            *args - nonkeyword arguments to runnable_func
            **kwargs - keyword arguments to runnable_func"""

        # Create new event to watch for both RPC termination and work completion
        self.__done_or_aborted_event = threading.Event()

        # Register the event with our call aborter so it fires if the RPC terminates
        self.call_aborter = call_aborter
        self.call_aborter.add_event(self.__done_or_aborted_event)

        # Create a new thread to do the work, which will set the event if it finished
        self.__runnable_func = runnable_func
        self.__work_thread = DestroyableThread(
            self.__runnable_func,
            *args,
            work_done_event=self.__done_or_aborted_event,
            **kwargs,
        )

    def do(self):
        # Start the work and wait
        self.__work_thread.start()
        self.__done_or_aborted_event.wait()

        # Now, check the call aborter to see what happened.
        # Option 1: The RPC was terminated. Kill the work thread and raise an exception
        if self.call_aborter.must_abort():
            log.info(
                "<RUN14653271I>", "Aborting work in progress: %s", self.__runnable_func
            )
            self.__work_thread.destroy()
            self.__work_thread.join()
            raise AbortedException("Aborted work: {}".format(self.__runnable_func))

        # Options 2: Work thread finished normally. Hooray!
        log.debug("Work finished: %s", self.__runnable_func)
        self.__work_thread.join()
        return self.__work_thread.get_or_throw()


class WorkWatcher:
    """This class implements a listener which will observe all ongoing work registered with
    ActionAborters and raise exceptions in the working threads if they need to be aborted.

    This offers a performance advantage over using `AbortableActions`, since only one extra
    listener thread is created that lives for the whole lifetime of the program. The caveat
    is that all work to be aborted must be running in a thread that is safe to kill: this would
    not be safe to use with asyncio tasks running in an event loop.
    """

    def __init__(self):
        # Using a SimpleQueue because we don't need the Queue's task api
        self._queue = SimpleQueue()
        self._thread = threading.Thread(target=self._watch_loop)

        self._context_thread_map: Dict[uuid.UUID, int] = {}

        self._shutdown_event = threading.Event()

        self._total_rip_count = 0
        self._total_registered = 0
        self._total_unregistered = 0

    def start(self):
        log.debug4("Starting WorkWatcher")
        self._thread.start()

    def stop(self):
        log.debug4("Stopping WorkWatcher")
        self._shutdown_event.set()
        self._queue.put(0)
        self._thread.join(timeout=1)

    def register(self, context_id: uuid, thread: int) -> None:
        self._context_thread_map[context_id] = thread
        self._total_registered += 1

    def unregister(self, context_id: uuid) -> None:
        self._context_thread_map.pop(context_id, None)
        self._total_unregistered += 1

    def kill(self, context_id: uuid) -> None:
        # Put this context onto the queue for abortion and immediately return
        self._queue.put(context_id, block=False)

    def _watch_loop(self):
        while True:
            try:
                log.debug4("Waiting on any work to abort")
                context_id = self._queue.get()

                if self._shutdown_event.is_set():
                    log.debug4("Ending abort watch loop")
                    return

                self._kill_thread(context_id)

                # Ensure this context/thread pair is unregistered
                self.unregister(context_id)

            except Exception:
                log.warning("Caught exception while running abort loop", exc_info=True)

    def _kill_thread(self, context_id: uuid.UUID) -> bool:
        thread_id = self._context_thread_map.get(context_id, None)

        if thread_id:
            async_exception_result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), ctypes.py_object(AbortedException)
            )
            if async_exception_result > 1:
                log.warning("Failed to abort thread")
                return False

            self._total_rip_count += 1
            return True

        else:
            log.warning("AbortableWork context already unregistered")
            return False


class AbortableContext(AbortableContextBase):
    """Context manager for running work inside a context where it's safe to abort.

    This is a class instead of a `@contextmanager` function because __exit__ needs to
    happen on exception.
    """

    def __init__(self, aborter: ActionAborter, watcher: WorkWatcher):
        """Setup the context.
        The aborter is responsible for notifying this context if the work needs to be aborted.
        The watcher watches all such events, and kills the thread running in this context
        if the aborter notifies it to abort."""
        self.aborter = aborter
        self.watcher = watcher

        self.id = uuid.uuid4()

    def __enter__(self):
        if self.aborter and self.watcher:
            # Set this context on the aborter so that it can notify us when work should be aborted
            self.aborter.set_context(self)
            # Register this context with the watcher so that it knows which thread to kill
            thread_id = threading.get_ident()
            self.watcher.register(self.id, thread_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.aborter and self.watcher:
            # On any exit, whether an exception or not, we unregister with the watcher
            # This prevents the watcher from aborting this thread once this context has ended
            self.watcher.unregister(self.id)
            self.aborter.unset_context()

    def abort(self):
        """Called by the aborter when this context needs to be aborted"""
        if self.watcher:
            self.watcher.kill(self.id)
