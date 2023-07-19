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
from typing import Callable
import abc
import threading

# First Party
import alog

# Local
from caikit.core.toolkit.destroyable_thread import DestroyableThread
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
    def add_event(self, event: threading.Event):
        """Add an event to notify when abort happens"""


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
