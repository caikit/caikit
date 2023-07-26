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
from typing import Optional
import ctypes
import sys
import threading
import traceback

# First Party
import alog

# Local
from .destroyable import Destroyable

log = alog.use_channel("DESTROY-THRD")


class ThreadDestroyedException(RuntimeError):
    """Exception raised inside a DestroyableThread when it is destroyed by the thread managing
    its lifecycle."""

    def __init__(self):
        super().__init__(
            "Work thread intentionally destroyed by its lifecycle manager. "
            "This exception was not raised by the code running in this thread."
        )


# pylint: disable=too-many-instance-attributes
class DestroyableThread(threading.Thread, Destroyable):
    """A class for Destroyable Threads. When work is delegated to a thread but may need to be
    canceled while in progress, we use this class which allows us to raise an exception inside
    the work thread.

    Exceptions raised this way are asynchronous and they will not interrupt the python instruction
    that the thread is currently executing. E.g. a time.sleep() will finish sleeping before the
    exception is raised.

    This class may be initialized with a threading event, which it will set when the thread
    finishes executing, whether nominally or by raising an exception.
    """

    # The exception we'll throw to kill the thread
    __exception = ThreadDestroyedException

    def __init__(
        self,
        runnable_func,
        *runnable_args,
        work_done_event: Optional[threading.Event] = None,
        **runnable_kwargs
    ):
        threading.Thread.__init__(self)
        self.work_done_event = work_done_event or threading.Event()

        # These describe the work to be done
        self.runnable_func = runnable_func
        self.runnable_args = runnable_args
        self.runnable_kwargs = runnable_kwargs

        # These describe what happened with the work
        self.__runnable_result = None
        self.__runnable_exception = None
        self.__threw = False
        self.__started = False
        self.__ran = False

        # In case `destroy` is called before Python has actually started the thread, we need to
        # know to not do the work
        self.__destroyed = False

    @property
    def destroyed(self) -> bool:
        return self.__destroyed

    @property
    def canceled(self) -> bool:
        return self.destroyed and self.__started and not self.__ran

    @property
    def ran(self) -> bool:
        return self.__ran

    @property
    def threw(self) -> bool:
        return self.__threw

    # Run wraps the supplied function with logic to set the event when it finishes, and save any
    # result or raised error

    def run(self) -> None:
        """
            Overrides Thread.run()
            *Do not call*

        Returns:
            None
        """
        # Raise immediately if the thread was destroyed before
        if self.__destroyed:
            log.info(
                "<COR14653273I>",
                "Not starting work for %s, thread already cancelled",
                self.runnable_func,
            )
            self.__raise()

        self.__started = True
        try:
            self.__runnable_result = self.runnable_func(
                *self.runnable_args, **self.runnable_kwargs
            )
            self.__threw = False
        except:  # pylint: disable=bare-except
            # PEP8 complains, but in this case we really do want to re-throw _any_ exception that
            # occurred. In the interest of transparently wrapping any work in these threads, we
            # want to keep exception signatures identical. E.g. if I expect this thread to throw a
            # CaikitRuntimeException, I want to be able to catch a CaikitRuntimeException.
            # Rethrowing from `sys.exc_info()[1]` should retain all stack trace info later.
            e = sys.exc_info()[1]
            self.__runnable_exception = e
            self.__threw = True

            # Add a little bit of visibility to know why work failed
            if self.__destroyed:
                log.info(
                    {
                        "log_code": "<COR15827563I>",
                        "message": "Work for {} was aborted and threw".format(
                            self.runnable_func
                        ),
                        "stack_trace": traceback.format_exc(),
                    }
                )
            else:
                log.warning(
                    {
                        "log_code": "<COR16788843W>",
                        "message": "Work for {} threw exception: {}".format(
                            self.runnable_func, e
                        ),
                        "stack_trace": traceback.format_exc(),
                    }
                )

        finally:
            # Before setting the synchronization event, flag that the work was done
            self.__ran = True
            self.work_done_event.set()

    def get_or_throw(self):
        """
            After the thread has completed it's work, call this to get the output.
        Returns:
            The resulting value of runnable_func(*runnable_args, **runnable_kwargs)
        Raises:
            Any exception raised by runnable_func(*runnable_args, **runnable_kwargs)
        """
        if self.destroyed:
            log.error(
                "<COR14653274E>",
                "get_or_throw called on destroyed thread for %s, no value to return",
                self.runnable_func,
            )

        if not self.ran:
            log.error(
                "<COR14653275E>",
                "get_or_throw called on thread for %s, but it has not finished running",
                self.runnable_func,
            )

        if self.threw:
            raise self.__runnable_exception
        return self.__runnable_result

    def destroy(self) -> None:
        """
            Cancel any in-progress work and kill the thread if it is alive.
            Otherwise, prevent the thread from running at all.
        Returns:
            None
        """

        # Set the destroyed flag in case the thread has not started yet.
        # If it has, we should be able to kill it with the async exception below.
        self.__destroyed = True

        # The thread has already finished or is not yet alive, so we cannot kill it
        thread_id = self.__get_id()
        if thread_id is None or not self.is_alive():
            log.debug(
                "<COR14653276D>",
                "Destroying thread that is not currently alive: %s",
                self.runnable_func,
            )
            return

        # This is the code that raises an async exception in the target thread
        # (We can't just use raise, because the parent thread is in this control flow)
        async_exception_result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), ctypes.py_object(self.__exception)
        )
        if async_exception_result > 1:
            log.error(
                "<COR14653277E>",
                "Could not raise async exception on destroyable thread for %s. Result code: %s",
                self.runnable_func,
                async_exception_result,
            )

    @property
    def error(self) -> Optional[Exception]:
        if isinstance(self.__runnable_exception, Exception):
            return self.__runnable_exception

    def __get_id(self):
        # Returns the thread if if the thread is running
        for thread in threading.enumerate():
            if thread is self:
                return thread.ident

        # Otherwise, the thread has completed or has not started
        return None

    def __raise(self):
        # __exception is just a type, we need to be sure to initialize a value of it
        raise self.__exception()
