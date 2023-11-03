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
A DestroyableProcess implements a multiprocessing Process that captures errors
and communicates them to its parent with clean semantics for being destroyed.

NOTE: The "fork" start method is used for two reasons:
    1. It's faster
    2. For auto-generated classes (e.g. stream sources), "spawn" can result in
        missing classes since it performs a full re-import, but may not
        regenerate these classes.
"""
# Standard
from functools import partial
from typing import Any, Callable, Optional, Tuple
import multiprocessing
import os

# First Party
import alog

# Local
from .destroyable import Destroyable
from .pickling_exception import ExceptionPickler
from caikit.core.exceptions import error_handler

log = alog.use_channel("DESTROY-PROC")
error = error_handler.get(log)


OOM_EXIT_CODES = [137, 9, -9]

FORK_CTX = multiprocessing.get_context("fork")
SPAWN_CTX = multiprocessing.get_context("spawn")
FORKSERVER_CTX = multiprocessing.get_context("forkserver")


class _DestroyableProcess(
    multiprocessing.process.BaseProcess,
    Destroyable,
):  # pylint: disable=too-many-instance-attributes
    """The _DestroyableProcess base class implements a context-agnostic process
    class that manages the subprocess and allows it to be destroyed
    """

    def __init__(
        self,
        target: Optional[Callable] = None,
        completion_event: Optional[multiprocessing.Event] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[dict] = None,
        destroy_grace_period: float = 10,
        return_result: bool = False,
        **_kwargs,
    ):
        """Initialize with an event to use to signal completion"""
        self._parent_conn, self._child_conn = self.__class__._MP_CTX.Pipe()
        self._completion_event = completion_event or self.__class__._MP_CTX.Event()
        self._destroy_grace_period = destroy_grace_period
        self._return_result = return_result

        # This will hold the terminal result of the process which will either be
        # an Exception or some other value
        self.__result = None

        # Descriptions of completion state
        self.__started = False
        self.__destroyed = False
        self.__canceled = False

        # Bind the target to the target wrapper
        wrapped_target = partial(
            self._target_wrapper,
            target=target,
            args=args,
            kwargs=kwargs,
        )
        super().__init__(target=wrapped_target, **_kwargs)

    ## Destroyable Interface ##

    @property
    def destroyed(self) -> bool:
        return self.__destroyed

    @property
    def canceled(self) -> bool:
        return self.__canceled

    @property
    def ran(self) -> bool:
        return self.__started and not self.is_alive()

    @property
    def threw(self) -> bool:
        return self.error is not None

    def get_or_throw(self) -> Any:
        """Get the result of the execution or raise an error if one occurred"""
        if self.destroyed:
            log.error(
                "<COR24981767E>",
                "get_or_throw called on destroyed process, no value to return",
            )

        if not self.ran:
            log.error(
                "<COR12037430E>",
                "get_or_throw called on process, but it has not finished running",
            )

        # Update the result and throw if it's an error
        err = self.error
        if err is not None:
            raise err
        return self.__result

    def destroy(self):
        """Cancel any in-progress work"""
        self.__destroyed = True
        if self.is_alive() or not self.__started:
            self.__canceled = True
        if self.__started:
            self.terminate()
            self.join(self._destroy_grace_period)
            self.kill()
            self.join()
            self._completion_event.set()

    ## Process Interface ##

    def start(self):
        if self.destroyed:
            err_msg = "Not starting work on pre-canceled process"
            log.warning("<COR42191929W>", err_msg)
            self._completion_event.set()
            self.__result = RuntimeError(err_msg)
            return
        self.__started = True
        return super().start()

    def join(self, *args, **kwargs):
        if self.destroyed and not self.__started:
            return
        return super().join(*args, **kwargs)

    # NOTE: This functionality is covered by the unit tests, but pytest-cov is
    #   not correctly collecting the coverage info since it executes inside the
    #   subprocess
    def run(self):  # pragma: no cover
        try:
            # Run and indicate to the parent that no
            super().run()

        # Catch any errors thrown within a subprocess so that they can be
        # forwarded to the parent
        # pylint: disable=broad-exception-caught
        except Exception as err:
            err_str = repr(err)
            log.error(
                "<COR69863806E>",
                "Caught exception in destroyable process: %s",
                err_str,
                exc_info=True,
            )
            # Wrap error for safe pickling
            pickler = ExceptionPickler(err)
            self._child_conn.send(pickler)
        finally:
            self._completion_event.set()

    ## Impl ##

    def _update_result(self):
        if self._parent_conn.poll():
            self.__result = self._parent_conn.recv()

    @property
    def error(self) -> Optional[Exception]:
        self._update_result()

        if isinstance(self.__result, ExceptionPickler):
            return self.__result.get()

        if isinstance(self.__result, Exception):
            return self.__result

        if self.exitcode and self.exitcode != os.EX_OK:
            if self.exitcode in OOM_EXIT_CODES:
                return MemoryError("Training process died with OOM error!")
            if not self.canceled:
                return RuntimeError(
                    f"Training process died with exit code {self.exitcode}"
                )

    @property
    def completion_event(self) -> multiprocessing.Event:
        return self._completion_event

    # NOTE: This functionality is covered by the unit tests, but pytest-cov is
    #   not correctly collecting the coverage info since it executes inside the
    #   subprocess
    def _target_wrapper(self, target, args, kwargs):  # pragma: no cover
        result = target(*(args or []), **(kwargs or {}))
        log.debug3("Process target result: %s", result)
        if self._return_result:
            self._child_conn.send(result)
        self._completion_event.set()


class _ForkDestroyableProcess(FORK_CTX.Process, _DestroyableProcess):
    _MP_CTX = FORK_CTX


class _SpawnDestroyableProcess(SPAWN_CTX.Process, _DestroyableProcess):
    _MP_CTX = SPAWN_CTX


class _ForkserverDestroyableProcess(FORKSERVER_CTX.Process, _DestroyableProcess):
    _MP_CTX = FORKSERVER_CTX


_PROCESS_TYPES = {
    "fork": _ForkDestroyableProcess,
    "forkserver": _ForkserverDestroyableProcess,
    "spawn": _SpawnDestroyableProcess,
}


def DestroyableProcess(start_method: str, *args, **kwargs):
    """Class wrapper that returns the appropriate process type based on the
    requested start method.

    NOTE: Naming intentionally looks like a class!
    """
    error.value_check(
        "<COR16699811E>",
        start_method in _PROCESS_TYPES,
        "Unsupported start_method: {}",
        start_method,
    )
    return _PROCESS_TYPES[start_method](*args, **kwargs)
