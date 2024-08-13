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
The LocalJobBase is the base class for running training and prediction
jobs using either local threads or subprocesses.

model_management:
    <job type>:
        <job executor name>:
            type: LOCAL
            config:
                # How long to retain results
                retention_duration: <null or str>
"""

# Standard
from concurrent.futures.thread import _threads_queues
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union
import abc
import os
import re
import threading
import uuid

# First Party
import aconfig
import alog

# Local
from ...interfaces.common.data_model.stream_sources import S3Path
from ..data_model import JobStatus
from ..exceptions import error_handler
from ..modules import ModuleBase
from .job_base import JobBase, JobFutureBase, JobInfo
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.toolkit.concurrency.destroyable_process import DestroyableProcess
from caikit.core.toolkit.concurrency.destroyable_thread import DestroyableThread

log = alog.use_channel("LOC-TRNR")
error = error_handler.get(log)


# 🌶️🌶️🌶️
# Fix for python3.9, 3.10 and 3.11 issue where forked processes always exit with exitcode 1
# when it's created inside a ThreadPoolExecutor: https://github.com/python/cpython/issues/88110
# Fix taken from https://github.com/python/cpython/pull/101940
# Credit: marmarek, https://github.com/marmarek

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_threads_queues.clear)


class LocalJobBase(JobBase):
    """LocalJobBase is the base class for running background jobs on a local
    machine and does not depend on any datastore or backend. Any result
    is cleared after the defined retention_duration"""

    class LocalJobFuture(JobFutureBase):
        """A local job future manages running a job in a thread and tracking
        its status
        """

        def __init__(
            self,
            future_name: str,
            module_class: Type[ModuleBase],
            save_path: Optional[Union[str, S3Path]],
            save_with_id: bool,
            future_id: Optional[str],
            args: Iterable[Any],
            kwargs: Dict[str, Any],
            model_name: Optional[str] = None,
            use_subprocess: bool = False,
            extra_path_args: Optional[List[str]] = None,
            subprocess_start_method: Optional[str] = "fork",
        ):
            super().__init__(
                future_name=future_name,
                future_id=future_id or str(uuid.uuid4()),
                model_name=model_name,
                use_reversible_hash=future_id is None,
            )

            self._module_class = module_class

            # Placeholder for the time when the future completed
            self._completion_time = None
            # Set the submission time as right now. (Maybe this should be supplied instead?)
            self._submission_time = datetime.now()

            # Other implementations should deal with an S3 ref first and not pass it along here
            if save_path and isinstance(save_path, S3Path):
                raise ValueError("S3 output path not supported by this runtime")
            self._save_path = self.__class__._save_path_with_id(
                save_path,
                save_with_id,
                self._id,
                model_name,
                extra_path_args,
            )

            # Set up the worker and start it
            self._use_subprocess = use_subprocess
            self._subprocess_start_method = subprocess_start_method
            if self._use_subprocess:
                log.debug2("Running background task %s as a SUBPROCESS", self.id)
                self._worker = DestroyableProcess(
                    start_method=self._subprocess_start_method,
                    target=self.run,
                    return_result=False,
                    args=args,
                    kwargs={
                        **kwargs,
                    },
                )
                # If background task in a subprocess without a save path, the result
                # will be unreachable once completed!
                if not self.save_path:
                    log.warning(
                        "<COR28853922W>",
                        "Background task %s launched in a subprocess with no save path",
                        self.id,
                    )
            else:
                log.debug2("Running training %s as a THREAD", self.id)
                self._worker = DestroyableThread(
                    self.run,
                    *args,
                    **kwargs,
                )
            self._worker.start()

        @property
        def save_path(self) -> Optional[str]:
            """If created with a save path, the future must expose it, including
            any injected background id
            """
            return self._save_path

        @property
        def completion_time(self) -> Optional[datetime]:
            return self._completion_time

        ## Class Abstraction ##
        @abc.abstractmethod
        def run(self):
            """Abstract method for starting the job. Needs to be implemented by
            the trainers and predictors"""

        ## Interface Implementation ##

        def get_info(self) -> JobInfo:
            """The local job checks the info by the current thread
            status
            """

            # The worker was canceled while doing work. It may still be in the
            # process of terminating and thus still alive.
            if self._worker.canceled:
                return self._make_background_info(status=JobStatus.CANCELED)

            # If the worker is currently alive it's doing work
            if self._worker.is_alive():
                return self._make_background_info(status=JobStatus.RUNNING)

            # The worker threw outside of a cancellation process
            if self._worker.threw:
                return self._make_background_info(
                    status=JobStatus.ERRORED, errors=[self._worker.error]
                )

            # The worker completed its work without being canceled or raising
            if self._worker.ran:
                return self._make_background_info(status=JobStatus.COMPLETED)

            # If it's not alive and not done, it hasn't started yet
            return self._make_background_info(status=JobStatus.QUEUED)

        def cancel(self):
            """Terminate the given job"""
            log.debug("Canceling background job %s", self.id)
            with alog.ContextTimer(
                log.debug2, "Done canceling background job %s in: ", self.id
            ):
                log.debug3("Destroying worker in %s", self.id)
                self._worker.destroy()

        def wait(self):
            """Block until the job reaches a terminal state"""
            log.debug2("Waiting for %s", self.id)
            self._worker.join()
            log.debug2("Done waiting for %s", self.id)
            self._completion_time = self._completion_time or datetime.now()

        ## Impl ##
        def _delete_result(self):
            """Helper function to clear out the result when purging"""
            if self.save_path and Path(self.save_path).exists():
                Path(self.save_path).unlink(missing_ok=True)

        def _make_background_info(
            self, status: JobStatus, errors: Optional[List[Exception]] = None
        ) -> JobInfo:
            return JobInfo(
                status=status,
                errors=errors,
                completion_time=self._completion_time,
                submission_time=self._submission_time,
            )

        @classmethod
        def _save_path_with_id(
            cls,
            save_path: Optional[str],
            save_with_id: bool,
            future_id: str,
            model_name: Optional[str],
            extra_path_args: Optional[List[str]],
        ) -> Optional[str]:
            """If asked to save_with_id, child classes should use this shared
            utility to construct the final save path
            """
            if save_path is None:
                return save_path

            final_path_parts = [save_path]
            # If told to save with the ID in the path, inject it before the
            # model name.
            if save_with_id and future_id not in save_path:
                # (Don't inject training id if its already in the path)
                final_path_parts.append(future_id)

            if model_name and model_name not in save_path:
                final_path_parts.append(model_name)

            final_path_parts.extend(extra_path_args or [])
            return os.path.join(*final_path_parts)

    ## Interface ##

    # Expression for parsing retention policy
    _timedelta_expr = re.compile(
        r"^((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d*\.?\d*?)s)?$"
    )

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with a shared dict for all the jobs"""
        self._instance_name = instance_name
        self._retention_duration = config.get("retention_duration")
        if self._retention_duration is not None:
            try:
                log.debug2("Parsing retention duration: %s", self._retention_duration)
                self._retention_duration = timedelta(
                    **{
                        key: float(val)
                        for key, val in self._timedelta_expr.match(
                            self._retention_duration
                        )
                        .groupdict()
                        .items()
                        if val is not None
                    }
                )
            except AttributeError:
                error(
                    "<COR63897671E>",
                    ValueError(
                        f"Invalid retention_duration: {self._retention_duration}"
                    ),
                )

        # The shared dict of futures and a lock to serialize mutations to it
        self._futures: Dict[str, self.LocalJobFuture] = {}
        self._futures_lock = threading.Lock()

    def get_local_future(self, job_id: str) -> LocalJobFuture:
        """Look up the model future for the given id"""
        self._purge_old_futures()
        if model_future := self._futures.get(job_id):
            return model_future
        raise CaikitCoreException(
            status_code=CaikitCoreStatusCode.NOT_FOUND,
            message=f"Unknown background_id: {job_id}",
        )

    ## Impl ##

    def _purge_old_futures(self):
        """If a retention duration is configured, purge any futures that are
        older than the policy
        """
        if self._retention_duration is None:
            return
        now = datetime.now()
        purged_ids = {
            fid: future
            for fid, future in self._futures.items()
            if future.completion_time is not None
            and future.completion_time + self._retention_duration < now
        }
        if not purged_ids:
            log.debug3("No ids to purge")
            return
        log.debug3("Purging ids: %s", purged_ids)
        with self._futures_lock:
            for fid, future in purged_ids.items():
                # NOTE: Concurrent purges could have already done this, so don't
                #   error if the id is already gone
                self._futures.pop(fid, None)

                # Attempt to delete results if another purge hasn't already
                future._delete_result()
