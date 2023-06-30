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
The LocalModelTrainer uses a local thread to launch and manage each training job
"""

# Standard
from concurrent.futures.thread import _threads_queues
from datetime import datetime, timedelta
from typing import Optional, Type
import os
import re
import threading
import uuid

# First Party
import aconfig
import alog

# Local
from ..modules import ModuleBase
from ..toolkit.destroyable_thread import DestroyableThread
from ..toolkit.errors import error_handler
from .model_trainer_base import ModelTrainerBase

log = alog.use_channel("TH-TAINER")
error = error_handler.get(log)


OOM_EXIT_CODE = 137


# 🌶️🌶️🌶️
# Fix for python3.9, 3.10 and 3.11 issue where forked processes always exit with exitcode 1
# when it's created inside a ThreadPoolExecutor: https://github.com/python/cpython/issues/88110
# Fix taken from https://github.com/python/cpython/pull/101940
# Credit: marmarek, https://github.com/marmarek

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_threads_queues.clear)


class LocalModelTrainer(ModelTrainerBase):
    __doc__ = __doc__

    name = "LOCAL"

    class LocalModelFuture(ModelTrainerBase.ModelFutureBase):
        """A local model future manages an execution thread for a single train
        operation
        """

        def __init__(
            self,
            module_class: Type[ModuleBase],
            *args,
            save_path: Optional[str],
            save_with_id: bool,
            **kwargs,
        ):
            self._id = str(uuid.uuid4())
            self._module_class = module_class
            self._save_path = ModelTrainerBase.save_path_with_id(
                save_path, save_with_id, self.id
            )

            # Placeholder for the time when the future completed
            self._completion_time = None

            # Set up the worker and start it
            self._complete_event = threading.Event()
            self._worker = DestroyableThread(
                self._complete_event,
                self._train_and_save,
                *args,
                **kwargs,
            )
            self._worker.start()

        @property
        def completion_time(self) -> Optional[datetime]:
            return self._completion_time

        ## Interface ##

        @property
        def id(self) -> str:
            return self._id

        @property
        def save_path(self) -> str:
            return self._save_path

        def get_status(self) -> ModelTrainerBase.TrainingStatus:
            """Every model future must be able to poll the status of the
            training job
            """
            # If the thread is currently alive it's doing work
            if self._worker.is_alive():
                return ModelTrainerBase.TrainingStatus.RUNNING

            # If the thread is not alive, threw, and was destroyed, the throw
            # was caused by a deliberate cancellation
            if self._worker.destroyed and self._worker.threw:
                return ModelTrainerBase.TrainingStatus.CANCELED

            # If the worker threw, but was not destroyed, it was an error in the
            # train function
            if self._worker.threw:
                return ModelTrainerBase.TrainingStatus.ERRORED

            # If the thread ran and none of the non-success termination states
            # is true, it completed successfully
            if self._worker.ran:
                return ModelTrainerBase.TrainingStatus.COMPLETED

            # If it's not alive and not done, it hasn't started yet
            return ModelTrainerBase.TrainingStatus.QUEUED

        def cancel(self):
            """Terminate the given training"""
            log.debug("Canceling training %s", self.id)
            with alog.ContextTimer(
                log.debug2, "Done canceling training %s in: ", self.id
            ):
                self._worker.destroy()
                self.wait()

        def wait(self):
            """Block until the job reaches a terminal state"""
            self._complete_event.wait()
            self._worker.join()

        def load(self) -> ModuleBase:
            """Wait for the training to complete, then return the resulting
            model or raise any errors that happened during training.
            """
            self.wait()
            return self._worker.get_or_throw()

        ## Impl ##

        def _train_and_save(self, *args, **kwargs):
            """Function that will run in the worker thread"""
            with alog.ContextTimer(log.debug, "Training %s finished in: ", self.id):
                trained_model = self._module_class.train(*args, **kwargs)
            if self.save_path is not None:
                log.debug("Saving training %s to %s", self.id, self.save_path)
                with alog.ContextTimer(log.debug, "Training %s saved in: ", self.id):
                    trained_model.save(self.save_path)
            self._completion_time = datetime.now()
            return trained_model

    ## Interface ##

    # Expression for parsing retention policy
    _timedelta_expr = re.compile(
        r"^((?P<days>\d+?)d)?((?P<hours>\d+?)hr)?((?P<minutes>\d+?)m)?((?P<seconds>\d*\.?\d*?)s)?$"
    )

    def __init__(self, config: aconfig.Config):
        """Initialize with a shared dict of all trainings"""
        self._retention_duration = config.get("retention_duration")
        if self._retention_duration is not None:
            try:
                self._retention_duration = timedelta(
                    **{
                        key: float(val)
                        for key, val in self._timedelta_expr.match("23d")
                        .groupdict()
                        .items()
                        if val is not None
                    }
                )
            except AttributeError:
                error(
                    "<COR63897671E>",
                    f"Invalid retention_duration: {self._retention_duration}",
                )

        # The shared dict of futures and a lock to serialize mutations to it
        self._futures = {}
        self._futures_lock = threading.Lock()

    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[str] = None,
        save_with_id: bool = False,
        **kwargs,
    ) -> "LocalModelFuture":
        """Start training the given module and return a future to the trained
        model instance
        """
        # Always purge old futures
        self._purge_old_futures()

        # Create the new future
        model_future = self.LocalModelFuture(
            module_class,
            *args,
            save_path=save_path,
            save_with_id=save_with_id,
            **kwargs,
        )

        # Lock the global futures dict and add it to the dict
        with self._futures_lock:
            assert (
                model_future.id not in self._futures
            ), f"UUID collision for model future {model_future.id}"
            self._futures[model_future.id] = model_future

        # Return the future
        return model_future

    def get_model_future(self, training_id: str) -> "LocalModelFuture":
        """Look up the model future for the given id"""
        self._purge_old_futures()
        if model_future := self._futures.get(training_id):
            return model_future
        raise ValueError(f"Unknown training_id: {training_id}")

    ## Impl ##

    def _purge_old_futures(self):
        """If a retention duration is configured, purge any futures that are
        older than the policy
        """
        if self._retention_duration is None:
            return
        now = datetime.now()
        purged_ids = {
            fid
            for fid, future in self._futures.items()
            if future.completion_time is not None
            and future.completion_time + self._retention_duration < now
        }
        if not purged_ids:
            log.debug3("No ids to purge")
            return
        log.debug3("Purging ids: %s", purged_ids)
        with self._futures_lock:
            for fid in purged_ids:
                # NOTE: Concurrent purges could have already done this, so don't
                #   error if the id is already gone
                self._futures.pop(fid, None)
