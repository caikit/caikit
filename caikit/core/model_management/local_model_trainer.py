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

model_management:
    trainers:
        <predictor name>:
            type: LOCAL
            config:
                # ! Inherits config from LocalJobBase
                # Whether to train in a subprocess or different thread
                use_subprocess: <bool: False>
                # Type of subprocess to use if use_subprocess=True
                subprocess_start_method: <string: spawn>
                
"""

# Standard
from datetime import datetime
from typing import Optional, Type
import os
import uuid

# First Party
import aconfig
import alog

# Local
from ..exceptions import error_handler
from ..exceptions.caikit_core_exception import CaikitCoreException, CaikitCoreStatusCode
from ..modules import ModuleBase
from ..toolkit.logging import configure as configure_logging
from .local_job_base import LocalJobBase
from .model_trainer_base import ModelTrainerBase, ModelTrainerFutureBase
import caikit

log = alog.use_channel("LOC-TRNR")
error = error_handler.get(log)


class LocalModelTrainer(LocalJobBase, ModelTrainerBase):
    __doc__ = __doc__

    class LocalModelTrainFuture(LocalJobBase.LocalJobFuture, ModelTrainerFutureBase):
        """LocalModelTrainFuture takes a module class and training request and
        runs it in a thread or subprocess"""

        def run(self, *args, **kwargs):
            """Function that will run in the worker thread"""
            # If running in a spawned subprocess, reconfigure logging
            if self._use_subprocess and self._subprocess_start_method != "fork":
                configure_logging()
            with alog.ContextTimer(log.debug, "Training %s finished in: ", self.id):
                trained_model = self._module_class.train(*args, **kwargs)
            if self.save_path is not None:
                log.debug("Saving training %s to %s", self.id, self.save_path)
                with alog.ContextTimer(log.debug, "Training %s saved in: ", self.id):
                    trained_model.save(self.save_path)
            self._completion_time = self._completion_time or datetime.now()
            log.debug2("Completion time for %s: %s", self.id, self._completion_time)
            return trained_model

        def load(self) -> ModuleBase:
            """Wait for the training to complete, then return the resulting
            model or raise any errors that happened during training.
            """
            self.wait()
            if self._use_subprocess:
                log.debug2("Loading model saved in subprocess")
                error.value_check(
                    "<COR16745216E>",
                    self.save_path,
                    "Unable to load model from training {} "
                    + "trained in subprocess without a save_path",
                    self.id,
                )
                error.value_check(
                    "<COR59551640E>",
                    os.path.exists(self.save_path),
                    "Unable to load model from training {} "
                    + "saved in subprocess, path does not exist: {}",
                    self.id,
                    self.save_path,
                )
                result = caikit.load(self.save_path)
            else:
                result = self._worker.get_or_throw()
            return result

    LocalModelFuture = LocalModelTrainFuture

    name = "LOCAL"

    ## Interface ##

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with a shared dict of all trainings"""
        self._use_subprocess = config.get("use_subprocess", False)
        self._subprocess_start_method = config.get("subprocess_start_method", "spawn")
        super().__init__(config, instance_name)

    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[str] = None,
        save_with_id: bool = False,
        external_training_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> "LocalModelFuture":
        """Start training the given module and return a future to the trained
        model instance
        """
        # Always purge old futures
        self._purge_old_futures()

        # Wrap any models in the kwargs for safe spawning if needed
        if self._use_subprocess and self._subprocess_start_method != "fork":
            wrapped_models = {
                key: _SpawnProcessModelWrapper(val)
                for key, val in kwargs.items()
                if isinstance(val, ModuleBase)
            }
            log.debug2("Subprocess wrapped models: %s", wrapped_models.keys())
            kwargs.update(wrapped_models)

        # If there's an external ID, make sure it's not currently running before
        # launching the job
        if external_training_id and (
            current_future := self._futures.get(external_training_id)
        ):
            error.value_check(
                "<COR79850511E>",
                current_future.get_info().status.is_terminal,
                "Cannot restart training {} that is currently running",
                external_training_id,
            )

        # Update kwargs with required information
        future_id = external_training_id or str(uuid.uuid4())

        # Create the new future
        model_future = self.LocalModelFuture(
            self._instance_name,
            module_class,
            save_path=save_path,
            future_id=future_id,
            save_with_id=save_with_id,
            use_subprocess=self._use_subprocess,
            subprocess_start_method=self._subprocess_start_method,
            model_name=model_name,
            args=args,
            kwargs=kwargs,
        )

        # Lock the global futures dict and add it to the dict
        with self._futures_lock:
            if current_future := self._futures.get(model_future.id):
                error.value_check(
                    "<COR35431427E>",
                    current_future.get_info().status.is_terminal,
                    "UUID collision for model future {}",
                    model_future.id,
                )
            self._futures[model_future.id] = model_future

        # Return the future
        return model_future

    def get_model_future(self, training_id: str) -> "LocalModelFuture":
        """Look up the model future for the given id. Use custom implementation
        to retain error codes"""
        self._purge_old_futures()
        if model_future := self._futures.get(training_id):
            return model_future
        raise CaikitCoreException(
            status_code=CaikitCoreStatusCode.NOT_FOUND,
            message=f"Unknown training_id: {training_id}",
        )


class _SpawnProcessModelWrapper(ModuleBase):
    """This class wraps up a model to make it safe to pass to a spawned
    subprocess. It will not be efficient, but it will be safe!
    """

    def __init__(self, model: ModuleBase):
        super().__init__()
        self._model = model

    def __getattr__(self, name):
        """Forward attributes that are not found on the base class to the model

        NOTE: This does _not_ forward base class attributes since those are
            resolved before __getattr__ is called.
        """
        return getattr(self._model, name)

    def save(self, *args, **kwargs):
        """Directly forward save to the model so that it is not called by the
        base class
        """
        return self._model.save(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Directly forward run to the model so that it is not called by the
        base class
        """
        return self._model.run(*args, **kwargs)

    def __getstate__(self) -> bytes:
        """When pickling, only send the serialized model body for non-fork. This
        is not a general-purpose pickle solution for models, but makes them safe
        for training jobs that need to move models between processes.
        """
        return self._model.as_bytes()

    def __setstate__(self, pickled: bytes):
        """When unpickling, deserialize the body if the model is not already
        loaded in the model manager. This must be used in conjunction with the
        above __getstate__ across a process boundary and should not be used as a
        general-purpose deserialization for models.
        """
        retrieved_model = caikit.core.load(pickled)
        self._model = retrieved_model
