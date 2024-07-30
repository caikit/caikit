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
The LocalJobPredictor uses a local thread to launch and manage each prediction job

model_management:
    job_predictors:
        <predictor name>:
            type: LOCAL
            config:
                # ! Inherits config from LocalJobBase
                # Path to a local directory that holds the results. Defaults
                # to a temporary directory
                result_dir: <null or str>
"""

# Standard
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
import re

# First Party
import aconfig
import alog

# Local
from ..data_model import DataObjectBase
from ..exceptions import error_handler
from ..modules import ModuleBase
from .job_predictor_base import JobPredictorBase
from .local_job_base import LocalJobBase
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)

log = alog.use_channel("LOC-TRNR")
error = error_handler.get(log)


class LocalJobPredictor(LocalJobBase, JobPredictorBase):
    __doc__ = __doc__

    class LocalJobPredictorFuture(LocalJobBase.LocalJobFuture):
        """LocalJobPredictorFuture takes a model instance and prediction function
        and runs the function in a destroyable thread"""

        def __init__(
            self,
            model_instance: ModuleBase,
            prediction_func_name: str,
            *args,
            **kwargs: Dict[str, Any],
        ):
            self._model_instance = model_instance
            self._prediction_func_name = prediction_func_name
            self._result_type = None
            super().__init__(*args, **kwargs)

        def run(self, *args, **kwargs):
            """Run the prediction and save the results in a binary format to a file"""
            # If running in a spawned subprocess, reconfigure logging
            with alog.ContextTimer(log.debug, "Inference %s finished in: ", self.id):
                model_run_fn = getattr(self._model_instance, self._prediction_func_name)
                infer_result = model_run_fn(*args, **kwargs)
            if self.save_path is not None:
                save_path_pathlib = Path(self.save_path)
                log.debug("Saving inference %s to %s", self.id, self.save_path)
                save_path_pathlib.parent.mkdir(exist_ok=True)
                with alog.ContextTimer(
                    log.debug, "Inference %s saved in: ", self.id
                ) and save_path_pathlib.open("wb") as output_file:
                    output_file.write(infer_result.to_binary_buffer())

            self._result_type = infer_result.__class__
            self._completion_time = self._completion_time or datetime.now()
            log.debug2("Completion time for %s: %s", self.id, self._completion_time)
            return infer_result

        def result(self) -> DataObjectBase:
            """Fetch the result from the result directory"""
            if not self.completion_time:
                raise CaikitCoreException(
                    CaikitCoreStatusCode.NOT_FOUND,
                    f"Prediction {self.id} is still in progress",
                )

            result_path = Path(self.save_path)
            if not result_path.exists():
                raise CaikitCoreException(
                    CaikitCoreStatusCode.NOT_FOUND,
                    f"Prediction result for {self.id} is not found",
                )

            assert self._result_type
            return self._result_type.from_binary_buffer(result_path.read_bytes())

    LocalModelFuture = LocalJobPredictorFuture

    name = "LOCAL"

    ## Interface ##

    # Expression for parsing retention policy
    _timedelta_expr = re.compile(
        r"^((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?((?P<seconds>\d*\.?\d*?)s)?$"
    )

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize by creating a result temporary directory or
        just holding a reference"""
        self._result_dir = config.get("result_dir", None)
        self._tmp_dir = None
        if not self._result_dir:
            self._tmp_dir = TemporaryDirectory()
            self._result_dir = self._tmp_dir.name
        super().__init__(config, instance_name)

    def __del__(self):
        if self._tmp_dir:
            self._tmp_dir.cleanup()

    def predict(
        self,
        model_instance: ModuleBase,
        prediction_func_name: str,
        *args,
        save_with_id: bool = True,
        external_inference_id: Optional[str] = None,
        **kwargs,
    ) -> LocalJobPredictorFuture:
        """Start training the given module and return a future to the trained
        model instance
        """
        # Always purge old futures
        self._purge_old_futures()

        # If there's an external ID, make sure it's not currently running before
        # launching the job
        if external_inference_id and (
            current_future := self._futures.get(external_inference_id)
        ):
            error.value_check(
                "<COR79850561E>",
                current_future.get_info().status.is_terminal,
                "Cannot restart inference {} that is currently running",
                external_inference_id,
            )

        # Create the new future
        model_future = self.LocalModelFuture(
            future_name=self._instance_name,
            model_instance=model_instance,
            prediction_func_name=prediction_func_name,
            save_path=self._result_dir,
            future_id=external_inference_id,
            save_with_id=save_with_id,
            use_subprocess=False,  # don't use subprocess
            module_class=model_instance.__class__,
            args=args,
            kwargs=kwargs,
            extra_path_args=["result.bin"],
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

    def get_model_future(self, inference_id: str) -> "LocalModelFuture":
        """Look up the model future for the given id"""
        self._purge_old_futures()
        if model_future := self._futures.get(inference_id):
            return model_future
        raise CaikitCoreException(
            status_code=CaikitCoreStatusCode.NOT_FOUND,
            message=f"Unknown training_id: {inference_id}",
        )
