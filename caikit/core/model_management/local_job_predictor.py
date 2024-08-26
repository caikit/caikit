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
import shutil

# First Party
import aconfig
import alog

# Local
from ..data_model import DataObjectBase
from ..exceptions import error_handler
from ..modules import ModuleBase
from .job_predictor_base import JobPredictorBase, JobPredictorFutureBase
from .local_job_base import LocalJobBase
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)

log = alog.use_channel("LOC-TRNR")
error = error_handler.get(log)

# Constant for the result file name. This is exposed here to better support
# subclassing
RESULT_FILE_NAME = "result.bin"


class LocalJobPredictor(LocalJobBase, JobPredictorBase):
    __doc__ = __doc__

    class LocalJobPredictorFuture(LocalJobBase.LocalJobFuture, JobPredictorFutureBase):
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
            super().__init__(*args, extra_path_args=[RESULT_FILE_NAME], **kwargs)

        def run(self, *args, **kwargs):
            """Run the prediction and save the results in a binary format to a file"""

            # Create parent prediction directory before starting request
            save_path_pathlib = Path(self.save_path)
            log.debug3(
                "Creating prediction result directory %s", save_path_pathlib.parent
            )
            save_path_pathlib.parent.mkdir(exist_ok=True, parents=True)

            with alog.ContextTimer(
                log.debug, "Inference in job %s finished in: ", self.id
            ):
                model_run_fn = getattr(self._model_instance, self._prediction_func_name)
                infer_result = model_run_fn(*args, **kwargs)
                self._result_type = infer_result.__class__

            # If save path was provided then output result
            if self.save_path is not None:
                self._save_result(infer_result)

            self._completion_time = self._completion_time or datetime.now()
            log.debug2("Completion time for %s: %s", self.id, self._completion_time)
            return infer_result

        def result(self) -> DataObjectBase:
            """Fetch the result from the result directory"""
            # Wait for future to complete
            self.wait()

            result_path = Path(self.save_path)
            if not result_path.exists():
                raise CaikitCoreException(
                    CaikitCoreStatusCode.NOT_FOUND,
                    f"Prediction result for {self.id} is not found",
                )

            assert self._result_type
            return self._result_type.from_binary_buffer(result_path.read_bytes())

        def _delete_result(self):
            # Delete the result.bin if one exists
            super()._delete_result()

            # Delete parent directory to clear out future id
            if self.save_path:
                parent_path = Path(self.save_path).parent
                if parent_path.exists():
                    shutil.rmtree(parent_path, ignore_errors=True)

        def _save_result(self, result: DataObjectBase):
            """Helper function to save a result to disk. This abstraction helps
            subclasses implement their own save method like S3"""
            save_path_pathlib = Path(self.save_path)
            log.debug("Saving inference %s to %s", self.id, self.save_path)
            save_path_pathlib.parent.mkdir(exist_ok=True, parents=True)
            with alog.ContextTimer(
                log.debug, "Inference %s saved in: ", self.id
            ) and save_path_pathlib.open("wb") as output_file:
                output_file.write(result.to_binary_buffer())

    name = "LOCAL"

    ## Interface ##

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
        external_prediction_id: Optional[str] = None,
        **kwargs,
    ) -> LocalJobPredictorFuture:
        """Start prediction the given model and return a future to the result
        of the prediction
        """
        # Always purge old futures
        self._purge_old_futures()

        # If there's an external ID, make sure it's not currently running before
        # launching the job
        if external_prediction_id and (
            current_future := self._futures.get(external_prediction_id)
        ):
            error.value_check(
                "<COR79850561E>",
                current_future.get_info().status.is_terminal,
                "Cannot restart prediction {} that is currently running",
                external_prediction_id,
            )

        # Create the new future
        model_future = self.LocalJobPredictorFuture(
            future_name=self._instance_name,
            model_instance=model_instance,
            prediction_func_name=prediction_func_name,
            save_path=self._result_dir,
            future_id=external_prediction_id,
            save_with_id=save_with_id,
            use_subprocess=False,  # don't use subprocess
            module_class=model_instance.__class__,
            args=args,
            kwargs=kwargs,
        )

        # Lock the global futures dict and add it to the dict
        with self._futures_lock:
            if current_future := self._futures.get(model_future.id):
                error.value_check(
                    "<COR30431427E>",
                    current_future.get_info().status.is_terminal,
                    "UUID collision for model future {}",
                    model_future.id,
                )
            self._futures[model_future.id] = model_future

        # Return the future
        return model_future

    def get_prediction_future(self, future_id: str) -> LocalJobPredictorFuture:
        """Look up the model future for the given id"""
        return self.get_local_future(future_id)
