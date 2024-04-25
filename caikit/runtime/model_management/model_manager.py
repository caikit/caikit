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
from collections import Counter as DictCounter
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union
import atexit
import gc
import os
import shutil
import threading
import time

# Third Party
from grpc import StatusCode
from prometheus_client import Counter, Gauge, Summary

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import ModuleBase
from caikit.core.exceptions import error_handler
from caikit.core.model_management import ModelFinderBase, ModelInitializerBase
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_loader import ModelLoader
from caikit.runtime.model_management.model_sizer import ModelSizer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("MODEL-MANAGR")
error = error_handler.get(log)

MODEL_SIZE_GAUGE = Gauge(
    "total_loaded_models_size",
    "Total size of loaded models reported to model-mesh",
    ["model_type"],
)
MODEL_COUNT_GAUGE = Gauge(
    "total_loaded_models", "Total number of loaded models", ["model_type", "model_id"]
)
LOAD_MODEL_EXCEPTION_COUNTER = Counter(
    "load_model_exception_count",
    "Count of exceptions raised during loadModel RPCs",
    ["model_type"],
)
LOAD_MODEL_DURATION_SUMMARY = Summary(
    "load_model_duration_seconds",
    "Summary of the duration (in seconds) of loadModel RPCs",
    ["model_type"],
)
LOCAL_MODEL_TYPE = "LOCAL"


class ModelManager:  # pylint: disable=too-many-instance-attributes
    """Model Manager class. The singleton class contains the core implementational details
    for the Model Runtime (i.e., load/unload functionality, etc). It does not provide the core
    details for predict calls."""

    __instance = None

    __model_size_gauge_lock = threading.Lock()

    _LOCAL_MODEL_TYPE = "standalone-model"

    ## Construction ##

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """This method returns the instance of Model Manager"""
        if not cls.__instance:
            cls.__instance = ModelManager()
        return cls.__instance

    def __init__(self):
        """Initialize a ModelManager instance."""
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        ModelManager.__instance = self

        # Pull in a ModelLoader and ModelSizer
        self.model_loader = ModelLoader.get_instance()
        self.model_sizer = ModelSizer.get_instance()

        # In-memory mapping of model_id to LoadedModel instance
        self.loaded_models: Dict[str, LoadedModel] = {}

        # Lock for mutating operations on loaded_models
        self._loaded_models_lock = threading.Lock()

        # Optionally load models mounted into a local directory
        runtime_cfg = get_config().runtime
        self._local_models_dir = runtime_cfg.local_models_dir or ""
        if self._local_models_dir and not os.path.exists(self._local_models_dir):
            log.warning(
                "<RUN53709826W>",
                "Invalid runtime.local_models_dir %s. Does not exist",
                self._local_models_dir,
            )
            self._local_models_dir = ""

        # Keep track of whether lazy loading is enabled
        self._lazy_load_local_models = runtime_cfg.lazy_load_local_models

        if self._lazy_load_local_models:
            error.value_check(
                "<RUN44773525E>",
                runtime_cfg.local_models_dir is not None,
                (
                    "runtime.local_models_dir must be set"
                    " if using runtime.lazy_load_local_models. "
                ),
            )

            error.value_check(
                "<RUN44773514E>",
                self._local_models_dir,
                (
                    "runtime.local_models_dir must be a valid path"
                    " if set with runtime.lazy_load_local_models. "
                    f"Provided path: {runtime_cfg.local_models_dir}"
                ),
            )

        # Set up local model periodic sync
        self._lazy_load_poll_period_seconds = runtime_cfg.lazy_load_poll_period_seconds
        error.type_check(
            "<RUN59138047E>",
            int,
            float,
            allow_none=True,
            lazy_load_poll_period_seconds=self._lazy_load_poll_period_seconds,
        )
        self._lazy_sync_timer = None
        self._enable_lazy_load_poll = (
            self._local_models_dir
            and self._lazy_load_local_models
            and self._lazy_load_poll_period_seconds
        )
        self._lazy_load_write_detection_period_seconds = (
            runtime_cfg.lazy_load_write_detection_period_seconds
        )
        error.type_check(
            "<RUN58138047E>",
            int,
            float,
            allow_none=True,
            lazy_load_write_detection_period_seconds=self._lazy_load_write_detection_period_seconds,
        )
        if self._enable_lazy_load_poll:
            atexit.register(self.shut_down)

        # Do the initial local models load
        if self._local_models_dir:
            wait = runtime_cfg.wait_for_initial_model_loads
            load = runtime_cfg.load_new_local_models
            log.info(
                "<RUN44739400I>",
                "Initializing local_models_dir %s. Wait: %s. Load: %s",
                self._local_models_dir,
                wait,
                load,
            )
            self.sync_local_models(wait=wait, load=load)

    def shut_down(self):
        """Shut down cache purging"""
        self._enable_lazy_load_poll = False
        timer = getattr(self, "_lazy_sync_timer", None)
        if timer is not None:
            timer.cancel()
            if timer.is_alive():
                timer.join()

    ## Model Management ##

    def load_model(
        self,
        model_id: str,
        local_model_path: str,
        model_type: str,
        wait: bool = True,
        retries: Optional[int] = None,
        finder: Optional[Union[str, ModelFinderBase]] = None,
        initializer: Optional[Union[str, ModelInitializerBase]] = None,
    ) -> LoadedModel:
        """Load a model using model_path (in Cloud Object Storage) & give it a model ID
        Args:
            model_id (str):  Model ID string for the model to load.
            local_model_path (str): Local path to load the model from.
            model_type (str): Type of the model to load.
            wait (bool): Wait for the model to finish loading
            retries (Optional[int]): Number of times to retry on load failure
        Returns:
            model (LoadedModel): The LoadedModel instance
        """
        with LOAD_MODEL_DURATION_SUMMARY.labels(model_type=model_type).time():

            # If already loaded, just return the size
            # NOTE: We make the dict access atomic here to avoid the race where
            #   we check if model_id in the map, then re-look it up to get the
            #   size which could fail if it is unloaded between the two.
            model = self.loaded_models.get(model_id)
            if model is not None:
                log.debug("Model '%s' is already loaded", model_id)
                return model

            # Grab the mutation lock and load the model if needed
            with self._loaded_models_lock:
                # Re-check now that the mutation lock is held
                model = self.loaded_models.get(model_id)
                if model is None:
                    log.debug("Loading %s from %s", model_id, local_model_path)
                    try:
                        model = self.model_loader.load_model(
                            model_id,
                            local_model_path,
                            model_type,
                            fail_callback=partial(self.unload_model, model_id),
                            retries=retries,
                            finder=finder,
                            initializer=initializer,
                        )
                    except Exception as ex:
                        self.__increment_load_model_exception_count_metric(model_type)
                        raise ex

                    # Estimate the model's size and update the LoadedModel
                    try:
                        model_size = self.model_sizer.get_model_size(
                            model_id, local_model_path, model_type
                        )
                    except CaikitRuntimeException:
                        log.debug(
                            "Unable to estimate model size for non-disk model: %s",
                            model_id,
                        )
                        model_size = 0
                    model.set_size(model_size)

                    # Add model + helpful metadata to our loaded models map
                    self.loaded_models[model_id] = model

                    # Update Prometheus metrics
                    self.__increment_model_count_metric(model_type, model_id)
                    self.__report_total_model_size_metric()

            # If waiting, do so outside of the mutation lock
            if wait:
                model.wait()

            # Return the loaded model handle
            return model

    def sync_local_models(self, wait: bool = False, load: bool = True):
        """Sync in-memory models with models in the configured local_model_dir

        New models will be loaded and models previously loaded from local will
        be unloaded.

        Args:
            wait (bool): After starting all loads, wait for them to complete
            load (bool): Perform loading during sync
        """
        try:
            self._local_models_dir_sync(wait, load)
        except StopIteration:
            log.warning(
                "<RUN56519883W>",
                "local_models_dir %s unreachable. Terminating synchronization",
                self._local_models_dir,
            )
            self._enable_lazy_load_poll = False
        except Exception as err:  # pylint: disable=broad-exception-caught
            log.warning(
                "<RUN44524933W>",
                "Exception raised during local_models_dir sync: %s",
                str(err),
                exc_info=True,
            )

        # If running periodically, kick off the next iteration
        if self._enable_lazy_load_poll:
            if self._lazy_sync_timer is None:
                log.info(
                    "Initializing local_models_dir sync with period %s",
                    self._lazy_load_poll_period_seconds,
                )
            if self._lazy_sync_timer is not None and self._lazy_sync_timer.is_alive():
                log.debug3("Canceling live timer")
                self._lazy_sync_timer.cancel()
            log.debug3(
                "Starting next poll timer for %ss", self._lazy_load_poll_period_seconds
            )
            log.debug4(
                "All open threads: %s",
                [thread.name for thread in threading.enumerate()],
            )
            self._lazy_sync_timer = threading.Timer(
                self._lazy_load_poll_period_seconds,
                self.sync_local_models,
                kwargs={"load": load},
            )
            self._lazy_sync_timer.daemon = True
            self._lazy_sync_timer.start()

    def unload_model(self, model_id) -> int:
        """Unload a model by ID model.

        Args:
            model_id (string):  Model ID string for the model to unload. If None,
                (default) the default model id will be used.
        Returns:
            Model_size (int) : Size of the loaded model in bytes
        """
        log.debug("List of loaded models: %s", str(self.loaded_models))
        try:
            # If the model failed to load, just return 0; no need to throw an error here.
            model = self.loaded_models.pop(model_id, None)
            if model is None:
                log.debug(
                    "Model '%s' is not loaded, so it cannot be unloaded!", model_id
                )
                return 0

            # Temporarily store model size and type info
            model_type = model.type()
            model_size = model.size()

            # If the model is still loading, we need to wait for it to finish so
            # that we can do our best to fully free it
            model.wait()
            del model
        except CaikitRuntimeException:
            raise
        except Exception as ex:
            log.debug("Model '%s' failed to unload with error: %s", model_id, repr(ex))
            raise CaikitRuntimeException(
                StatusCode.INTERNAL,
                "Model could not be unloaded!!",
                {"model_id": model_id},
            ) from ex

        # Invoke the garbage collector to ensure that memory is freed and ready for loading new
        # models.  This also helps troubleshooting when examining memory usage during unload.
        gc.collect()

        # Update Prometheus metrics
        self.__report_total_model_size_metric()
        self.__decrement_model_count_metric(model_type, model_id)

        return model_size

    def unload_all_models(self):
        """Unload all loaded models"""
        all_model_ids = list(self.loaded_models.keys())
        for model_id in all_model_ids:
            self.unload_model(model_id)

    def get_model_size(self, model_id) -> int:
        """Look up size of a model by model ID.
        Args:
            model_id (string):  Model ID string for the model. Throw Exception if empty,
                or it is an ID of a model that is not currently loaded.
        Returns:
            Model_size (int) : Size of the loaded model in bytes
        """
        if not model_id or model_id not in self.loaded_models:
            msg = (
                "Unable to retrieve the size of model '%s'; it is unregistered or unloaded."
                % model_id
            )
            log.debug(msg)
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND, msg, {"model_id": model_id}
            )

        loaded_model = self.loaded_models[model_id]
        self.__report_total_model_size_metric()
        return loaded_model.size()

    def estimate_model_size(self, model_id, local_model_path, model_type) -> int:
        """Predict size of a model using model ID and path.
        Args:
            model_id (string):  Model ID string for the model to predict size of.
            local_model_path (string): Local path to the model.
            model_type (string): Type of the model
        Returns:
            Model_size (int) : Estimated size of the model in bytes.
        """
        return self.model_sizer.get_model_size(model_id, local_model_path, model_type)

    def retrieve_model(self, model_id: str) -> ModuleBase:
        """Retrieve a model from the loaded model map by model ID.

        Args:
            model_id (str): Model ID of the model to retrieve
        Returns:
            response (caikit.core.module.ModuleBase):
                A loaded Caikit model
        """
        if not model_id or not isinstance(model_id, str):
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT, "Missing required model ID"
            )

        # Now retrieve the model and fall back to lazy loading
        loaded_model = self.loaded_models.get(model_id)
        if not loaded_model and self._lazy_load_local_models:
            local_model_path = os.path.join(self._local_models_dir, model_id)
            log.debug2(
                "Lazy loading local model %s from %s", model_id, local_model_path
            )
            # If the model is not present on disk, attempt to lazy load it
            # anyway using the model_id as the "model_path". This allows
            # auto-finders that can infer the model's config to load based on
            # the ID.
            if not os.path.exists(local_model_path):
                log.debug2("Attempting to load ephemeral model %s", model_id)
                local_model_path = model_id
            loaded_model = self.load_model(
                model_id=model_id,
                local_model_path=local_model_path,
                model_type=self._LOCAL_MODEL_TYPE,
                wait=True,
                retries=get_config().runtime.lazy_load_retries,
            )

        # If still not loaded, there's nothing to find, so raise NOT_FOUND
        if not loaded_model:
            msg = f"Model '{model_id}' not loaded"
            log.debug(
                {"log_code": "<RUN61105243D>", "message": msg, "model_id": model_id}
            )
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND, msg, {"model_id": model_id}
            )

        # NOTE: If the model is partially loaded, this call will wait on the
        #   model future in the LoadedModel
        return loaded_model.model()

    def deploy_model(
        self,
        model_id: str,
        model_files: Dict[str, bytes],
        **kwargs,
    ) -> LoadedModel:
        """Given in-memory model files, this will save the model to the local
        models dir, then load it locally.
        """
        error.value_check(
            "<RUN05068605E>",
            self._local_models_dir,
            "runtime.local_models_dir must be a valid path to deploy models directly.",
        )
        try:
            # If the model directory already exists, it's an error
            model_dir = os.path.join(self._local_models_dir, model_id)
            if os.path.exists(model_dir):
                msg = f"Model '{model_id}' already exists"
                raise CaikitRuntimeException(
                    StatusCode.ALREADY_EXISTS, msg, {"model_id": model_id}
                )

            # Create the model directory directory
            os.makedirs(model_dir)

            # Write out all of the files
            for fname, data in model_files.items():
                fname = fname.strip()
                if not fname:
                    raise CaikitRuntimeException(
                        StatusCode.INVALID_ARGUMENT,
                        f"Got whitespace-only model file name: [{fname}]",
                        {"model_id": model_id},
                    )
                fpath = os.path.join(model_dir, fname)
                if not os.path.commonpath([model_dir, fpath]).lstrip(os.sep):
                    raise CaikitRuntimeException(
                        StatusCode.INVALID_ARGUMENT,
                        f"Cannot use absolute paths for model files: {fname}",
                        {"model_id": model_id},
                    )

                # Make sure intermediate dirs exist
                parent_dir = os.path.dirname(fpath)
                if os.path.relpath(parent_dir, model_dir) != ".":
                    os.makedirs(parent_dir, exist_ok=True)

                log.debug2(
                    "Writing model file %s of size %s to %s", fname, len(data), fpath
                )
                with open(fpath, "wb") as handle:
                    handle.write(data)

            # Load the model
            return self.load_model(
                model_id=model_id,
                local_model_path=model_dir,
                model_type=self._LOCAL_MODEL_TYPE,
                **kwargs,
            )

        except PermissionError as err:
            raise CaikitRuntimeException(
                StatusCode.FAILED_PRECONDITION,
                f"Unable to save model (PermissionError): {err}",
                {"model_id": model_id},
            ) from err

        except OSError as err:
            raise CaikitRuntimeException(
                StatusCode.UNKNOWN,
                f"Unable to save model (OSError): {err}",
                {"model_id": model_id},
            ) from err

    def undeploy_model(self, model_id: str):
        """Remove the given model from the loaded model map and delete the
        artifacts from the local models dir.
        """
        error.value_check(
            "<RUN05068606E>",
            self._local_models_dir,
            "runtime.local_models_dir must be a valid path to undeploy models directly.",
        )

        # Check to see if the model exists in `local_models_dir` and delete it
        # if so
        local_model_path = os.path.join(self._local_models_dir, model_id)
        if os.path.exists(local_model_path):
            log.debug("Removing local model path: %s", local_model_path)
            shutil.rmtree(local_model_path)

            # If currently loaded in memory, unload it (unload_model will not
            # raise if not found)
            self.unload_model(model_id)

        else:
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND,
                f"Cannot undeploy unknown model {model_id}",
                {"model_id": model_id},
            )

    ## Implementation Details ##

    def _local_models_dir_sync(self, wait: bool = False, load: bool = True):
        """This function implements the mechanics of synchronizing the
        local_models_dir and the in-memory loaded_models map. It may raise and
        therefore errors should be handled by the wrapper function.

        NOTE: In the case that the local_models_dir becomes unreadable, it will
            raise StopIteration to indicate that any periodic synchronization
            should terminate.
        """
        # Get the list of models on disk
        # NOTE: If the local_models_dir has disappeared, this is likely a unit
        #   test with a temp dir, but in any event, we should stop trying to
        #   sync going forward
        try:
            disk_models = os.listdir(self._local_models_dir)
        except FileNotFoundError as err:
            log.error(
                "<RUN44739499E>", "Failed to read model ids from disk", exc_info=True
            )
            raise StopIteration() from err

        log.debug3("All models found in local disk cache: %s", disk_models)
        log.debug3("Currently loaded models: %s", list(self.loaded_models.keys()))

        # Find all models that aren't currently loaded
        if load:
            new_models = [
                model_id
                for model_id in disk_models
                if model_id not in self.loaded_models
            ]
            log.debug("New local models: %s", new_models)
        else:
            log.debug("Skipping new model loading")
            new_models = []

        # Find all models that are currently loaded from the local models dir
        # that no longer exist
        unload_models = [
            model_id
            for model_id, loaded_model in self.loaded_models.items()
            if loaded_model.path().startswith(self._local_models_dir)
            and not os.path.exists(loaded_model.path())
        ]
        log.debug("Unloaded local models: %s", unload_models)

        # Load new models
        for model_id in new_models:
            model_path = os.path.join(self._local_models_dir, model_id)

            if self._model_write_in_progress(model_path):
                log.debug("Model %s is still being written", model_id)
                continue

            self.load_model(
                model_id,
                model_path,
                self._LOCAL_MODEL_TYPE,
                wait=False,
                retries=get_config().runtime.lazy_load_retries,
            )

        # Unload old models
        # NOTE: No need for error handling here since unload_model will warn on
        #   errors and move on
        for model_id in unload_models:
            log.debug2("Unloading local model %s", model_id)
            self.unload_model(model_id)

        # Wait for models to load
        if wait:
            for model_id in new_models:
                loaded_model = self.loaded_models.get(model_id)
                # If somehow already purged, there's nothing to wait on. This is
                # extremely unlikely since it would require another thread to
                # explicitly call unload on the model AND have the model finish
                # loading between then and now. Better to be safe than sorry,
                # though!
                if loaded_model is None:  # pragma: no cover
                    continue
                # Wait for it and make sure it didn't fail
                try:
                    loaded_model.wait()
                except CaikitRuntimeException as err:
                    log.debug(
                        "<RUN56627485D>",
                        "Failed to load model %s: %s",
                        model_id,
                        repr(err),
                        exc_info=True,
                    )

    def _model_write_in_progress(self, model_dir: str) -> bool:
        """Returns true if model_dir is currently being written to. Uses the
        runtime.lazy_load_write_detection_period_seconds configuration to sleep between
        consecutive size checks of the directory.

        Always returns false if runtime.lazy_load_write_detection_period_seconds is zero,
        negative, or None.
        """
        if (
            self._lazy_load_write_detection_period_seconds is None
            or self._lazy_load_write_detection_period_seconds <= 0
        ):
            return False

        # Get the current directory size
        size = self._get_total_disk_size(model_dir)
        # Sleep a bit to wait out another write
        time.sleep(self._lazy_load_write_detection_period_seconds)
        # Get the size again. If it has changed, then a write is currently  in progress
        return self._get_total_disk_size(model_dir) != size

    @staticmethod
    def _get_total_disk_size(model_dir: str) -> int:
        """Returns the sum of st_size of all files contained within the directory structure rooted
        at model_dir.
        """
        dir_path = Path(model_dir)
        return sum([f.stat().st_size for f in dir_path.rglob("*") if f.is_file()])

    def __report_total_model_size_metric(self):
        # Just a happy little lock to ensure that with concurrent loading and unloading,
        # the last metric reported will be correct.
        with self.__model_size_gauge_lock:
            cnt = DictCounter()
            for model in self.loaded_models.values():
                cnt[model.type()] += model.size()

            for model_type, total_size in cnt.items():
                MODEL_SIZE_GAUGE.labels(model_type=model_type).set(total_size)

    @staticmethod
    def __increment_model_count_metric(model_type, model_id):
        MODEL_COUNT_GAUGE.labels(model_type=model_type, model_id=model_id).inc()

    @staticmethod
    def __decrement_model_count_metric(model_type, model_id):
        MODEL_COUNT_GAUGE.labels(model_type=model_type, model_id=model_id).dec()

    @staticmethod
    def __increment_load_model_exception_count_metric(model_type):
        LOAD_MODEL_EXCEPTION_COUNTER.labels(model_type=model_type).inc()
