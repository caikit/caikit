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
from typing import Dict, Optional
import atexit
import gc
import os
import threading

# Third Party
from grpc import StatusCode
from prometheus_client import Counter, Gauge, Summary

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import ModuleBase
from caikit.core.toolkit.errors import error_handler
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_loader import ModelLoader
from caikit.runtime.model_management.model_sizer import ModelSizer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.work_management.abortable_action import ActionAborter

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
        error.value_check(
            "<RUN44773514E>",
            not self._lazy_load_local_models or self._local_models_dir,
            "Must set runtime.local_models_dir with runtime.lazy_load_local_models",
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
        if self._enable_lazy_load_poll:
            atexit.register(self.shut_down)

        # Do the initial local models load
        if self._local_models_dir:
            log.info("<RUN44739400I>", "Loading local models into Caikit Runtime...")
            self.sync_local_models(wait=True)

    def shut_down(self):
        """Shut down cache purging"""
        timer = getattr(self, "_lazy_sync_timer", None)
        if timer is not None:
            timer.cancel()

    def load_model(
        self,
        model_id: str,
        local_model_path: str,
        model_type: str,
        wait: bool = True,
        aborter: Optional[ActionAborter] = None,
    ) -> int:
        """Load a model using model_path (in Cloud Object Storage) & give it a model ID
        Args:
            model_id (str):  Model ID string for the model to load.
            local_model_path (str): Local path to load the model from.
            model_type (str): Type of the model to load.
            wait (bool): Wait for the model to finish loading
        Returns:
            Model_size (int) : Size of the loaded model in bytes
        """
        with LOAD_MODEL_DURATION_SUMMARY.labels(model_type=model_type).time():

            # If already loaded, just return the size
            model = self.loaded_models.get(model_id)
            if model is not None:
                log.debug("Model '%s' is already loaded", model_id)
                return model.size()

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
                            aborter=aborter,
                            fail_callback=partial(self.unload_model, model_id),
                        )
                    except Exception as ex:
                        self.__increment_load_model_exception_count_metric(model_type)
                        raise ex

                    # Estimate the model's size and update the LoadedModel
                    model_size = self.model_sizer.get_model_size(
                        model_id, local_model_path, model_type
                    )
                    model.set_size(model_size)

                    # Add model + helpful metadata to our loaded models map
                    self.loaded_models[model_id] = model

                    # Update Prometheus metrics
                    self.__increment_model_count_metric(model_type, model_id)
                    self.__report_total_model_size_metric()

            # If waiting, do so outside of the mutation lock
            if wait:
                model.wait()

            # Return the model's size
            return model.size()

    def sync_local_models(self, wait: bool = False):
        """Sync in-memory models with models in the configured local_model_dir

        New models will be loaded and models previously loaded from local will
        be unloaded.

        Args:
            wait (bool): Wait for loading to complete
        """
        if not self._local_models_dir:
            return

        # Get the list of models on disk
        # NOTE: If the local_models_dir has disappeared, this is likely a unit
        #   test with a temp dir, but in any event, we should stop trying to
        #   sync going forward
        try:
            disk_models = os.listdir(self._local_models_dir)
        except FileNotFoundError:
            return

        # Find all models that aren't currently loaded
        # NOTE: If one of these models gets loaded after this check, that will
        #   be handled in load_models
        new_models = [
            model_id for model_id in disk_models if model_id not in self.loaded_models
        ]
        log.debug("New local models: %s", new_models)

        # Find all models that are currently loaded from the local models dir
        # that no longer exist
        unload_models = [
            model_id
            for model_id, loaded_model in self.loaded_models.items()
            if model_id not in disk_models
            and loaded_model.path().startswith(
                self._local_models_dir,
            )
        ]
        log.debug("Unloaded local models: %s", unload_models)

        # Load new models
        for model_id in new_models:
            try:
                # Use the file name as the model id
                model_path = os.path.join(self._local_models_dir, model_id)
                self.load_model(
                    model_id, model_path, self._LOCAL_MODEL_TYPE, wait=False
                )
            except CaikitRuntimeException as err:
                log.warning(
                    "<RUN56627484W>",
                    "Failed to load model %s: %s",
                    model_id,
                    repr(err),
                    exc_info=True,
                )
                continue

        # Unload old models
        for model_id in unload_models:
            log.debug2("Unloading local model %s", model_id)
            self.unload_model(model_id)

        # Wait for models to load and purge out any that failed
        if wait:
            for model_id in new_models:
                loaded_model = self.loaded_models.get(model_id)
                # If somehow already purged, there's nothing to wait on
                if loaded_model is None:
                    continue
                # Wait for it and make sure it didn't fail
                try:
                    loaded_model.wait()
                except CaikitRuntimeException as err:
                    log.warning(
                        "<RUN56627485W>",
                        "Failed to load model %s: %s",
                        model_id,
                        repr(err),
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
                log.debug2("Canceling live timer")
                self._lazy_sync_timer.cancel()
            self._lazy_sync_timer = threading.Timer(
                self._lazy_load_poll_period_seconds, self.sync_local_models
            )
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
        # If the model failed to load, just return 0; no need to throw an error here.
        if model_id not in self.loaded_models:
            log.debug("Model '%s' is not loaded, so it cannot be unloaded!", model_id)
            return 0

        # Temporarily store model size and type info
        model_type = self.loaded_models[model_id].type()
        model_size = self.loaded_models[model_id].size()

        # Delete the model and remove it from the model map
        try:
            model = self.loaded_models.pop(model_id)
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

    def unload_all_models(self) -> None:
        """Unload all loaded models. This will also remove any lingering artifacts from strangely
        packaged zip files, which initially resulted in load failures."""
        try:
            self.loaded_models.clear()
        except Exception as ex:
            log.debug("Unload all models failed with error: %s", repr(ex))
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "All models could not be unloaded!!"
            ) from ex

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

        # Model sizes should all be cached
        if not loaded_model.has_size():
            # This really shouldn't happen, because we size the models on load
            log.warning(
                "<RUN61106343W>",
                "Loaded model %s unexpectedly did not have a size. Sizing it again now.",
                model_id,
            )
            loaded_model.set_size(
                self.model_sizer.get_model_size(
                    model_id, loaded_model.path(), loaded_model.type()
                )
            )
            self.loaded_models[model_id] = loaded_model

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

    def retrieve_model(self, model_id) -> ModuleBase:
        """Retrieve a model from the loaded model map by model ID.

        Args:
            model_id(string): Model ID of the model to retrieve
        Returns:
            response (caikit.core.module.ModuleBase):
                A loaded Caikit model
        """
        if not model_id:
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT, "Missing required model ID"
            )

        # Now retrieve the model and fall back to lazy loading
        model_loaded = model_id in self.loaded_models
        if not model_loaded and self._lazy_load_local_models:
            local_model_path = os.path.join(self._local_models_dir, model_id)
            if os.path.exists(local_model_path):
                log.debug2(
                    "Lazy loading local model %s from %s", model_id, local_model_path
                )
                self.load_model(
                    model_id=model_id,
                    local_model_path=local_model_path,
                    model_type=self._LOCAL_MODEL_TYPE,
                    wait=True,
                )
                model_loaded = True

        # If still not loaded, there's nothing to find, so raise NOT_FOUND
        if not model_loaded:
            msg = "Model '%s' not loaded" % model_id
            log.debug(
                {"log_code": "<RUN61105243D>", "message": msg, "model_id": model_id}
            )
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND, msg, {"model_id": model_id}
            )

        # NOTE: If the model is partially loaded, this call will wait on the
        #   model future in the LoadedModel
        return self.loaded_models[model_id].model()

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

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """This method returns the instance of Model Manager"""
        if not cls.__instance:
            cls.__instance = ModelManager()
        return cls.__instance
