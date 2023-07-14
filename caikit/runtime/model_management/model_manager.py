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
from typing import Dict
import gc
import os
import shutil
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


class ModelManager:
    """Model Manager class. The singleton class contains the core implementational details
    for the Model Runtime (i.e., load/unload functionality, etc). It does not provide the core
    details for predict calls."""

    __instance = None

    __model_size_gauge_lock = threading.Lock()

    # Model type name used when loading local models
    _STANDALONE_MODEL = "standalone-model"

    def __init__(self):
        """Initialize a ModelManager instance."""
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        ModelManager.__instance = self

        # Pull in a ModelLoader
        self.model_loader = ModelLoader.get_instance()
        # And a ModelSizer
        self.model_sizer = ModelSizer.get_instance()

        # set up the local map for model ID to loaded model
        self.loaded_models: Dict[str, LoadedModel] = {}

        # Keep track of the local model cache dir and make sure it's valid
        runtime_cfg = get_config().runtime
        self._local_models_cache_dir = runtime_cfg.local_models_cache_dir
        if self._local_models_cache_dir:
            error.value_check(
                "<RUN75903138E>", os.path.isdir(self._local_models_cache_dir)
            )
        self._unload_local_models_cache = runtime_cfg.unload_local_models_cache

        # Optionally load models mounted into a local directory
        local_models_dir = runtime_cfg.local_models_dir
        if os.path.exists(local_models_dir) and len(os.listdir(local_models_dir)) > 0:
            log.info("<RUN44739400I>", "Loading local models into Caikit Runtime...")
            self.load_local_models(local_models_dir)

        # If the local_models_dir and local_models_cache_dir overlap and purging
        # is enabled, raise a big warning!
        if (
            self._local_models_cache_dir
            and local_models_dir == self._local_models_cache_dir
            and self._unload_local_models_cache
        ):
            log.warning(
                "<RUN41922990W>",
                "WARNING! Running with unsafe unloading may cause model artifact loss. "
                + "Use local_models_dir != local_models_cache_dir to avoid this",
            )

    def load_model(
        self,
        model_id: str,
        local_model_path: str,
        model_type: str,
    ) -> int:
        """Load a model using model_path (in Cloud Object Storage) & give it a model ID
        Args:
            model_id (string):  Model ID string for the model to load.
            local_model_path (string): Local path to load the model from.
            model_type (string): Type of the model to load.
        Returns:
            Model_size (int) : Size of the loaded model in bytes
        """
        with LOAD_MODEL_DURATION_SUMMARY.labels(model_type=model_type).time():
            if model_id in self.loaded_models:
                log.debug("Model '%s' is already loaded", model_id)
            else:
                try:
                    model = self.model_loader.load_model(
                        model_id, local_model_path, model_type
                    )
                except Exception as ex:
                    self.__increment_load_model_exception_count_metric(model_type)
                    raise ex

                model_size = self.model_sizer.get_model_size(
                    model_id, local_model_path, model_type
                )
                model.set_size(model_size)

                # Add model + helpful metadata to our loaded models map
                self.loaded_models[model_id] = model

                # Update Prometheus metrics
                self.__increment_model_count_metric(model_type, model_id)
                self.__report_total_model_size_metric()

            # If using a local models cache, save it to the cache.
            #
            # NOTE: This assumes immutable model ids! If a model id is reused
            #   with different content, this will not update the content in the
            #   cache directory.
            if self._local_models_cache_dir:
                model_cache_path = os.path.join(self._local_models_cache_dir, model_id)
                if not os.path.exists(model_cache_path):
                    log.debug("Caching local model %s", model_id)
                    model.module().save(model_cache_path)

            return self.loaded_models[model_id].size()

    def load_local_models(self, local_models_dir: str):
        """Load models mounted into a local directory

        Args:
            local_models_dir (str): The directory where local models are stored
        """
        for model_base_path in os.listdir(local_models_dir):
            try:
                self.load_local_model(model_base_path, local_models_dir)
            except CaikitRuntimeException as err:
                log.warning(
                    "<RUN56627484W>",
                    "Failed to load model %s: %s",
                    model_base_path,
                    repr(err),
                    exc_info=True,
                )
                continue

        if len(self.loaded_models) == 0:
            log.error(
                "<RUN56336804E>", "No models loaded in directory: %s", local_models_dir
            )
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "No standalone models loaded"
            )

    def load_local_model(self, model_id: str, local_models_dir: str):
        """Try to load a model in local_models_dir by its id

        Args:
            model_id (str): Model ID that is the name of the directory or zip in
                the local_models_dir
            local_models_dir (str): The directory where local models are stored
        """
        # Use the file name as the model id
        model_path = os.path.join(local_models_dir, model_id)
        self.load_model(model_id, model_path, self._STANDALONE_MODEL)

    def unload_model(self, model_id: str) -> int:
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
            del self.loaded_models[model_id]
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

        # If using a local model cache and purging is enabled, delete the cached
        # model from disk
        if self._local_models_cache_dir and self._unload_local_models_cache:
            cache_model_path = os.path.join(self._local_models_cache_dir, model_id)
            if os.path.exists(cache_model_path):
                log.info(
                    "<RUN21819699I>",
                    "Purging cache model %s at %s",
                    model_id,
                    cache_model_path,
                )
                if os.path.isdir(cache_model_path):
                    shutil.rmtree(cache_model_path)
                else:
                    os.remove(cache_model_path)

        return model_size

    def unload_all_models(self):
        """Unload all loaded models. This will also remove any lingering artifacts from strangely
        packaged zip files, which initially resulted in load failures."""
        try:
            self.loaded_models.clear()
        except Exception as ex:
            log.debug("Unload all models failed with error: %s", repr(ex))
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "All models could not be unloaded!!"
            ) from ex

    def get_model_size(self, model_id: str) -> int:
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

    def estimate_model_size(
        self, model_id: str, local_model_path: str, model_type: str
    ) -> int:
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
            model_id(string): Model ID of the model to retrieve
        Returns:
            response (caikit.core.module.ModuleBase):
                A loaded Caikit model
        """
        if not model_id:
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT, "Missing required model ID"
            )

        # Now retrieve the model
        found = model_id in self.loaded_models

        # If enabled, try to load the model from local_models_dir
        if not found and self._local_models_cache_dir:
            log.debug("Attempting to lazily load local model %s", model_id)
            try:
                self.load_local_model(model_id, self._local_models_cache_dir)
                log.info("<RUN75038623I>", "Lazily loaded %s", model_id)
                found = True
            except CaikitRuntimeException as err:
                log.debug2(
                    "Unable to lazily load %s: %s",
                    model_id,
                    repr(err),
                    exc_info=True,
                )

        # If still not found, log and return NOT_FOUND
        if not found:
            msg = f"Model '{model_id}' not loaded"
            log.debug(
                {"log_code": "<RUN61105243D>", "message": msg, "model_id": model_id}
            )
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND, msg, {"model_id": model_id}
            )

        # Return the loaded model
        return self.loaded_models[model_id].module()

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
