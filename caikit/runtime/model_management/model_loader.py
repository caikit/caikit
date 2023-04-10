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
from typing import Union

# Third Party
from grpc import StatusCode
from prometheus_client import Summary

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.module_backend_config import configure
from caikit.runtime.model_management.batcher import Batcher
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.config_parser import ConfigParser
import caikit.core

log = alog.use_channel("MODEL-LOADER")

CAIKIT_CORE_LOAD_DURATION_SUMMARY = Summary(
    "caikit_core_load_model_duration_seconds",
    "Summary of the duration (in seconds) of caikit.core.load(model)",
    ["model_type"],
)


class ModelLoader:
    """Model Loader class. The singleton class contains the core implementation details
    for loading models in from S3."""

    __instance = None

    def __init__(self):
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        ModelLoader.__instance = self
        self.config_parser = ConfigParser.get_instance()

        # If distributed loading is enabled, set up the loader
        if self.config_parser.distributed.enabled:
            log.info("<RUN89711118I>", "Configuring for distributed loading")
            log.debug("Distributed config: %s", self.config_parser.distributed.config)
            configure(**self.config_parser.distributed.config)

    def load_model(self, model_id, local_model_path, model_type) -> LoadedModel:
        """Load a model using model_path (in Cloud Object Storage) & give it a model ID.
        This always cleans up files on disk, no matter if the load succeeds or fails

        Args:
            model_id (string):  Model ID string for the model to load.
            local_model_path (string): Path to COS to load the model from.
            model_type (string): Type of the model to load.
        Returns:
            model (LoadedModel) : The model that was loaded
        """
        with CAIKIT_CORE_LOAD_DURATION_SUMMARY.labels(model_type=model_type).time():
            caikit_core_model = self._load_module(
                local_model_path, model_id, model_type
            )

        model = (
            LoadedModel.Builder()
            .id(model_id)
            .type(model_type)
            .path(local_model_path)
            .module(caikit_core_model)
            .build()
        )

        return model

    def _load_module(self, model_path, model_id, model_type):
        try:
            log.info("<RUN89711114I>", "Loading model '%s'", model_id)

            # Load using the caikit.core
            model = caikit.core.load(model_path)

            # If this model needs batching, configure a Batcher to wrap it
            model = self._wrap_in_batcher_if_configured(
                model,
                model_type,
                model_id,
            )
        except FileNotFoundError as fnfe:
            log_dict = {
                "log_code": "<RUN98613924E>",
                "message": "load failed to find model: %s with error: %s"
                % (model_path, repr(fnfe)),
                "model_id": model_id,
            }
            log.error(log_dict)
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND,
                f"Model {model_id} not found. Nested error: {fnfe}",
            ) from fnfe
        except Exception as ex:
            log_dict = {
                "log_code": "<RUN62912924E>",
                "message": "load failed when processing path: %s with error: %s"
                % (model_path, repr(ex)),
                "model_id": model_id,
            }
            log.error(log_dict)
            raise CaikitRuntimeException(
                StatusCode.INTERNAL,
                f"Model {model_id} failed to load. Nested error: {ex}",
            ) from ex

        cache_info = MODEL_MANAGER.get_singleton_model_cache_info()
        log.info("<RUN89713784I>", "Singleton cache: '%s'", str(cache_info))

        return model

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """This method returns the instance of Model Manager"""
        if not cls.__instance:
            cls.__instance = ModelLoader()
        return cls.__instance

    def _wrap_in_batcher_if_configured(
        self,
        caikit_core_model: caikit.core.ModuleBase,
        model_type: str,
        model_id: str,
    ) -> Union[Batcher, caikit.core.ModuleBase]:
        """Perform Batcher wrapping on the given module if configured, otherwise
        return the model as is
        """
        batch_config = self.config_parser.batching.get(
            model_type,
            self.config_parser.batching.get("default", {}),
        )
        log.debug2("Batch config for model type [%s]: %s", model_type, batch_config)
        batch_size = batch_config.get("size", 0)
        if batch_size > 0:
            log.info(
                "<RUN89713768I>",
                "Enabling batch size [%s] for [%s] of type [%s]",
                batch_size,
                model_id,
                model_type,
            )
            return Batcher(
                model_name=model_id,
                model=caikit_core_model,
                batch_size=batch_size,
                batch_collect_delay_s=batch_config.get("collect_delay_s"),
            )
        return caikit_core_model
