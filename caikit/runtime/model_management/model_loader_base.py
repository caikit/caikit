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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional, Union
import abc

# Third Party
from grpc import StatusCode

# First Party
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.core import MODEL_MANAGER, ModuleBase
from caikit.core.model_management import ModelFinderBase, ModelInitializerBase
from caikit.core.toolkit.factory import FactoryConstructible
from caikit.runtime.model_management.batcher import Batcher
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("MODEL-LOADER")


class ModelLoaderBase(FactoryConstructible):
    """Model Loader Base class which describes how models are loaded."""

    _load_thread_pool = None

    def __init__(self, config: aconfig.Config, instance_name: str):
        """A FactoryConstructible object must be constructed with a config
        object that it uses to pull in all configuration
        """
        if ModelLoaderBase._load_thread_pool is None:
            ModelLoaderBase._load_thread_pool = ThreadPoolExecutor(
                get_config().runtime.load_threads
            )

        super().__init__(config, instance_name)
        # Instead of storing config-based batching information here, we call
        # get_config() when needed to support dynamic config changes for
        # batching

    @abc.abstractmethod
    def load_module_instance(
        self,
        model_path: str,
        model_id: str,
        model_type: str,
        finder: Optional[Union[str, ModelFinderBase]] = None,
        initializer: Optional[Union[str, ModelInitializerBase]] = None,
    ) -> ModuleBase:
        """Load an instance of a Caikit Model

        Args:
            model_path (str): The model path to load from
            model_id (str): The model's id
            model_type (str): The type of model being load
            finder (Optional[Union[str, ModelFinderBase]], optional): The ModelFinder to use for
              loading. Defaults to None.
            initializer (Optional[Union[str, ModelInitializerBase]], optional): The
              ModelInitializer to use for loading. Defaults to None.

        Returns:
            ModuleBase: a loaded model
        """

    def load_model(
        self,
        model_id: str,
        local_model_path: str,
        model_type: str,
        fail_callback: Optional[Callable] = None,
        retries: int = 0,
        finder: Optional[Union[str, ModelFinderBase]] = None,
        initializer: Optional[Union[str, ModelInitializerBase]] = None,
    ) -> LoadedModel:
        """Start loading a model from disk and associate the ID/size with it

        Args:
            model_id (str): Model ID string for the model to load.
            local_model_path (str): Local filesystem path to load the model from.
            model_type (str): Type of the model to load.
            fail_callback (Optional[Callable]): Optional no-arg callback to call
                on load failure
            retries (int): Number of times to retry loading
        Returns:
            model (LoadedModel) : The model that was loaded
        """
        # Set up the basics of the model's metadata
        model_builder = (
            LoadedModel.Builder()
            .id(model_id)
            .type(model_type)
            .path(local_model_path)
            .fail_callback(fail_callback)
            .retries(retries)
        )

        # Set up the async loading
        args = (local_model_path, model_id, model_type, finder, initializer)
        log.debug2("Loading model %s async", model_id)
        future_factory = partial(
            self._load_thread_pool.submit, self._wrapped_load_model, *args
        )
        model_builder.model_future_factory(future_factory)

        # Return the built model with the future handle
        return model_builder.build()

    def _wrapped_load_model(
        self,
        model_path: str,
        model_id: str,
        model_type: str,
        finder: Optional[Union[str, ModelFinderBase]] = None,
        initializer: Optional[Union[str, ModelInitializerBase]] = None,
    ) -> Union[Batcher, ModuleBase]:
        try:
            log.info("<RUN89711114I>", "Loading model '%s'", model_id)

            model = self.load_module_instance(
                model_path, model_id, model_type, finder, initializer
            )

            # If this model needs batching, configure a Batcher to wrap it
            model = self._wrap_in_batcher_if_configured(
                model,
                model_type,
                model_id,
            )
        except CaikitRuntimeException as cre:
            log_dict = {
                "log_code": "<RUN98613924E>",
                "message": f"load failed to load model: {model_path} with error: {repr(cre)}",
                "model_id": model_id,
            }
            log.error(log_dict)
            raise cre
        except FileNotFoundError as fnfe:
            log_dict = {
                "log_code": "<RUN98613924E>",
                "message": f"load failed to find model: {model_path} with error: {repr(fnfe)}",
                "model_id": model_id,
            }
            log.error(log_dict)
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND,
                f"Model {model_id} not found. Nested error: {fnfe}",
            ) from fnfe
        except ValueError as ve:
            log_dict = {
                "log_code": "<RUN38617724E>",
                "message": f"load failed to find model: {model_path} with error: {repr(ve)}",
                "model_id": model_id,
            }
            log.error(log_dict)
            raise CaikitRuntimeException(
                StatusCode.NOT_FOUND,
                f"Model {model_id} not found. Nested error: {ve}",
            ) from ve
        except Exception as ex:
            log_dict = {
                "log_code": "<RUN62912924E>",
                "message": f"load failed when processing path: {model_path} with error: {repr(ex)}",
                "model_id": model_id,
            }
            log.error(log_dict, exc_info=True)
            raise CaikitRuntimeException(
                StatusCode.INTERNAL,
                f"Model {model_id} failed to load. Nested error: {ex}",
            ) from ex

        cache_info = MODEL_MANAGER.get_singleton_model_cache_info()
        log.info("<RUN89713784I>", "Singleton cache: '%s'", str(cache_info))

        return model

    def _wrap_in_batcher_if_configured(
        self,
        caikit_core_model: ModuleBase,
        model_type: str,
        model_id: str,
    ) -> Union[Batcher, ModuleBase]:
        """Perform Batcher wrapping on the given module if configured, otherwise
        return the model as is
        """
        batch_config = get_config().runtime.batching.get(
            model_type,
            get_config().runtime.batching.get("default", {}),
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
