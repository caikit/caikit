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
from typing import Optional, Union

# Third Party
from prometheus_client import Summary

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER, ModuleBase
from caikit.core.model_management import ModelFinderBase, ModelInitializerBase
from caikit.runtime.model_management.model_loader_base import ModelLoaderBase

log = alog.use_channel("MODEL-LOADER")

CAIKIT_CORE_LOAD_DURATION_SUMMARY = Summary(
    "caikit_core_load_model_duration_seconds",
    "Summary of the duration (in seconds) of caikit.core.load(model)",
    ["model_type"],
)


class CoreModelLoader(ModelLoaderBase):
    """The CoreModelLoader loads a model using the caikit core.ModelManager"""

    name = "CORE"

    def load_module_instance(
        self,
        model_path: str,
        model_id: str,
        model_type: str,
        finder: Optional[Union[str, ModelFinderBase]] = None,
        initializer: Optional[Union[str, ModelInitializerBase]] = None,
    ) -> ModuleBase:
        """Start loading a model from disk and associate the ID/size with it"""
        log.info("<RUN89711114I>", "Loading model '%s'", model_id)

        # Only pass finder/initializer if they have values so that defaults are used otherwise
        load_kwargs = {}
        if finder:
            load_kwargs["finder"] = finder
        if initializer:
            load_kwargs["initializer"] = initializer

        # Load using the caikit.core
        with CAIKIT_CORE_LOAD_DURATION_SUMMARY.labels(model_type=model_type).time():
            return MODEL_MANAGER.load(model_path, **load_kwargs)
