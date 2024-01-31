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
This module contains the implementation for retrieving information about the
library and services.
"""
# Have pylint ignore Class XXXX has no YYYY member so that we can use gRPC enums.
# pylint: disable=E1101

# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from grpc import StatusCode
import importlib_metadata

# First Party
import alog

# Local
from caikit.config import get_config
from caikit.interfaces.runtime.data_model import (
    ModelInfo,
    ModelInfoRequest,
    ModelInfoResponse,
    RuntimeInfoResponse,
)
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("RI-SERVICR-I")


class InfoServicer:
    """This class contains the implementation for retrieving information about the
    library and services."""

    def GetModelsInfo(
        self, request: ModelInfoRequest, context  # pylint: disable=unused-argument
    ) -> ModelInfoResponse:
        """Get information on the loaded models for the GRPC server

        Args:
            request: ModelInfoRequest
            context

        Returns:
            models_info: ModelInfoResponse
                DataObject containing the model info
        """
        return self._get_models_info(model_ids=request.model_ids).to_proto()

    def get_models_info_dict(
        self, model_ids: Optional[List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get information on models for the HTTP server

        Returns:
            model_info_dict: Dict[str, List[Dict[str, str]]]
                Dict representation of ModelInfoResponse
        """
        return self._get_models_info(model_ids=model_ids).to_dict()

    def _get_models_info(
        self, model_ids: Optional[List[str]] = None
    ) -> ModelInfoResponse:
        """Helper function to get the list of models

        Returns:
            model_info: ModelInfoResponse
                DataObject with model information
        """
        model_manager = ModelManager.get_instance()

        # Get list of models based on input list or all loaded models
        loaded_model_list = []
        if model_ids:
            for model_name in model_ids:
                loaded_model = model_manager.loaded_models.get(model_name)
                if not loaded_model:
                    raise CaikitRuntimeException(
                        StatusCode.NOT_FOUND, f"Model {model_name} is not loaded"
                    )

                loaded_model_list.append((model_name, loaded_model))
        else:
            loaded_model_list = model_manager.loaded_models.items()

        # Get all loaded models
        response = ModelInfoResponse(models=[])
        for name, loaded_module in loaded_model_list:
            model_instance = loaded_module.model()
            response.models.append(
                ModelInfo(
                    model_path=loaded_module.path(),
                    name=name,
                    size=loaded_module.size(),
                    metadata=model_instance.public_model_info,
                    loaded=loaded_module.loaded(),
                    module_id=model_instance.MODULE_ID,
                    module_metadata=model_instance.module_metadata,
                )
            )
        return response

    def GetRuntimeInfo(
        self, request, context  # pylint: disable=unused-argument
    ) -> RuntimeInfoResponse:
        """Get information on versions of libraries and server for GRPC"""
        return self._get_runtime_info().to_proto()

    def get_version_dict(self) -> Dict[str, Union[str, Dict]]:
        """Get information on versions of libraries and server for HTTP"""
        return self._get_runtime_info().to_dict()

    def _get_runtime_info(self) -> RuntimeInfoResponse:
        """Get information on versions of libraries and server from config"""
        config_version_info = get_config().runtime.version_info or {}
        python_packages = {
            package: version
            for package, version in config_version_info.get(
                "python_packages", {}
            ).items()
            if package != "all"
        }
        all_packages = (config_version_info.get("python_packages") or {}).get("all")

        for lib, dist_names in importlib_metadata.packages_distributions().items():
            if (
                all_packages or (len(lib.split(".")) == 1 and lib.startswith("caikit"))
            ) and (version := self._try_lib_version(dist_names[0])):
                python_packages[lib] = version

        runtime_image = config_version_info.get("runtime_image")

        return RuntimeInfoResponse(
            python_packages=python_packages,
            runtime_version=runtime_image,
        )

    def _try_lib_version(self, name) -> str:
        """Get version of python modules"""
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None
