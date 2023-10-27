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
from typing import Dict, Union

# Third Party
import importlib_metadata

# First Party
import alog

# Local
from caikit.config import get_config
from caikit.interfaces.runtime.data_model import RuntimeInfoResponse

log = alog.use_channel("RI-SERVICR-I")


class InfoServicer:
    """This class contains the implementation for retrieving information about the
    library and services."""

    def GetRuntimeInfo(
        self, request, context  # pylint: disable=unused-argument
    ) -> RuntimeInfoResponse:
        """Get information on versions of libraries and server for GRPC"""
        return self.get_runtime_info_impl().to_proto()

    def get_runtime_info_impl(self) -> RuntimeInfoResponse:
        """Get information on versions of libraries and server from config"""
        python_packages = {}
        config_version_info = get_config().runtime.version_info or {}
        all_packages = (config_version_info.get("python_packages") or {}).get("all")

        for lib, dist_names in importlib_metadata.packages_distributions().items():
            if (
                (all_packages or (len(lib.split(".")) == 1 and lib.startswith("caikit")))
                and version := self.try_lib_version(dist_names[0])
            ):
                python_packages[lib] = version

        runtime_image = config_version_info.get("runtime_image")

        return RuntimeInfoResponse(
            python_packages=python_packages,
            runtime_version=runtime_image,
        )

    def get_version_dict(self) -> Dict[str, Union[str, Dict]]:
        """Get information on versions of libraries and server for HTTP"""
        return self.get_runtime_info_impl().to_dict()

    def try_lib_version(self, name) -> str:
        """Get version of python modules"""
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None
