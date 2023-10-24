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

# Have pylint ignore Class XXXX has no YYYY member so that we can use gRPC enums.
# pylint: disable=E1101

# Standard
from typing import Dict
import importlib.metadata
import sys

# First Party
import alog

# Local
from caikit.config import get_config
from caikit.interfaces.runtime.data_model import RuntimeInfoResponse

log = alog.use_channel("RI-SERVICR-I")


class RuntimeInfoServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    def GetRuntimeInfo(self, request, context) -> RuntimeInfoResponse:  # pylint: disable=unused-argument
        """Get information on versions of libraries and server for GRPC"""
        return self.get_runtime_info_impl()

    def get_runtime_info_impl(self) -> RuntimeInfoResponse:
        """Get information on versions of libraries and server from config"""
        versions = {}
        for lib in sys.modules:
            if (
                get_config().runtime.versioning
                and get_config().runtime.versioning.sys_modules
            ):
                if len(lib.split(".")) == 1:
                    version = self.try_lib_version(lib)
                    if version:
                        versions[lib] = version
            # just get caikit versions
            else:
                if len(lib.split(".")) == 1 and lib.startswith("caikit"):
                    versions[lib] = self.try_lib_version(lib)

        if get_config().runtime.versioning:
            versions["runtime_image"] = get_config().runtime.versioning.get(
                "runtime_image"
            )

        return RuntimeInfoResponse(
            version_info=versions,
        ).to_proto()

    def get_version_dict(self) -> Dict[str, str]:
        """Get information on versions of libraries and server for HTTP"""
        runtime_info_response = self.get_runtime_info_impl()
        return runtime_info_response.version_info

    # TODO: fix so can get versions for something like alog --> alchemy_logging
    def try_lib_version(self, name) -> str:
        """Get version of python modules"""
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
