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
from importlib.metadata import version as pip_version

# Third Party
import grpc

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER
from caikit.interfaces.runtime.data_model import (
    RuntimeInfoStatusResponse,
    RuntimeInfoStatusResponseDict,
)
from caikit.config import get_config

log = alog.use_channel("RI-SERVICR-I")


class RuntimeInfoServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    def GetRuntimeInfo(self, request, context):
        """Get information on versions of libraries and server from config"""
        version_dict = {}
        # TODO: how to get library version for extensions?
        # version_dict["caikit_nlp_version"] = pip_version("caikit_nlp")
        version_dict["caikit_version"] = pip_version("caikit")
        # TODO: why is cls.config != get_config() within this method --> fails with error  AttributeError: type object 'RuntimeHTTPServer' has no attribute 'config'
        print("DOES VERSIONING EXIST", get_config().runtime.versioning)

        if get_config().runtime.versioning:
            print("GET_CONFIG", get_config().runtime.versioning)
            # TODO: how to get library that is being run -- aka get caikit_nlp part dynamically
            version_dict.update(get_config().runtime.versioning)

        # return version_dict

        # return RuntimeInfoStatusResponseDict(
        #     version_info=version_dict
        # ).to_proto()


        return RuntimeInfoStatusResponse(
            caikit_version=version_dict["caikit_version"],
            runtime_image_version=version_dict.get("runtime_image"),
            version_info=version_dict,
        ).to_proto()