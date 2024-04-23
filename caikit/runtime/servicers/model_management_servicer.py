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
The Model Management Service is responsible for deploying and undeploying models
"""

# Third Party
import grpc

# First Party
import alog

# Local
from caikit.interfaces.runtime.data_model import (
    DeployModelRequest,
    ModelInfo,
    UndeployModelRequest,
)
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("MM-SERVICR-I")


# Define types for the proto versions of the DM classes
DeployModelRequestProto = DeployModelRequest.get_proto_class()
ModelInfoProto = ModelInfo.get_proto_class()
UndeployModelRequestProto = UndeployModelRequest.get_proto_class()


class ModelManagementServicerImpl:
    __doc__ = __doc__

    def __init__(self):
        self._model_manager = ModelManager.get_instance()

    def DeployModel(
        self,
        request: DeployModelRequestProto,  # type: ignore
        context: grpc.RpcContext,  # pylint: disable=unused-argument
    ) -> ModelInfoProto:  # type: ignore
        """Deploy a model to the runtime"""
        if not request.model_name:
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Must provide model_name",
            )
        if not request.model_files or any(
            not f.filename.strip() for f in request.model_files
        ):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Must provide at least one model_files entry and all must be valid file names",
            )

        # Deploy the model to the model manager
        loaded_model = self._model_manager.deploy_model(
            model_id=request.model_name,
            model_files={f.filename: f.data for f in request.model_files},
            wait=False,
        )

        # Return the model info
        return ModelInfo(
            model_path=loaded_model.path(),
            name=loaded_model.id(),
            size=loaded_model.size(),
            loaded=loaded_model.loaded(),
        ).to_proto()

    def UndeployModel(
        self,
        request: UndeployModelRequestProto,  # type: ignore
        context: grpc.RpcContext,  # pylint: disable=unused-argument
    ) -> UndeployModelRequestProto:  # type: ignore
        """Un-deploy a model to the runtime"""
        if not request.model_name:
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Must provide model_name",
            )
        self._model_manager.undeploy_model(request.model_name)
        return UndeployModelRequestProto(model_name=request.model_name)
