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

# First Party
import alog

# Local
from caikit import get_config
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.protobufs import model_runtime_pb2, model_runtime_pb2_grpc
from caikit.runtime.types.aborted_exception import AbortedException
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.work_management.abortable_action import AbortableAction
from caikit.runtime.work_management.rpc_aborter import RpcAborter

log = alog.use_channel("MR-SERVICR-I")


class ModelRuntimeServicerImpl(model_runtime_pb2_grpc.ModelRuntimeServicer):
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    def __init__(self):
        self.model_manager = ModelManager.get_instance()

    def loadModel(self, request, context):
        """Model loading .
        Args:
            request(model_runtime_pb2.LoadModelRequest):
                gRPC request, gen from model-runtime.proto
            context(grpc.ServicerContext):
                Context object (contains request metadata, etc)
        Returns:
            model_runtime_pb2.LoadModelResponse: Gen from model-runtime.proto
        """
        try:
            log.info(
                {
                    "log_code": "<RUN10000106I>",
                    "message": "Loading model '%s'" % request.modelId,
                    "model_id": request.modelId,
                }
            )
            caikit_config = get_config()
            if caikit_config.runtime.use_abortable_threads:
                aborter = RpcAborter(context)
                work = AbortableAction(
                    aborter,
                    self.model_manager.load_model,
                    request.modelId,
                    request.modelPath,
                    request.modelType,
                    aborter=aborter,
                )
                model_size = work.do()
            else:
                model_size = self.model_manager.load_model(
                    request.modelId, request.modelPath, request.modelType
                )

            log.info(
                {
                    "log_code": "<RUN10000107I>",
                    "message": "Model '%s' loaded! Model size [%s]"
                    % (request.modelId, str(model_size)),
                    "model_id": request.modelId,
                }
            )

        except AbortedException as e:
            log.warning(
                {
                    "log_code": "<RUN82590507W>",
                    "message": "Model '%s' was not loaded due to the rpc aborting"
                    % request.modelId,
                    "model_id": request.modelId,
                    "error_id": e.id,
                }
            )

            # Unload the model in case it had actually finished loading
            self.model_manager.unload_model(request.modelId)

            raise e

        except CaikitRuntimeException as e:
            log.warning(
                {
                    "log_code": "<RUN84720101W>",
                    "message": "Model '%s' could not be loaded! Reason: [%s]"
                    % (request.modelId, str(e.message)),
                    "model_id": request.modelId,
                    "error_id": e.id,
                }
            )
            raise e

        # get concurrency
        model_mesh_config = get_config().inference_plugin.model_mesh
        if request.modelType in model_mesh_config.max_model_concurrency_per_type:
            max_concurrency = model_mesh_config.max_model_concurrency_per_type[
                request.modelType
            ]
        else:
            max_concurrency = model_mesh_config.max_model_concurrency

        return model_runtime_pb2.LoadModelResponse(
            sizeInBytes=model_size, maxConcurrency=max_concurrency
        )

    def unloadModel(self, request, context):
        """Model unloading.

        Args:
            request(model_runtime_pb2.UnloadModelRequest):
                gRPC request, gen from model-runtime.proto
            context(grpc.ServicerContext):
                Context object (contains request metadata, etc)
        Returns:
            model_runtime_pb2.UnloadModelResponse: Gen from model-runtime.proto
        """
        try:
            log.info(
                {
                    "log_code": "<RUN10000110I>",
                    "message": "Unloading model '%s'" % request.modelId,
                    "model_id": request.modelId,
                }
            )
            model_size = self.model_manager.unload_model(request.modelId)
            log.info(
                {
                    "log_code": "<RUN10000111I>",
                    "message": "Unloaded model '%s' (Reclaimed size: %s)"
                    % (request.modelId, model_size),
                    "model_id": request.modelId,
                }
            )
        except CaikitRuntimeException as e:
            log.warning(
                {
                    "log_code": "<RUN18471838W>",
                    "message": "Model '%s' could not be unloaded! Reason: [%s]"
                    % (request.modelId, str(e.message)),
                    "model_id": request.modelId,
                    "error_id": e.id,
                }
            )
            raise e
        return model_runtime_pb2.UnloadModelResponse()

    def predictModelSize(self, request, context):
        """Predict size of not-yet-loaded model

        Args:
            request(model_runtime_pb2.PredictModelSizeRequest):
                gRPC request, gen from model-runtime.proto
            context(grpc._server._Context):
                Context object (contains request metadata, etc)
        Returns:
            model_runtime_pb2.PredictModelSizeResponse: Gen from model-runtime.proto
        """
        try:
            log.info(
                {
                    "log_code": "<RUN10000120I>",
                    "message": "Predicting size of model '%s'" % request.modelId,
                    "model_id": request.modelId,
                }
            )
            predicted_size = self.model_manager.estimate_model_size(
                request.modelId, request.modelPath, request.modelType
            )
            log.info(
                {
                    "log_code": "<RUN10000123I>",
                    "message": "Predicted model '%s' size: [%s]"
                    % (request.modelId, str(predicted_size)),
                    "model_id": request.modelId,
                }
            )

        except CaikitRuntimeException as e:
            log.warning(
                {
                    "log_code": "<RUN14920102W>",
                    "message": "Model '%s' size could not be predicted! Reason: [%s]"
                    % (request.modelId, e.message),
                    "model_id": request.modelId,
                    "error_id": e.id,
                }
            )
            raise e
        return model_runtime_pb2.PredictModelSizeResponse(sizeInBytes=predicted_size)

    def modelSize(self, request, context):
        """Compute size (memory consumption) of currently-loaded model

        Args:
            request(model_runtime_pb2.ModelSizeRequest):
                gRPC request, gen from model-runtime.proto
            context(grpc._server._Context):
                Context object (contains request metadata, etc)
        Returns:
            model_runtime_pb2.ModelSizeResponse: Gen from model-runtime.proto
        """
        try:
            log.info(
                {
                    "log_code": "<RUN10000121I>",
                    "message": "Computing size of model '%s'" % request.modelId,
                    "model_id": request.modelId,
                }
            )
            model_size = self.model_manager.get_model_size(request.modelId)
            log.info(
                {
                    "log_code": "<RUN10000122I>",
                    "message": "Computed model '%s' size: [%s]"
                    % (request.modelId, str(model_size)),
                    "model_id": request.modelId,
                }
            )
        except CaikitRuntimeException as e:
            log.warning(
                {
                    "log_code": "<RUN14440122W>",
                    "message": "Failed to calculate model '%s' size! Reason: [%s]"
                    % (request.modelId, e.message),
                    "model_id": request.modelId,
                    "error_id": e.id,
                }
            )
            raise e
        return model_runtime_pb2.ModelSizeResponse(sizeInBytes=model_size)

    def runtimeStatus(self, request, context):
        """Runtime status checking.

        Args:
            request(model_runtime_pb2.RuntimeStatusRequest):
                GRPC request, gen from model-runtime.proto
            context(grpc.ServicerContext):
                Context object (contains request metadata, etc)
        Returns:
            model_runtime_pb2.RuntimeStatusResponse:
                Gen from model-runtime.proto
        """
        model_mesh_config = get_config().inference_plugin.model_mesh
        log.info(
            "<RUN25209721I>",
            "Starting Model Runtime version: %s",
            model_mesh_config.runtime_version,
        )
        return model_runtime_pb2.RuntimeStatusResponse(
            status=model_runtime_pb2.RuntimeStatusResponse.READY,
            capacityInBytes=model_mesh_config.capacity,
            maxLoadingConcurrency=model_mesh_config.max_loading_concurrency,
            modelLoadingTimeoutMs=model_mesh_config.model_loading_timeout_ms,
            defaultModelSizeInBytes=model_mesh_config.default_model_size,
            runtimeVersion=model_mesh_config.runtime_version,
            numericRuntimeVersion=model_mesh_config.numeric_runtime_version,
            limitModelConcurrency=model_mesh_config.latency_based_autoscaling_enabled,
        )
