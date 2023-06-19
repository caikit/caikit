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
from contextlib import contextmanager
from importlib.metadata import version
from typing import Iterable, Optional, Set, Union
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtobufMessage
from grpc import ServicerContext, StatusCode
from prometheus_client import Counter, Summary

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import ModuleBase
from caikit.core.data_model import DataBase
from caikit.runtime.metrics.rpc_meter import RPCMeter
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import clean_lib_names
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    build_proto_response,
    build_proto_stream,
    get_metadata,
    validate_data_model,
)
from caikit.runtime.work_management.abortable_action import AbortableAction
from caikit.runtime.work_management.call_aborter import CallAborter

PREDICT_RPC_COUNTER = Counter(
    "predict_rpc_count",
    "Count of global predict-managed RPC calls",
    ["grpc_request", "code", "model_id"],
)
PREDICT_FROM_PROTO_SUMMARY = Summary(
    "predict_from_proto_duration_seconds",
    "Histogram of predict request unmarshalling duration (in seconds)",
    ["grpc_request", "model_id"],
)
PREDICT_CAIKIT_LIBRARY_SUMMARY = Summary(
    "predict_caikit_library_duration_seconds",
    "Histogram of predict Caikit Library run duration (in seconds)",
    ["grpc_request", "model_id"],
)
PREDICT_TO_PROTO_SUMMARY = Summary(
    "predict_to_proto_duration_seconds",
    "Histogram of predict response marshalling duration (in seconds)",
    ["grpc_request", "model_id"],
)

log = alog.use_channel("GP-SERVICR-I")

# Protobuf non primitives
# Ref: https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor
NON_PRIMITIVE_TYPES = [FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.TYPE_ENUM]


class GlobalPredictServicer:
    """This class contains RPC calls affiliated with the Caikit Runtime that are
    not a part of the Model Runtime proto definition.  They will be serviced by
    mocking the particular RPC (based on the message type), leveraging the
    CaikitRuntimeServicerMock to return the appropriate mock response for a
    given request
    """

    # Invocation metadata key for the model ID, provided by Model Mesh
    MODEL_MESH_MODEL_ID_KEY = "mm-model-id"

    # Input size in code points, provided by orchestrator
    INPUT_SIZE_KEY = "input-length"

    def __init__(
        self,
        inference_service: ServicePackage,
        use_abortable_threads: bool = get_config().runtime.use_abortable_threads,
    ):
        self._model_manager = ModelManager.get_instance()
        if get_config().runtime.metering.enabled:
            self.rpc_meter = RPCMeter()
            log.info(
                "<RUN76773775I>",
                "Metering is enabled, to disable set `metering.enabled` in config to false",
            )
        else:
            log.info(
                "<RUN76773776I>",
                "Metering is disabled, to enable set `metering.enabled` in config to true",
            )

        self.use_abortable_threads = use_abortable_threads
        self._inference_service = inference_service
        # Validate that the Caikit Library CDM is compatible with our service descriptor
        validate_data_model(self._inference_service.descriptor)
        log.info("<RUN76773778I>", "Validated Caikit Library CDM successfully")

        # Duplicate code in global_train_servicer
        # pylint: disable=duplicate-code
        library = clean_lib_names(get_config().runtime.library)[0]
        try:
            lib_version = version(library)
        except Exception:  # pylint: disable=broad-exception-caught
            lib_version = "unknown"

        log.info(
            "<RUN76884779I>",
            "Constructed inference service for library: %s, version: %s",
            library,
            lib_version,
        )
        super()

    def Predict(
        self,
        request: ProtobufMessage,
        context: ServicerContext,
    ) -> Union[ProtobufMessage, Iterable[ProtobufMessage]]:
        """Global predict RPC -- Mocks the invocation of a Caikit Library module.run()
        method for a loaded Caikit Library model

        Args:
            request (ProtobufMessage):
                A deserialized RPC request message
            context (ServicerContext):
                Context object (contains request metadata, etc)

        Returns:
            response (Union[ProtobufMessage, Iterable[ProtobufMessage]]):
                A Caikit Library data model response object
        """
        # Make sure the request has a model before doing anything
        request_name = request.DESCRIPTOR.name
        model_id = get_metadata(context, self.MODEL_MESH_MODEL_ID_KEY)
        with self._handle_predict_exceptions(model_id, request_name):
            with alog.ContextLog(
                log.debug, "GlobalPredictServicer.Predict:%s", request_name
            ):
                # Retrieve the model from the model manager
                log.debug("<RUN52259029D>", "Retrieving model '%s'", model_id)
                model = self._model_manager.retrieve_model(model_id)
                model_class = type(model)
                # Unmarshall the request object into the required module run argument(s)
                with PREDICT_FROM_PROTO_SUMMARY.labels(
                    grpc_request=request_name, model_id=model_id
                ).time():
                    caikit_library_request = build_caikit_library_request_dict(
                        request,
                        model_class.RUN_SIGNATURE,
                    )
                response = self.predict_model(
                    request_name,
                    model_id,
                    aborter=CallAborter(context)
                    if self.use_abortable_threads
                    else None,
                    **caikit_library_request,
                )

                # Marshall the response to the necessary return type
                with PREDICT_TO_PROTO_SUMMARY.labels(
                    grpc_request=request_name, model_id=model_id
                ).time():
                    if model.TASK_CLASS.is_output_streaming_task():
                        response_proto = build_proto_stream(response)
                    else:
                        response_proto = build_proto_response(response)
                return response_proto

    def predict_model(
        self,
        request_name: str,
        model_id: str,
        aborter: Optional[CallAborter] = None,
        **kwargs,
    ) -> Union[DataBase, Iterable[DataBase]]:
        """Run a prediction against the given model using the raw arguments to
        the model's run function.

        Args:
            request_name (str):
                The name of the request message to validate the model's task
            model_id (str):
                The ID of the loaded model
            aborter (Optional[CallAborter]):
                If using abortable calls, this is the aborter to use
            **kwargs:
                Keyword arguments to pass to the model's run function

        Returns:
            response (Union[DataBase, Iterable[DataBase]]):
                The object (unary) or objects (output stream) produced by the
                inference request
        """

        with self._handle_predict_exceptions(model_id, request_name):
            model = self._model_manager.retrieve_model(model_id)
            self._verify_model_task(model, request_name)

            # NB: we previously recorded the size of the request, and timed this module to
            # provide a rudimentary throughput metric of size / time
            with alog.ContextLog(
                log.debug,
                "GlobalPredictServicer.Predict.caikit_library_run:%s",
                request_name,
            ):
                with PREDICT_CAIKIT_LIBRARY_SUMMARY.labels(
                    grpc_request=request_name, model_id=model_id
                ).time():
                    if aborter is not None:
                        work = AbortableAction(aborter, model.run, **kwargs)
                        response = work.do()
                    else:
                        response = model.run(**kwargs)

            # Update Prometheus metrics
            PREDICT_RPC_COUNTER.labels(
                grpc_request=request_name, code=StatusCode.OK.name, model_id=model_id
            ).inc()
            if get_config().runtime.metering.enabled:
                self.rpc_meter.update_metrics(str(type(model)))
            return response

    ## Implementation Details ##################################################

    @contextmanager
    def _handle_predict_exceptions(self, model_id: str, request_name: str):
        try:
            yield

        except CaikitRuntimeException as e:
            log_dict = {
                "log_code": "<RUN50530380W>",
                "message": e.message,
                "model_id": model_id,
                "error_id": e.id,
            }
            log.warning({**log_dict, **e.metadata})
            PREDICT_RPC_COUNTER.labels(
                grpc_request=request_name, code=e.status_code.name, model_id=model_id
            ).inc()
            raise e

        # Duplicate code in global_train_servicer
        # pylint: disable=duplicate-code
        except (TypeError, ValueError) as e:
            log_dict = {
                "log_code": "<RUN490439039W>",
                "message": repr(e),
                "model_id": model_id,
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            PREDICT_RPC_COUNTER.labels(
                grpc_request=request_name,
                code=StatusCode.INVALID_ARGUMENT.name,
                model_id=model_id,
            ).inc()
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT,
                f"Exception raised during inference. This may be a problem with your input: {e}",
            ) from e

        except Exception as e:
            log_dict = {
                "log_code": "<RUN49049070W>",
                "message": repr(e),
                "model_id": model_id,
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            PREDICT_RPC_COUNTER.labels(
                grpc_request=request_name,
                code=StatusCode.INTERNAL.name,
                model_id=model_id,
            ).inc()
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "Unhandled exception during prediction"
            ) from e

    def _verify_model_task(self, model: ModuleBase, request_name: str):
        """Raise if the model is not supported for the task"""
        rpc_set: Set[TaskPredictRPC] = self._inference_service.caikit_rpcs
        module_rpc: TaskPredictRPC = next(
            (rpc for rpc in rpc_set if model.TASK_CLASS == rpc.task),
            None,
        )

        if not module_rpc:
            raise CaikitRuntimeException(
                status_code=StatusCode.INVALID_ARGUMENT,
                message=f"Inference for model class {type(model)} not supported by this runtime",
            )

        request_rpc: TaskPredictRPC = next(
            (rpc for rpc in rpc_set if rpc.request.name == request_name), None
        )
        if request_rpc != module_rpc:
            raise CaikitRuntimeException(
                status_code=StatusCode.INVALID_ARGUMENT,
                message=f"Wrong inference RPC invoked for model class {type(model)}. "
                f"Use {request_rpc.name} instead of {request_name}",
            )
