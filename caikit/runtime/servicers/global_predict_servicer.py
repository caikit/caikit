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
from typing import Any, Dict, Iterable, Optional, Set, Union
import itertools
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtobufMessage
from grpc import RpcError, ServicerContext, StatusCode
from prometheus_client import Counter, Summary

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import ModuleBase, TaskBase
from caikit.core.data_model import DataBase, DataStream
from caikit.core.exceptions.caikit_core_exception import CaikitCoreException
from caikit.core.signature_parsing import CaikitMethodSignature
from caikit.interfaces.runtime.data_model import RuntimeServerContextType
from caikit.runtime.metrics.rpc_meter import RPCMeter
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.names import MODEL_MESH_MODEL_ID_KEY
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import clean_lib_names
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    build_proto_response,
    build_proto_stream,
    get_metadata,
    raise_caikit_runtime_exception,
    validate_data_model,
)
from caikit.runtime.work_management.abortable_context import (
    AbortableContext,
    ThreadInterrupter,
)
from caikit.runtime.work_management.rpc_aborter import RpcAborter

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

    # Input size in code points, provided by orchestrator
    INPUT_SIZE_KEY = "input-length"

    def __init__(
        self,
        inference_service: ServicePackage,
        interrupter: ThreadInterrupter = None,
    ):
        self._started_metering = False
        self._model_manager = ModelManager.get_instance()
        self._rpc_meter = None
        if get_config().runtime.metering.enabled:
            self._started_metering = True
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

        self._interrupter = interrupter
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
        request: Union[ProtobufMessage, Iterable[ProtobufMessage]],
        context: ServicerContext,
        caikit_rpc: TaskPredictRPC,
        *_,
        **__,
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
        model_id = get_metadata(context, MODEL_MESH_MODEL_ID_KEY)
        request_name = caikit_rpc.request.name

        with self._handle_predict_exceptions(model_id, request_name), alog.ContextLog(
            log.debug, "GlobalPredictServicer.Predict:%s", request_name
        ):
            # Retrieve the model from the model manager
            log.debug("<RUN52259029D>", "Retrieving model '%s'", model_id)
            model = self._model_manager.retrieve_model(model_id)
            model_class = type(model)

            # Little hackity hack: Calling _verify_model_task upfront here as well to
            # short-circuit requests where the model is _totally_ unsupported
            self._verify_model_task(model)

            # Unmarshall the request object into the required module run argument(s)
            with PREDICT_FROM_PROTO_SUMMARY.labels(
                grpc_request=request_name, model_id=model_id
            ).time():
                inference_signature = model_class.get_inference_signature(
                    input_streaming=caikit_rpc.input_streaming,
                    output_streaming=caikit_rpc.output_streaming,
                    task=caikit_rpc.task,
                )
                if not inference_signature:
                    raise CaikitRuntimeException(
                        StatusCode.INVALID_ARGUMENT,
                        f"Model class {model_class} does not support {caikit_rpc.name}",
                    )
                if caikit_rpc.input_streaming:
                    caikit_library_request = self._build_caikit_library_request_stream(
                        request, inference_signature, caikit_rpc
                    )
                else:
                    caikit_library_request = build_caikit_library_request_dict(
                        request,
                        inference_signature,
                    )
            response = self.predict_model(
                request_name,
                model_id,
                input_streaming=caikit_rpc.input_streaming,
                output_streaming=caikit_rpc.output_streaming,
                task=caikit_rpc.task,
                aborter=RpcAborter(context) if self._interrupter else None,
                context=context,
                context_arg=inference_signature.context_arg,
                **caikit_library_request,
            )

            # Marshall the response to the necessary return type
            with PREDICT_TO_PROTO_SUMMARY.labels(
                grpc_request=request_name, model_id=model_id
            ).time():
                if caikit_rpc.output_streaming:
                    response_proto = build_proto_stream(response)
                else:
                    response_proto = build_proto_response(response)
            return response_proto

    def predict_model(
        self,
        request_name: str,
        model_id: str,
        inference_func_name: str = "run",
        input_streaming: Optional[bool] = None,
        output_streaming: Optional[bool] = None,
        task: Optional[TaskBase] = None,
        aborter: Optional[RpcAborter] = None,
        context: Optional[RuntimeServerContextType] = None,  # noqa: F821
        context_arg: Optional[str] = None,
        **kwargs,
    ) -> Union[DataBase, Iterable[DataBase]]:
        """Run a prediction against the given model using the raw arguments to
        the model's run function.

        Args:
            request_name (str):
                The name of the request message to validate the model's task
            model_id (str):
                The ID of the loaded model
            inference_func_name (str):
                Explicit name of the inference function to predict (ignored if
                input_streaming and output_streaming set)
            input_streaming (Optional[bool]):
                Use the task function with input streaming
            output_streaming (Optional[bool]):
                Use the task function with output streaming
            task (Optional[TaskBase])
                The task to use for inference (if multitask model)
            aborter (Optional[RpcAborter]):
                If using abortable calls, this is the aborter to use
            **kwargs: Keyword arguments to pass to the model's run function
        Returns:
            response (Union[DataBase, Iterable[DataBase]]):
                The object (unary) or objects (output stream) produced by the
                inference request
        """

        with self._handle_predict_exceptions(model_id, request_name):
            model = self._model_manager.retrieve_model(model_id)
            self._verify_model_task(model)
            if input_streaming is not None and output_streaming is not None:
                inference_sig = model.get_inference_signature(
                    output_streaming=output_streaming,
                    input_streaming=input_streaming,
                    task=task,
                )
                inference_func_name = inference_sig.method_name
                context_arg = inference_sig.context_arg

                log.debug2(
                    "Deduced inference function name: %s and context_arg: %s",
                    inference_func_name,
                    context_arg,
                )

            # If a context arg was supplied then add the context
            if context_arg:
                kwargs[context_arg] = context

            model_run_fn = getattr(model, inference_func_name)
            # NB: we previously recorded the size of the request, and timed this module to
            # provide a rudimentary throughput metric of size / time
            # 🌶️🌶️🌶️ The `AbortableContext` will only abort if both `self._interrupter` and
            # `aborter` are set
            with alog.ContextLog(
                log.debug,
                "GlobalPredictServicer.Predict.caikit_library_run:%s",
                request_name,
            ), PREDICT_CAIKIT_LIBRARY_SUMMARY.labels(
                grpc_request=request_name, model_id=model_id
            ).time(), AbortableContext(
                aborter, self._interrupter
            ):
                response = model_run_fn(**kwargs)

            # Update Prometheus metrics
            PREDICT_RPC_COUNTER.labels(
                grpc_request=request_name, code=StatusCode.OK.name, model_id=model_id
            ).inc()
            if get_config().runtime.metering.enabled:
                self.rpc_meter.update_metrics(str(type(model)))
            return response

    def stop_metering(self):
        if self._started_metering:
            self.rpc_meter.flush_metrics()
            self.rpc_meter.end_writer_thread()
            self._started_metering = False

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
        except CaikitCoreException as e:
            raise_caikit_runtime_exception(exception=e)
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
                f"{e}",
            ) from e

        # NOTE: Specifically handling RpcError here is to pass through
        # grpc client errors, since we expect those clients to be common
        except RpcError as e:
            log_dict = {
                "log_code": "<RUN29029171W>",
                "message": repr(e),
                "model_id": model_id,
            }
            log.warning(log_dict)
            raise CaikitRuntimeException(
                e.code(),
                e.details(),
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
                StatusCode.INTERNAL,
                f"{e}",
            ) from e

    def _verify_model_task(self, model: ModuleBase):
        """Raise if the model is not supported for the task"""
        rpc_set: Set[TaskPredictRPC] = set(self._inference_service.caikit_rpcs.values())
        module_rpc: TaskPredictRPC = next(
            (rpc for rpc in rpc_set if rpc.task in model.__class__.tasks),
            None,
        )

        if not module_rpc:
            raise CaikitRuntimeException(
                status_code=StatusCode.INVALID_ARGUMENT,
                message=f"Inference for model class {type(model)} not supported by this runtime",
            )

    def _build_caikit_library_request_stream(
        self,
        request_stream: Iterable[ProtobufMessage],
        module_signature: CaikitMethodSignature,
        caikit_rpc: TaskPredictRPC,
    ) -> Dict[str, Any]:
        """Builds the kwargs dict to pass to a caikit module.
        Specifically handles the case of constructing input `DataStreams` for some parameters
        which are meant to be streamed in.

        See caikit.runtime.build_caikit_library_request_dict
        """

        def call_build_request_dict(request: ProtobufMessage) -> Dict[str, Any]:
            """This is instead of using a lambda to map each request in the stream"""
            return build_caikit_library_request_dict(request, module_signature)

        streaming_params = caikit_rpc.task.get_required_parameters(input_streaming=True)

        # We need n+1 streams because the first stream is peeked in order to read all the
        # non-streaming parameters off of the first message
        num_streams = 1 + len(streaming_params)
        all_the_streams = itertools.tee(request_stream, num_streams)

        # Read the non-streaming parameters off of the first message in the stream
        stream_num = 0
        kwargs_dict = build_caikit_library_request_dict(
            next(all_the_streams[stream_num]), module_signature
        )
        stream_num += 1

        for param in streaming_params:
            # For each "streaming" parameter, grab one of the tee'd streams and map it to return
            # a `DataStream` of that individual parameter

            def build_getter_from_request_dict(param_name: str) -> Any:
                # This builder is required to correctly closure the `param_name` of the streaming
                # parameter that we're interested in
                def get_fn(request_dict):
                    # Return this parameter out of the request dict
                    return request_dict.get(param_name)

                return get_fn

            param_stream = (
                DataStream.from_iterable(all_the_streams[stream_num])
                .map(call_build_request_dict)
                .map(build_getter_from_request_dict(param_name=param))
            )
            # Add the datastream of this one parameter into the final kwargs dict
            kwargs_dict[param] = param_stream
            stream_num += 1

        return kwargs_dict
