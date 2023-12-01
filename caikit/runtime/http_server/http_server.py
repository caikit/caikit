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
This module holds the implementation of caikit's primary HTTP server entrypoint.
The server is responsible for binding caikit workloads to a consistent REST/SSE
API based on the task definitions available at boot.
"""
# Standard
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, Iterable, Optional, Type, Union, get_args
import asyncio
import inspect
import io
import json
import os
import re
import ssl
import tempfile
import threading
import time
import traceback
import uuid

# Third Party
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from grpc import StatusCode
from sse_starlette import EventSourceResponse, ServerSentEvent
import pydantic
import uvicorn

# First Party
from py_to_proto.dataclass_to_proto import get_origin  # Imported here for 3.8 compat
import aconfig
import alog

# Local
from .pydantic_wrapper import (
    dataobject_to_pydantic,
    pydantic_from_request,
    pydantic_to_dataobject,
)
from .request_aborter import HttpRequestAborter
from .utils import convert_json_schema_to_multipart, flatten_json_schema
from caikit.config import get_config
from caikit.core.data_model import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.toolkit.sync_to_async import async_wrap_iter
from caikit.runtime.server_base import RuntimeServerBase
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import (
    CaikitRPCBase,
    ModuleClassTrainRPC,
    TaskPredictRPC,
)
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.servicers.info_servicer import InfoServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

## Globals #####################################################################

log = alog.use_channel("SERVR-HTTP")


# Mapping from GRPC codes to their corresponding HTTP codes
# pylint: disable=line-too-long
# CITE: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/refs/tags/v1.21.4-pre1/doc/statuscodes.md
GRPC_CODE_TO_HTTP = {
    StatusCode.OK: 200,
    StatusCode.INVALID_ARGUMENT: 400,
    StatusCode.FAILED_PRECONDITION: 400,
    StatusCode.OUT_OF_RANGE: 400,
    StatusCode.UNAUTHENTICATED: 401,
    StatusCode.PERMISSION_DENIED: 403,
    StatusCode.NOT_FOUND: 404,
    StatusCode.ALREADY_EXISTS: 409,
    StatusCode.ABORTED: 409,
    StatusCode.RESOURCE_EXHAUSTED: 429,
    StatusCode.CANCELLED: 499,
    StatusCode.UNKNOWN: 500,
    StatusCode.DATA_LOSS: 500,
    StatusCode.UNIMPLEMENTED: 501,
    StatusCode.UNAVAILABLE: 501,
    StatusCode.DEADLINE_EXCEEDED: 504,
}


# These keys are used to define the logical sections of the request and response
# data structures.
REQUIRED_INPUTS_KEY = "inputs"
OPTIONAL_INPUTS_KEY = "parameters"
MODEL_ID = "model_id"

# Endpoint to use for health checks
HEALTH_ENDPOINT = "/health"

# Endpoint to use for server info
RUNTIME_INFO_ENDPOINT = "/info/version"


# Stream event types enum
class StreamEventTypes(Enum):
    MESSAGE = "message"
    ERROR = "error"


# Small dataclass for consolidating TLS files
@dataclass
class _TlsFiles:
    server_cert: Optional[str] = None
    server_key: Optional[str] = None
    client_cert: Optional[str] = None


## RuntimeHTTPServer ###########################################################


class RuntimeHTTPServer(RuntimeServerBase):
    """An implementation of a FastAPI server that serves caikit runtimes"""

    ###############
    ## Interface ##
    ###############

    def __init__(self, tls_config_override: Optional[aconfig.Config] = None):
        super().__init__(get_config().runtime.http.port, tls_config_override)

        self.app = FastAPI()

        # Request validation
        @self.app.exception_handler(RequestValidationError)
        async def request_validation_exception_handler(
            _, exc: RequestValidationError
        ) -> Response:
            err_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            error_content = {
                "details": exc.errors()[0]["msg"]
                if len(exc.errors()) > 0 and "msg" in exc.errors()[0]
                else exc.errors(),
                "additional_info": exc.errors(),
                "code": err_code,
                "id": uuid.uuid4().hex,
            }
            log.error("<RUN59871106E>", error_content, exc_info=True)
            return JSONResponse(
                content=jsonable_encoder(error_content),
                status_code=err_code,
            )

        # Response validation
        @self.app.exception_handler(ResponseValidationError)
        async def validation_exception_handler(_, exc: ResponseValidationError):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
            )

        # Start metrics server
        RuntimeServerBase._start_metrics_server()

        # Placeholders for global servicers
        self.global_predict_servicer = None
        self.global_train_servicer = None
        self.info_servicer = InfoServicer()

        # Set up inference if enabled
        if self.enable_inference:
            log.info("<RUN77183426I>", "Enabling HTTP inference service")
            self.global_predict_servicer = GlobalPredictServicer(self.inference_service)
            self._bind_routes(self.inference_service)

        # Set up training if enabled
        if self.enable_training:
            log.info("<RUN77183427I>", "Enabling HTTP training service")
            self.global_train_servicer = GlobalTrainServicer(self.training_service)
            self._bind_routes(self.training_service)

        # Add the health endpoint
        self.app.get(HEALTH_ENDPOINT, response_class=PlainTextResponse)(
            self._health_check
        )

        # Add runtime info endpoint
        self.app.get(RUNTIME_INFO_ENDPOINT, response_class=JSONResponse)(
            self.info_servicer.get_version_dict
        )

        # Parse TLS configuration
        # If any of the TLS values are not files, we assume that they're inline
        # content. The python SslContext only takes files to load, so we use a
        # temporary directory just long enough to load the config files.
        with self._tls_files() as tls_files:
            tls_kwargs = {}
            if tls_files.server_key and tls_files.server_cert:
                log.info("<RUN10001905I>", "Running with TLS")
                tls_kwargs["ssl_keyfile"] = tls_files.server_key
                tls_kwargs["ssl_certfile"] = tls_files.server_cert
                if tls_files.client_cert:
                    log.info("<RUN10001809I>", "Running with mutual TLS")
                    tls_kwargs["ssl_ca_certs"] = tls_files.client_cert
                    tls_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            else:
                log.info("<RUN10539515I>", "Running INSECURE")

            # Start the server with a timeout_graceful_shutdown
            # if not set in config, this is None and unvicorn accepts None or number of seconds
            unvicorn_timeout_graceful_shutdown = (
                get_config().runtime.http.server_shutdown_grace_period_seconds
            )
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level=None,
                log_config=None,
                timeout_graceful_shutdown=unvicorn_timeout_graceful_shutdown,
                **tls_kwargs,
            )
            # Make sure the config loads TLS files here so they can safely be
            # deleted if they're ephemeral
            config.load()

        # Start the server with the loaded config
        self.server = uvicorn.Server(config=config)

        # Patch the exit handler to call this server's stop
        original_handler = self.server.handle_exit

        def shutdown_wrapper(*args, **kwargs):
            original_handler(*args, **kwargs)
            self.stop()

        self.server.handle_exit = shutdown_wrapper

        # Placeholder for thread when running without blocking
        self._uvicorn_server_thread = None

    def __del__(self):
        config = get_config()
        if config and config.runtime.metering.enabled and self.global_predict_servicer:
            self.global_predict_servicer.stop_metering()

    def start(self, blocking: bool = True):
        """Boot the http server. Can be non-blocking, or block until shutdown

        Args:
            blocking (boolean): Whether to block until shutdown
        """
        if blocking:
            self.server.run()
        else:
            self.run_in_thread()

    def stop(self):
        """Stop the server, with an optional grace period.

        Args:
            grace_period_seconds (Union[float, int]): Grace period for service shutdown.
                Defaults to application config
        """
        self.server.should_exit = True
        if (
            self._uvicorn_server_thread is not None
            and self._uvicorn_server_thread.is_alive()
        ):
            self._uvicorn_server_thread.join()

        # Ensure we flush out any remaining billing metrics and stop metering
        if self.config.runtime.metering.enabled and self.global_predict_servicer:
            self.global_predict_servicer.stop_metering()

        # Shut down the model manager's model polling if enabled
        self._shut_down_model_manager()

    def run_in_thread(self):
        self._uvicorn_server_thread = threading.Thread(target=self.server.run)
        self._uvicorn_server_thread.start()
        while not self.server.started:
            time.sleep(1e-3)
        log.info("HTTP Server is running in thread")

    ##########
    ## Impl ##
    ##########
    def _bind_routes(self, service: ServicePackage):
        """Bind all rpcs as routes to the given app"""
        for rpc in service.caikit_rpcs.values():
            rpc_info = rpc.create_rpc_json("")
            if isinstance(rpc, TaskPredictRPC):
                if hasattr(rpc, "input_streaming") and rpc.input_streaming:
                    # Skipping the binding of this route since we don't have support
                    log.info(
                        "No support for input streaming on REST Server yet!"
                        "Skipping this rpc %s with input type %s",
                        rpc_info["name"],
                        rpc_info["input_type"],
                    )
                    continue
                if hasattr(rpc, "output_streaming") and rpc.output_streaming:
                    self._add_unary_input_stream_output_handler(rpc)
                else:
                    self._add_unary_input_unary_output_handler(rpc)
            elif isinstance(rpc, ModuleClassTrainRPC):
                self._train_add_unary_input_unary_output_handler(rpc)

    def _get_model_id(self, request: Type[pydantic.BaseModel]) -> Dict[str, Any]:
        """Get the model id from the payload"""
        request_kwargs = dict(request)
        model_id = request_kwargs.get(MODEL_ID, None)
        if model_id is None:
            raise CaikitRuntimeException(
                status_code=StatusCode.INVALID_ARGUMENT,
                message="Please provide model_id in payload",
            )
        return model_id

    def _get_request_params(
        self, rpc: CaikitRPCBase, request: Type[pydantic.BaseModel]
    ) -> Dict[str, Any]:
        """Get the request params based on the RPC's req params, also
        convert to DM objects"""
        request_kwargs = dict(request)
        input_name = None
        required_params = None
        if isinstance(rpc, TaskPredictRPC):
            required_params = rpc.task.get_required_parameters(rpc.input_streaming)
        # handle required param input name
        if required_params and len(required_params) == 1:
            input_name = list(required_params.keys())[0]
        # flatten inputs and params into a dict
        # would have been useful to call dataobject.to_dict()
        # but unfortunately we now have converted pydantic objects
        combined_dict = {}
        for field, value in request_kwargs.items():
            if field == MODEL_ID:
                continue
            if value:
                if field == REQUIRED_INPUTS_KEY and input_name:
                    combined_dict[input_name] = value
                else:
                    combined_dict.update(**dict(request_kwargs[field]))
        # remove non-none items
        request_params = {k: v for k, v in combined_dict.items() if v is not None}
        # convert pydantic objects to our DM objects
        for param_name, param_value in request_params.items():
            if issubclass(type(param_value), pydantic.BaseModel):
                request_params[param_name] = pydantic_to_dataobject(param_value)
        return request_params

    def _train_add_unary_input_unary_output_handler(self, rpc: CaikitRPCBase):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = dataobject_to_pydantic(
            DataBase.get_class_for_name(rpc.request.name)
        )
        response_data_object = self._get_response_dataobject(rpc)
        pydantic_response = dataobject_to_pydantic(response_data_object)

        @self.app.post(
            self._get_route(rpc),
            responses=self._get_response_openapi(
                response_data_object, pydantic_response
            ),
            openapi_extra=self._get_request_openapi(pydantic_request),
            response_class=Response,
        )
        # pylint: disable=unused-argument
        async def _handler(context: Request) -> Response:
            log.debug("In unary handler for %s", rpc.name)
            loop = asyncio.get_running_loop()

            # build request DM object
            request = await pydantic_from_request(pydantic_request, context)
            http_request_dm_object = pydantic_to_dataobject(request)

            try:
                call = partial(
                    self.global_train_servicer.run_training_job,
                    request=http_request_dm_object.to_proto(),
                    module=rpc.clz,
                    training_output_dir=None,  # pass None so that GTS picks up the config one # TODO: double-check? # noqa: E501
                    # context=context,
                    wait=True,
                )
                result = await loop.run_in_executor(None, call)
                if response_data_object.supports_file_operations:
                    return self._format_file_response(result)

                return Response(content=result.to_json(), media_type="application/json")
            except HTTPException as err:
                raise err
            except CaikitRuntimeException as err:
                error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                error_content = {
                    "details": err.message,
                    "code": error_code,
                    "id": err.id,
                }
                log.error("<RUN87691106E>", error_content, exc_info=True)
            except Exception as err:  # pylint: disable=broad-exception-caught
                error_code = 500
                error_content = {
                    "details": f"Unhandled exception: {str(err)}",
                    "code": error_code,
                    "id": uuid.uuid4().hex,
                }
                log.error("<RUN51231106E>", error_content, exc_info=True)
            return Response(
                content=json.dumps(error_content), status_code=error_code
            )  # pylint: disable=used-before-assignment

    def _add_unary_input_unary_output_handler(self, rpc: TaskPredictRPC):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = dataobject_to_pydantic(self._get_request_dataobject(rpc))
        response_data_object = self._get_response_dataobject(rpc)
        pydantic_response = dataobject_to_pydantic(response_data_object)

        @self.app.post(
            self._get_route(rpc),
            responses=self._get_response_openapi(
                response_data_object, pydantic_response
            ),
            openapi_extra=self._get_request_openapi(pydantic_request),
            response_class=Response,
        )
        # pylint: disable=unused-argument
        async def _handler(
            context: Request,
        ) -> Response:
            request = await pydantic_from_request(pydantic_request, context)
            request_params = self._get_request_params(rpc, request)

            try:
                model_id = self._get_model_id(request)
                log.debug4(
                    "Sending request %s to model id %s", request_params, model_id
                )

                log.debug("In unary handler for %s for model %s", rpc.name, model_id)
                loop = asyncio.get_running_loop()

                request_params = self._get_request_params(rpc, request)

                log.debug4(
                    "Sending request %s to model id %s", request_params, model_id
                )
                model = self.global_predict_servicer._model_manager.retrieve_model(
                    model_id
                )

                aborter_context = (
                    HttpRequestAborter(context)
                    if get_config().runtime.use_abortable_threads
                    else nullcontext()
                )

                with aborter_context as aborter:
                    # TODO: use `async_wrap_*`?
                    call = partial(
                        self.global_predict_servicer.predict_model,
                        model_id=model_id,
                        request_name=rpc.request.name,
                        inference_func_name=model.get_inference_signature(
                            output_streaming=False, input_streaming=False, task=rpc.task
                        ).method_name,
                        aborter=aborter,
                        **request_params,
                    )
                    result = await loop.run_in_executor(None, call)
                    log.debug4("Response from model %s is %s", model_id, result)

                if response_data_object.supports_file_operations:
                    return self._format_file_response(result)

                return Response(content=result.to_json(), media_type="application/json")

            except HTTPException as err:
                raise err
            except CaikitRuntimeException as err:
                error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                error_content = {
                    "details": err.message,
                    "code": error_code,
                    "id": err.id,
                }
                log.error("<RUN53211098E>", error_content, exc_info=True)
            except Exception as err:  # pylint: disable=broad-exception-caught
                error_code = 500
                error_content = {
                    "details": f"Unhandled exception: {str(err)}",
                    "code": error_code,
                    "id": uuid.uuid4().hex,
                }
                log.error("<RUN98751106E>", error_content, exc_info=True)
            return Response(
                content=json.dumps(error_content), status_code=error_code
            )  # pylint: disable=used-before-assignment

    def _add_unary_input_stream_output_handler(self, rpc: CaikitRPCBase):
        pydantic_request = dataobject_to_pydantic(self._get_request_dataobject(rpc))
        pydantic_response = dataobject_to_pydantic(self._get_response_dataobject(rpc))

        # pylint: disable=unused-argument
        @self.app.post(
            self._get_route(rpc),
            response_model=pydantic_response,
            openapi_extra=self._get_request_openapi(pydantic_request),
        )
        async def _handler(context: Request) -> EventSourceResponse:
            log.debug("In streaming handler for %s", rpc.name)

            request = await pydantic_from_request(pydantic_request, context)
            request_params = self._get_request_params(rpc, request)

            async def _generator() -> pydantic_response:
                try:
                    model_id = self._get_model_id(request)
                    log.debug4(
                        "Sending request %s to model id %s", request_params, model_id
                    )
                    model = self.global_predict_servicer._model_manager.retrieve_model(
                        model_id
                    )

                    aborter_context = (
                        HttpRequestAborter(context)
                        if get_config().runtime.use_abortable_threads
                        else nullcontext()
                    )

                    with aborter_context as aborter:
                        log.debug("In stream generator for %s", rpc.name)
                        async for result in async_wrap_iter(
                            self.global_predict_servicer.predict_model(
                                model_id=model_id,
                                request_name=rpc.request.name,
                                inference_func_name=model.get_inference_signature(
                                    output_streaming=True, input_streaming=False
                                ).method_name,
                                aborter=aborter,
                                **request_params,
                            )
                        ):
                            yield ServerSentEvent(
                                data=result.to_json(),
                                event=StreamEventTypes.MESSAGE.value,
                            )

                    return
                except HTTPException as err:
                    raise err
                except (TypeError, ValueError) as err:
                    log_dict = {
                        "log_code": "<RUN76624264W>",
                        "message": repr(err),
                        "stack_trace": traceback.format_exc(),
                    }
                    log.warning(log_dict)
                    error_code = 400
                    error_content = {
                        "details": repr(err),
                        "code": error_code,
                    }
                except CaikitRuntimeException as err:
                    error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                    error_content = {
                        "details": err.message,
                        "code": error_code,
                        "id": err.id,
                    }
                    log.error("<RUN53234506E>", error_content, exc_info=True)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    error_code = 500
                    error_content = {
                        "details": f"Unhandled exception: {str(err)}",
                        "code": error_code,
                        "id": uuid.uuid4().hex,
                    }
                    log.error("<RUN51891206E>", error_content, exc_info=True)

                # If an error occurs, yield an error response and terminate
                yield ServerSentEvent(
                    data=json.dumps(error_content), event=StreamEventTypes.ERROR.value
                )

            return EventSourceResponse(_generator())

    def _get_route(self, rpc: CaikitRPCBase) -> str:
        """Get the REST route for this rpc"""
        if rpc.name.endswith("Predict"):
            task_name = re.sub(
                r"(?<!^)(?=[A-Z])",
                "-",
                re.sub("Task$", "", re.sub("Predict$", "", rpc.name)),
            ).lower()
            route = "/".join([self.config.runtime.http.route_prefix, "task", task_name])
            if route[0] != "/":
                route = "/" + route
            return route
        if rpc.name.endswith("Train"):
            route = "/".join([self.config.runtime.http.route_prefix, rpc.name])
            if route[0] != "/":
                route = "/" + route
            return route
        raise NotImplementedError("No support for train rpcs yet!")

    def _get_request_dataobject(self, rpc: CaikitRPCBase) -> Type[DataBase]:
        """Get the dataobject request for the given rpc"""
        is_inference_rpc = hasattr(rpc, "task")
        if is_inference_rpc:
            required_params = rpc.task.get_required_parameters(rpc.input_streaming)
        else:  # train
            required_params = {
                entry[1]: entry[0]
                for entry in rpc.request.triples
                if entry[1] not in rpc.request.default_map
            }
        optional_params = {
            entry[1]: entry[0]
            for entry in rpc.request.triples
            if entry[1] not in required_params
        }

        inputs_type = None
        if is_inference_rpc:
            pkg_name = f"caikit.http.{rpc.task.__name__}"
        else:
            pkg_name = f"caikit.http.{rpc.name}"

        # Create a bundled sub-message for required parameters for multiple params
        if len(required_params) > 1:
            log.debug3("Using structured inputs type for %s", pkg_name)
            inputs_type = make_dataobject(
                name=f"{rpc.request.name}Inputs",
                annotations=required_params,
                package=pkg_name,
            )
        elif required_params:
            inputs_type = list(required_params.values())[0]
            log.debug3(
                "Using single inputs type for task %s: %s", pkg_name, inputs_type
            )

        # Always create a bundled sub-message for optional parameters
        parameters_type = None
        if optional_params:
            parameters_type = make_dataobject(
                name=f"{rpc.request.name}Parameters",
                annotations=optional_params,
                package=pkg_name,
            )

        # Create the top-level request message
        request_annotations = {}
        if inputs_type:
            request_annotations[REQUIRED_INPUTS_KEY] = inputs_type
        if parameters_type:
            request_annotations[OPTIONAL_INPUTS_KEY] = parameters_type
        request_annotations[MODEL_ID] = str
        request_message = make_dataobject(
            name=f"{rpc.request.name}HttpRequest",
            annotations=request_annotations,
            package=pkg_name,
        )

        return request_message

    @staticmethod
    def _get_response_dataobject(rpc: CaikitRPCBase) -> Type[DataBase]:
        """Get the dataobject response for the given rpc"""
        origin = get_origin(rpc.return_type)
        args = get_args(rpc.return_type)
        if isinstance(origin, type) and issubclass(origin, Iterable):
            assert args and len(args) == 1
            dm_obj = args[0]
        else:
            dm_obj = rpc.return_type
        assert isinstance(dm_obj, type) and issubclass(dm_obj, DataBase)
        return dm_obj

    @staticmethod
    def _format_file_response(dm_class: Type[DataBase]) -> Response:
        """Convert a dm_class into a fastapi file Response"""
        file_obj = io.BytesIO()
        file_info = dm_class.to_file(file_obj)

        file_type = "application/octet-stream"
        content_disposition = "attachment"
        if file_info:
            file_type = file_info.type if file_info.type else file_type
            content_disposition = f'attachment; filename="{file_info.filename}"'

        return Response(
            content=file_obj.getvalue(),
            headers={"Content-Disposition": content_disposition},
            media_type=file_type,
        )

    @staticmethod
    def _get_request_openapi(
        pydantic_model: Union[pydantic.BaseModel, Type, Type[pydantic.BaseModel]]
    ):
        """Helper to generate the openapi schema for a given request"""

        # Get the json schema from the pydantic model or TypeAdapter
        if inspect.isclass(pydantic_model) and issubclass(
            pydantic_model, pydantic.BaseModel
        ):
            raw_schema = pydantic_model.model_json_schema()
        else:
            raw_schema = pydantic.TypeAdapter(pydantic_model).json_schema()

        parsed_schema = flatten_json_schema(raw_schema)
        multipart_schema = convert_json_schema_to_multipart(parsed_schema)

        return {
            "requestBody": {
                "content": {
                    "multipart/form-data": {"schema": multipart_schema},
                    "application/json": {"schema": parsed_schema},
                },
                "required": True,
            }
        }

    @staticmethod
    def _get_response_openapi(
        dm_class: Type[DataBase], pydantic_model: Union[Type, Type[pydantic.BaseModel]]
    ):
        """Helper to generate the openapi schema for a given response"""

        if dm_class.supports_file_operations:
            response_schema = {
                "application/octet-stream": {"type": "string", "format": "binary"}
            }
        else:
            # Get the json schema from the pydantic model or TypeAdapter
            if inspect.isclass(pydantic_model) and issubclass(
                pydantic_model, pydantic.BaseModel
            ):
                json_schema = pydantic_model.model_json_schema()
            else:
                json_schema = pydantic.TypeAdapter(pydantic_model).json_schema()

            response_schema = {"application/json": flatten_json_schema(json_schema)}

        output = {200: {"content": response_schema}}
        return output

    @staticmethod
    def _health_check() -> str:
        log.debug4("Server healthy")
        return "OK"

    @contextmanager
    def _tls_files(self) -> _TlsFiles:
        """This contextmanager ensures that the tls config values are files on
        disk since SslContext requires files

        Returns:
            tls_files (_TlsFiles): The set of configured TLS files
        """
        tls_cfg = self.tls_config or {}
        tls_opts = {
            "server_key": tls_cfg.get("server", {}).get("key", ""),
            "server_cert": tls_cfg.get("server", {}).get("cert", ""),
            "client_cert": tls_cfg.get("client", {}).get("cert", ""),
        }
        non_file_opts = {
            key: val for key, val in tls_opts.items() if val and not os.path.isfile(val)
        }
        if not non_file_opts:
            log.debug3("No TLS inline files configured")
            yield _TlsFiles(**tls_opts)
            return
        # If any of the values are set and are not pointing to files, save them
        # to a temporary directory
        try:
            with tempfile.TemporaryDirectory() as tls_dir:
                updated_files = {
                    key: val
                    for key, val in tls_opts.items()
                    if key not in non_file_opts and val
                }
                for fname, content in non_file_opts.items():
                    temp_fname = os.path.join(tls_dir, fname)
                    updated_files[fname] = temp_fname
                    with open(temp_fname, "w", encoding="utf-8") as handle:
                        handle.write(content)
                yield _TlsFiles(**updated_files)
        except OSError as err:
            log.error(
                "<RUN80977064E>",
                (
                    "Cannot create temporary TLS files."
                    "Either pass config as file paths or run with write permissions."
                ),
                exc_info=True,
            )
            raise ValueError() from err


def main(blocking: bool = True):
    server = RuntimeHTTPServer()
    server._intercept_interrupt_signal()
    server.start(blocking)


if __name__ == "__main__":
    main()
