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
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Type, Union, get_args
import asyncio
import enum
import json
import re
import ssl
import threading
import time

# Third Party
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from grpc import StatusCode
from sse_starlette import EventSourceResponse, ServerSentEvent
import numpy as np
import pydantic
import uvicorn

# First Party
from py_to_proto.dataclass_to_proto import (  # Imported here for 3.8 compat
    Annotated,
    get_origin,
)
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.core.data_model import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.toolkit.sync_to_async import async_wrap_iter
from caikit.runtime.server_base import RuntimeServerBase
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import CaikitRPCBase
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit.core.toolkit.logging

## Globals #####################################################################

log = alog.use_channel("HTTP")

# Registry of DM -> Pydantic model mapping to avoid errors when reusing messages
# across endpoints
PYDANTIC_REGISTRY = {}


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

# Endpoint to use for health checks
HEALTH_ENDPOINT = "/health"

## RuntimeHTTPServer ###########################################################


class RuntimeHTTPServer(RuntimeServerBase):
    """An implementation of a FastAPI server that serves caikit runtimes"""

    ###############
    ## Interface ##
    ###############

    def __init__(self, tls_config_override: Optional[aconfig.Config] = None):
        super().__init__(get_config().runtime.http.port, tls_config_override)

        self.app = FastAPI()

        # Set up the central predict servicer
        inference_service = ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE,
        )
        self.global_predict_servicer = GlobalPredictServicer(inference_service)

        # Set up the central train servicer
        # TODO: uncomment later on
        # train_service = ServicePackageFactory().get_service_package(
        #     ServicePackageFactory.ServiceType.TRAINING,
        # )
        # self.global_train_servicer = GlobalTrainServicer(train_service)

        self.package_name = inference_service.descriptor.full_name.rsplit(".", 1)[0]

        # Add the health endpoint
        self.app.get(HEALTH_ENDPOINT, response_class=PlainTextResponse)(
            self._health_check
        )

        # Bind all routes to the server
        self._bind_routes(inference_service)
        # self._bind_routes(train_service)

        # Parse TLS configuration
        tls_kwargs = {}
        if (
            self.tls_config
            and self.tls_config.server.key
            and self.tls_config.server.cert
        ):
            log.info("<RUN10001905I>", "Running with TLS")
            tls_kwargs["ssl_keyfile"] = self.tls_config.server.key
            tls_kwargs["ssl_certfile"] = self.tls_config.server.cert
            if self.tls_config.client.cert:
                log.info("<RUN10001809I>", "Running with mutual TLS")
                tls_kwargs["ssl_ca_certs"] = self.tls_config.client.cert
                tls_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

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
        if get_config().runtime.metering.enabled:
            self.global_predict_servicer.stop_metering()

    def start(self, blocking: bool = True):
        """Boot the gRPC server. Can be non-blocking, or block until shutdown

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
        log.info("Shutting down HTTP Server")
        self.server.should_exit = True
        if (
            self._uvicorn_server_thread is not None
            and self._uvicorn_server_thread.is_alive()
        ):
            self._uvicorn_server_thread.join()

        # Ensure we flush out any remaining billing metrics and stop metering
        if self.config.runtime.metering.enabled:
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
            if rpc_info["client_streaming"]:
                # Skipping the binding of this route since we don't have support
                log.info(
                    "No support for input streaming on REST Server yet! Skipping this rpc %s with input type %s",
                    rpc_info["name"],
                    rpc_info["input_type"],
                )
                continue
            if rpc_info["server_streaming"]:
                self._add_unary_input_stream_output_handler(rpc)
            else:
                self._add_unary_input_unary_output_handler(rpc)

    def _get_request_params(
        self, rpc: CaikitRPCBase, request: Type[pydantic.BaseModel]
    ) -> Dict[str, Any]:
        """get the request params based on the RPC's req params"""
        request_kwargs = dict(request)
        required_params = rpc.task.get_required_parameters(rpc.input_streaming)
        input_name = None
        # handle required param input name
        if len(required_params) == 1:
            input_name = list(required_params.keys())[0]
        # flatten inputs and params into a dict
        # would have been useful to call dataobject.to_dict()
        # but unfortunately we now have converted pydantic objects
        combined_dict = {}
        for field, value in request_kwargs.items():
            if value:
                if field == REQUIRED_INPUTS_KEY and input_name:
                    combined_dict[input_name] = value
                else:
                    combined_dict.update(**dict(request_kwargs[field]))
        # remove non-none items
        request_params = {k: v for k, v in combined_dict.items() if v is not None}
        return request_params

    def _add_unary_input_unary_output_handler(self, rpc: CaikitRPCBase):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = self._dataobject_to_pydantic(
            self._get_request_dataobject(rpc, False)
        )
        pydantic_response = self._dataobject_to_pydantic(
            self._get_response_dataobject(rpc)
        )

        @self.app.post(self._get_route(rpc), response_model=pydantic_response)
        # pylint: disable=unused-argument
        async def _handler(
            model_id: str, request: pydantic_request, context: Request
        ) -> Union[pydantic_response, Response]:
            log.debug("In unary handler for %s for model %s", rpc.name, model_id)
            loop = asyncio.get_running_loop()

            request_params = self._get_request_params(rpc, request)

            log.debug4("Sending request %s to model id %s", request_params, model_id)
            try:
                model = self.global_predict_servicer._model_manager.retrieve_model(
                    model_id
                )

                # TODO: use `async_wrap_*`?
                call = partial(
                    self.global_predict_servicer.predict_model,
                    model_id=model_id,
                    request_name=rpc.request.name,
                    inference_func_name=model.get_inference_signature(
                        output_streaming=False, input_streaming=False
                    ).method_name,
                    **request_params,
                )
                result = await loop.run_in_executor(None, call)
                log.debug4("Response from model %s is %s", model_id, result)
                return result
            except CaikitRuntimeException as err:
                error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                error_content = {
                    "details": err.message,
                    "code": error_code,
                    "id": err.id,
                }
            except Exception as err:  # pylint: disable=broad-exception-caught
                error_code = 500
                error_content = {
                    "details": f"Unhandled exception: {str(err)}",
                    "code": error_code,
                    "id": None,
                }
                log.error("<RUN51881106E>", err, exc_info=True)
            return Response(content=json.dumps(error_content), status_code=error_code)

    def _add_unary_input_stream_output_handler(self, rpc: CaikitRPCBase):
        pydantic_request = self._dataobject_to_pydantic(
            self._get_request_dataobject(rpc, False)
        )
        pydantic_response = self._dataobject_to_pydantic(
            self._get_response_dataobject(rpc)
        )
        # pylint: disable=unused-argument
        @self.app.post(self._get_route(rpc), response_model=pydantic_response)
        async def _handler(
            model_id: str, request: pydantic_request, context: Request
        ) -> EventSourceResponse:
            log.debug("In streaming handler for %s", rpc.name)

            request_params = self._get_request_params(rpc, request)
            log.debug4("Sending request %s to model id %s", request_params, model_id)

            async def _generator() -> pydantic_response:
                try:
                    model = self.global_predict_servicer._model_manager.retrieve_model(
                        model_id
                    )

                    log.debug("In stream generator for %s", rpc.name)
                    async for result in async_wrap_iter(
                        self.global_predict_servicer.predict_model(
                            model_id=model_id,
                            request_name=rpc.request.name,
                            inference_func_name=model.get_inference_signature(
                                output_streaming=True, input_streaming=False
                            ).method_name,
                            **request_params,
                        )
                    ):
                        yield result
                    return
                except CaikitRuntimeException as err:
                    error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                    error_content = {
                        "details": err.message,
                        "code": error_code,
                        "id": err.id,
                    }
                except Exception as err:  # pylint: disable=broad-exception-caught
                    error_code = 500
                    error_content = {
                        "details": f"Unhandled exception: {str(err)}",
                        "code": error_code,
                        "id": None,
                    }
                    log.error("<RUN51881106E>", err, exc_info=True)

                # If an error occurs, yield an error response and terminate
                yield ServerSentEvent(data=json.dumps(error_content))

            return EventSourceResponse(_generator())

    def _get_route(self, rpc: CaikitRPCBase) -> str:
        """Get the REST route for this rpc"""
        if rpc.name.endswith("Predict"):
            task_name = re.sub(
                r"(?<!^)(?=[A-Z])",
                "-",
                re.sub("Task$", "", re.sub("Predict$", "", rpc.name)),
            ).lower()
            route = "/".join(
                [self.config.runtime.http.route_prefix, "{model_id}", "task", task_name]
            )
            if route[0] != "/":
                route = "/" + route
            return route
        if rpc.name.endswith("Train"):

            route = "/".join(
                [self.config.runtime.http.route_prefix, "{model_id}", rpc.name]
            )
            if route[0] != "/":
                route = "/" + route
            return route
        raise NotImplementedError("No support for train rpcs yet!")

    def _get_request_dataobject(
        self, rpc: CaikitRPCBase, input_streaming: bool
    ) -> Type[DataBase]:
        """Get the dataobject request for the given rpc"""
        is_inference_rpc = hasattr(rpc, "task")
        if is_inference_rpc:
            required_params = rpc.task.get_required_parameters(input_streaming)
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

    @classmethod
    # pylint: disable=too-many-return-statements
    def _get_pydantic_type(cls, field_type: type) -> type:
        """Recursive helper to get a valid pydantic type for every field type"""
        # pylint: disable=too-many-return-statements

        # Leaves: we should have primitive types and enums
        if np.issubclass_(field_type, np.integer):
            return int
        if np.issubclass_(field_type, np.floating):
            return float
        if field_type in (int, float, bool, str, bytes, type(None)):
            return field_type
        if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
            return field_type

        # These can be nested within other data models
        if (
            isinstance(field_type, type)
            and issubclass(field_type, DataBase)
            and not issubclass(field_type, pydantic.BaseModel)
        ):
            # NB: for data models we're calling the data model conversion fn
            return cls._dataobject_to_pydantic(field_type)

        # And then all of these types can be nested in other type annotations
        if get_origin(field_type) is Annotated:
            return cls._get_pydantic_type(get_args(field_type)[0])
        if get_origin(field_type) is Union:
            return Union[  # type: ignore
                tuple(
                    (
                        cls._get_pydantic_type(arg_type)
                        for arg_type in get_args(field_type)
                    )
                )
            ]
        if get_origin(field_type) is list:
            return List[cls._get_pydantic_type(get_args(field_type)[0])]

        raise TypeError(f"Cannot get pydantic type for type [{field_type}]")

    @classmethod
    def _dataobject_to_pydantic(
        cls, dm_class: Type[DataBase]
    ) -> Type[pydantic.BaseModel]:
        """Make a pydantic model based on the given proto message by using the data
        model class annotations to mirror as a pydantic model
        """
        if dm_class in PYDANTIC_REGISTRY:
            return PYDANTIC_REGISTRY[dm_class]

        annotations = {
            field_name: cls._get_pydantic_type(field_type)
            for field_name, field_type in dm_class.__annotations__.items()
        }
        pydantic_model = type(pydantic.BaseModel)(
            dm_class.get_proto_class().DESCRIPTOR.full_name,
            (pydantic.BaseModel,),
            {
                "__annotations__": annotations,
                **{name: None for name in dm_class.__annotations__},
            },
        )
        PYDANTIC_REGISTRY[dm_class] = pydantic_model
        return pydantic_model

    @staticmethod
    def _health_check() -> str:
        log.debug4("Server healthy")
        return "OK"


## Main ########################################################################


def main():
    caikit.core.toolkit.logging.configure()
    server = RuntimeHTTPServer()
    server.start()


if __name__ == "__main__":
    main()
