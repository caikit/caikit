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
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Type, Union, get_args
import asyncio
import inspect
import io
import json
import os
import signal
import ssl
import tempfile
import threading
import time
import traceback
import uuid

# Third Party
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse
from grpc import StatusCode
from sse_starlette import EventSourceResponse, ServerSentEvent
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
from .pydantic_wrapper import (
    dataobject_to_pydantic,
    pydantic_from_request,
    pydantic_to_dataobject,
)
from .request_aborter import HttpRequestAborter
from .utils import convert_json_schema_to_multipart
from caikit.config.config import get_config, merge_configs
from caikit.core.data_model import DataBase
from caikit.core.data_model.dataobject import make_dataobject
from caikit.core.exceptions import error_handler
from caikit.core.exceptions.caikit_core_exception import CaikitCoreException
from caikit.core.toolkit.name_tools import snake_to_upper_camel
from caikit.core.toolkit.sync_to_async import async_wrap_iter
from caikit.runtime.names import (
    EXTRA_OPENAPI_KEY,
    HEALTH_ENDPOINT,
    MODEL_ID,
    MODEL_MANAGEMENT_ENDPOINT,
    MODEL_MANAGEMENT_SERVICE_SPEC,
    MODELS_INFO_ENDPOINT,
    OPTIONAL_INPUTS_KEY,
    REQUIRED_INPUTS_KEY,
    RUNTIME_INFO_ENDPOINT,
    STATUS_CODE_TO_HTTP,
    TRAINING_MANAGEMENT_ENDPOINT,
    TRAINING_MANAGEMENT_SERVICE_SPEC,
    StreamEventTypes,
    get_http_route_name,
)
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
from caikit.runtime.servicers.model_management_servicer import (
    ModelManagementServicerImpl,
)
from caikit.runtime.servicers.training_management_servicer import (
    TrainingManagementServicerImpl,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import get_dynamic_module

## Globals #####################################################################

log = alog.use_channel("SERVR-HTTP")
error = error_handler.get(log)


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

        # Construct FastAPI spec and create placeholders for open api deps
        self.app = FastAPI()
        self._openapi_defs = {}

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
        self.model_management_servicer = None
        self.training_management_servicer = None
        self.info_servicer = InfoServicer()

        # NOTE: The order that the modules are bound is directly reflected in
        #   the swagger UI, so we intentionally bind inference, training,
        #   management, info, then health.

        # Set up inference if enabled
        if self.enable_inference:
            log.info("<RUN77183426I>", "Enabling HTTP inference service")
            self.global_predict_servicer = GlobalPredictServicer(
                self.inference_service, interrupter=self.interrupter
            )
            self._bind_routes(self.inference_service)

        # Set up training if enabled
        if self.enable_training:
            log.info("<RUN77183427I>", "Enabling HTTP training service")
            self.global_train_servicer = GlobalTrainServicer(self.training_service)
            self._bind_routes(self.training_service)

        # Set up management services
        if self.enable_inference:
            self.model_management_servicer = ModelManagementServicerImpl()
            self._bind_model_management_routes()
        if self.enable_training:
            self.training_management_servicer = TrainingManagementServicerImpl()
            self._bind_training_management_routes()

        # Add runtime info endpoints
        self.app.get(RUNTIME_INFO_ENDPOINT, response_class=JSONResponse)(
            self.info_servicer.get_version_dict
        )
        self.app.get(MODELS_INFO_ENDPOINT, response_class=JSONResponse)(
            self._model_info
        )

        # Add the health endpoint
        self.app.get(HEALTH_ENDPOINT, response_class=PlainTextResponse)(
            self._health_check
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
            # if not set in config, this is None and unvicorn accepts None or
            # number of seconds
            unvicorn_timeout_graceful_shutdown = (
                get_config().runtime.http.server_shutdown_grace_period_seconds
            )
            server_config = dict(**get_config().runtime.http.server_config)
            overlapping_tls_config = set(tls_kwargs).intersection(server_config)
            error.value_check(
                "<RUN30233180E>",
                not overlapping_tls_config,
                "Found overlapping config keys between TLS and server_config: %s",
                overlapping_tls_config,
            )
            config_kwargs = {
                "host": "0.0.0.0",
                "port": self.port,
                "log_level": None,
                "log_config": None,
                "timeout_graceful_shutdown": unvicorn_timeout_graceful_shutdown,
            }
            overlapping_kwarg_config = set(config_kwargs).intersection(server_config)
            error.value_check(
                "<RUN99488934E>",
                not overlapping_kwarg_config,
                "Found caikit-managed uvicorn config in server_config: %s",
                overlapping_kwarg_config,
            )

            # Set the default concurrency limit if not changed from the default
            # sentinel value
            concurrency_limit = server_config.get("limit_concurrency", 0)
            if not concurrency_limit or not isinstance(concurrency_limit, int):
                log.info(
                    "<RUN57106697I>", "Running HTTP server with unlimited concurrency"
                )
                concurrency_limit = None
            elif concurrency_limit < 0:
                max_threads = self.thread_pool._max_workers
                concurrency_limit = max_threads * 2
                log.info(
                    "<RUN57106696I>",
                    "Limiting HTTP server concurrency to %d",
                    concurrency_limit,
                )
            server_config["limit_concurrency"] = concurrency_limit

            # Make sure the config loads TLS files here so they can safely be
            # deleted if they're ephemeral
            config = uvicorn.Config(
                self.app,
                **config_kwargs,
                **tls_kwargs,
                **server_config,
            )
            config.load()

        # Build the server with the loaded config
        self.server = uvicorn.Server(config=config)

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
        log.info(
            "<RUN10001002I>",
            "Caikit Runtime is serving http on port: %s with thread pool size: %s",
            self.port,
            self.thread_pool._max_workers,
        )

        if self.interrupter:
            self.interrupter.start()

        # Patch the openapi spec to ensure defs are properly added
        self._patch_openapi_spec()

        # Patch the exit handler to retain correct signal handling behavior
        self._patch_exit_handler()

        if blocking:
            self.server.run()
        else:
            self._run_in_thread()

    def stop(self):
        """Stop the server, with an optional grace period.

        Args:
            grace_period_seconds (Union[float, int]): Grace period for service shutdown.
                Defaults to application config
        """
        log.info("Shutting down http server")

        if (
            self._uvicorn_server_thread is not None
            and self._uvicorn_server_thread.is_alive()
        ):
            # This is required to notify the server in the thread to exit
            self.server.should_exit = True
            self._uvicorn_server_thread.join()

        # Ensure we flush out any remaining billing metrics and stop metering
        if self.config.runtime.metering.enabled and self.global_predict_servicer:
            self.global_predict_servicer.stop_metering()

        # Shut down the model manager's model polling if enabled
        self._shut_down_model_manager()

        if self.interrupter:
            self.interrupter.stop()

    ######################
    ## Static Endpoints ##
    ######################

    def _model_info(
        self, model_ids: Annotated[List[str], Query(default_factory=list)]
    ) -> Dict[str, Any]:
        """Create wrapper for get_models_info so model_ids can be marked as a query parameter"""
        try:
            return self.info_servicer.get_models_info_dict(model_ids)
        except Exception as err:
            if error_content := self._handle_exception(err):
                return Response(
                    content=json.dumps(error_content), status_code=error_content["code"]
                )
            raise

    @staticmethod
    def _health_check() -> str:
        log.debug4("Server healthy")
        return "OK"

    async def _deploy_model(self, context: Request) -> Response:
        """POST handler for deploying a model"""
        assert hasattr(
            self, "_deploy_pydantic_request"
        ), "Cannot call _deploy_model without _bind_model_management_routes"
        try:
            request = await pydantic_from_request(
                self._deploy_pydantic_request, context
            )
            result = self.model_management_servicer.deploy_model(
                request.model_id,
                {f.filename: f.data for f in request.model_files},
            )
            return Response(
                content=result.to_json(),
                media_type="application/json",
            )
        except Exception as err:
            if error_content := self._handle_exception(err):
                return Response(
                    content=json.dumps(error_content),
                    status_code=error_content["code"],
                )
            raise

    async def _undeploy_model(self, model_id: Annotated[str, Query]) -> Response:
        """DELETE handler for undeploying a model"""
        try:
            result = self.model_management_servicer.undeploy_model(model_id)
            return Response(content=result.to_json(), media_type="application/json")
        except Exception as err:
            if error_content := self._handle_exception(err):
                return Response(
                    content=json.dumps(error_content),
                    status_code=error_content["code"],
                )
            raise

    def _get_training_status(self, training_id: Annotated[str, Query]) -> Response:
        """GET handler for fetching a training"""
        try:
            result = self.training_management_servicer.get_training_status(training_id)
            return Response(
                content=result.to_json(),
                media_type="application/json",
            )
        except Exception as err:
            if error_content := self._handle_exception(err):
                return Response(
                    content=json.dumps(error_content),
                    status_code=error_content["code"],
                )
            raise

    def _cancel_training(self, training_id: Annotated[str, Query]) -> Response:
        """DELETE handler for undeploying a model"""
        try:
            result = self.training_management_servicer.cancel_training(training_id)
            return Response(
                content=result.to_json(),
                media_type="application/json",
            )
        except Exception as err:
            if error_content := self._handle_exception(err):
                return Response(
                    content=json.dumps(error_content),
                    status_code=error_content["code"],
                )
            raise

    #####################
    ## Request Binding ##
    #####################

    def _bind_routes(self, service: ServicePackage):
        """Bind all caikit rpcs as routes to the given app"""
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

    def _bind_model_management_routes(self):
        """Bind the routes for deploy/undeploy"""

        # Bind POST to deploy a model
        deploy_spec = MODEL_MANAGEMENT_SERVICE_SPEC["service"]["rpcs"][0]
        assert deploy_spec["name"] == "DeployModel"
        deploy_dataobject_request = DataBase.get_class_for_name(
            deploy_spec["input_type"]
        )
        deploy_pydantic_request = dataobject_to_pydantic(deploy_dataobject_request)
        deploy_dataobject_response = DataBase.get_class_for_name(
            deploy_spec["output_type"]
        )
        deploy_pydantic_response = dataobject_to_pydantic(deploy_dataobject_response)

        # Bind deploy_model
        # NOTE: The deploy_pydantic_request must be bound to `self` so that it
        #   it does not need to be bound to the `_deploy_model` function which
        #   is hard since its async.
        self._deploy_pydantic_request = deploy_pydantic_request
        self.app.post(
            MODEL_MANAGEMENT_ENDPOINT,
            responses=self._get_response_openapi(
                deploy_dataobject_response, deploy_pydantic_response
            ),
            description=ModelManagementServicerImpl.DeployModel.__doc__,
            openapi_extra=self._get_request_openapi(deploy_pydantic_request),
            response_class=Response,
        )(self._deploy_model)

        # Bind DELETE to undeploy a model
        undeploy_spec = MODEL_MANAGEMENT_SERVICE_SPEC["service"]["rpcs"][1]
        assert undeploy_spec["name"] == "UndeployModel"
        undeploy_dataobject_response = DataBase.get_class_for_name(
            undeploy_spec["output_type"]
        )
        undeploy_pydantic_response = dataobject_to_pydantic(
            undeploy_dataobject_response
        )

        self.app.delete(
            MODEL_MANAGEMENT_ENDPOINT,
            responses=self._get_response_openapi(
                undeploy_dataobject_response, undeploy_pydantic_response
            ),
            description=ModelManagementServicerImpl.UndeployModel.__doc__,
            response_class=Response,
        )(self._undeploy_model)

    def _bind_training_management_routes(self):
        """Bind the routes for get/cancel trainings"""

        # Bind GET to fetch a training
        get_spec = TRAINING_MANAGEMENT_SERVICE_SPEC["service"]["rpcs"][0]
        assert get_spec["name"] == "GetTrainingStatus"
        get_dataobject_response = DataBase.get_class_for_name(get_spec["output_type"])
        get_pydantic_response = dataobject_to_pydantic(get_dataobject_response)

        self.app.get(
            TRAINING_MANAGEMENT_ENDPOINT,
            responses=self._get_response_openapi(
                get_dataobject_response, get_pydantic_response
            ),
            response_class=Response,
        )(self._get_training_status)

        # Bind DELETE to cancel a training
        cancel_spec = TRAINING_MANAGEMENT_SERVICE_SPEC["service"]["rpcs"][1]
        assert cancel_spec["name"] == "CancelTraining"
        cancel_dataobject_response = DataBase.get_class_for_name(
            cancel_spec["output_type"]
        )
        cancel_pydantic_response = dataobject_to_pydantic(cancel_dataobject_response)

        self.app.delete(
            TRAINING_MANAGEMENT_ENDPOINT,
            responses=self._get_response_openapi(
                cancel_dataobject_response, cancel_pydantic_response
            ),
            response_class=Response,
        )(self._cancel_training)

    def _train_add_unary_input_unary_output_handler(self, rpc: ModuleClassTrainRPC):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = dataobject_to_pydantic(
            DataBase.get_class_for_name(rpc.request.name)
        )
        response_data_object = self._get_response_dataobject(rpc)
        pydantic_response = dataobject_to_pydantic(response_data_object)

        @self.app.post(
            get_http_route_name(rpc.name),
            responses=self._get_response_openapi(
                response_data_object, pydantic_response
            ),
            description=rpc._method._method_pointer.__doc__,
            openapi_extra=self._get_request_openapi(pydantic_request),
            response_class=Response,
        )
        # pylint: disable=unused-argument
        async def _handler(context: Request) -> Response:
            log.debug("In unary handler for %s", rpc.name)
            loop = asyncio.get_running_loop()

            try:
                # build request DM object
                request = await pydantic_from_request(pydantic_request, context)
                http_request_dm_object = pydantic_to_dataobject(request)

                call = partial(
                    self.global_train_servicer.run_training_job,
                    request=http_request_dm_object.to_proto(),
                    module=rpc.clz,
                    training_output_dir=None,  # pass None so that GTS picks up the config one # TODO: double-check? # noqa: E501
                    # context=context,
                    wait=False,
                )
                result = await loop.run_in_executor(None, call)
                if response_data_object.supports_file_operations:
                    return self._format_file_response(result)

                return Response(content=result.to_json(), media_type="application/json")
            except Exception as err:
                if error_content := self._handle_exception(err):
                    return Response(
                        content=json.dumps(error_content),
                        status_code=error_content["code"],
                    )
                raise

    def _add_unary_input_unary_output_handler(self, rpc: TaskPredictRPC):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = dataobject_to_pydantic(self._get_request_dataobject(rpc))
        request_openapi = self._get_request_openapi(pydantic_request)
        response_data_object = self._get_response_dataobject(rpc)
        pydantic_response = dataobject_to_pydantic(response_data_object)

        # Merge the DataObject openapi schema into the task schema
        task_api_schema = merge_configs(
            rpc.task.get_metadata().get(EXTRA_OPENAPI_KEY, {}), request_openapi
        )

        @self.app.post(
            get_http_route_name(rpc.name),
            responses=self._get_response_openapi(
                response_data_object, pydantic_response
            ),
            include_in_schema=rpc.task.get_visibility(),
            description=rpc.task.__doc__,
            openapi_extra=task_api_schema,
            response_class=Response,
        )
        # pylint: disable=unused-argument
        async def _handler(
            context: Request,
        ) -> Response:
            try:
                request = await pydantic_from_request(pydantic_request, context)
                request_params = self._get_request_params(rpc, request)

                model_id = self._get_model_id(request)
                log.debug4(
                    "Sending request %s to model id %s", request_params, model_id
                )

                log.debug("In unary handler for %s for model %s", rpc.name, model_id)
                loop = asyncio.get_running_loop()

                log.debug4(
                    "Sending request %s to model id %s", request_params, model_id
                )

                aborter_context = (
                    HttpRequestAborter(context) if self.interrupter else nullcontext()
                )

                with aborter_context as aborter:
                    # TODO: use `async_wrap_*`?
                    call = partial(
                        self.global_predict_servicer.predict_model,
                        model_id=model_id,
                        request_name=rpc.request.name,
                        input_streaming=False,
                        output_streaming=False,
                        task=rpc.task,
                        aborter=aborter,
                        context=context,
                        **request_params,
                    )
                    result = await loop.run_in_executor(self.thread_pool, call)
                    log.debug4("Response from model %s is %s", model_id, result)

                if response_data_object.supports_file_operations:
                    return self._format_file_response(result)

                return Response(content=result.to_json(), media_type="application/json")

            except Exception as err:
                if error_content := self._handle_exception(err):
                    return Response(
                        content=json.dumps(error_content),
                        status_code=error_content["code"],
                    )
                raise

    def _add_unary_input_stream_output_handler(self, rpc: TaskPredictRPC):
        pydantic_request = dataobject_to_pydantic(self._get_request_dataobject(rpc))
        request_openapi = self._get_request_openapi(pydantic_request)
        pydantic_response = dataobject_to_pydantic(self._get_response_dataobject(rpc))

        # Merge the DataObject openapi schema into the task schema
        task_api_schema = merge_configs(
            rpc.task.get_metadata().get(EXTRA_OPENAPI_KEY, {}), request_openapi
        )

        # pylint: disable=unused-argument
        @self.app.post(
            get_http_route_name(rpc.name),
            response_model=pydantic_response,
            description=rpc.task.__doc__,
            include_in_schema=rpc.task.get_visibility(),
            openapi_extra=task_api_schema,
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

                    aborter_context = (
                        HttpRequestAborter(context)
                        if self.interrupter
                        else nullcontext()
                    )

                    with aborter_context as aborter:
                        log.debug("In stream generator for %s", rpc.name)
                        async for result in async_wrap_iter(
                            self.global_predict_servicer.predict_model(
                                model_id=model_id,
                                request_name=rpc.request.name,
                                input_streaming=False,
                                output_streaming=True,
                                task=rpc.task,
                                aborter=aborter,
                                context=context,
                                **request_params,
                            ),
                            pool=self.thread_pool,
                        ):
                            yield ServerSentEvent(
                                data=result.to_json(),
                                event=StreamEventTypes.MESSAGE.value,
                            )

                    return
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
                        "id": uuid.uuid4().hex,
                    }
                except Exception as err:
                    if (error_content := self._handle_exception(err)) is None:
                        raise
                yield ServerSentEvent(
                    data=json.dumps(error_content), event=StreamEventTypes.ERROR.value
                )

            return EventSourceResponse(_generator())

    #############
    ## Helpers ##
    #############

    @staticmethod
    def _handle_exception(err: Exception) -> Optional[dict]:
        """Common exception handling. This function will return a dict with
        "id," "code," and "details" if the exception should be handled with a
        returned error body. If None is returned, the exception should be
        re-raised.
        """
        # Native FastAPI exceptions should be reraised directly
        if isinstance(
            err, (HTTPException, RequestValidationError, ResponseValidationError)
        ):
            return None

        # Convert caikit exceptions to error bodies
        if isinstance(err, (CaikitCoreException, CaikitRuntimeException)):
            error_code = STATUS_CODE_TO_HTTP.get(err.status_code, 500)
            error_content = {
                "details": err.message,
                "code": error_code,
                "id": err.id,
            }
            log.error("<RUN87691106E>", error_content, exc_info=True)
            return error_content

        # Other exceptions are 500s
        error_code = 500
        error_content = {
            "details": f"Unhandled exception: {str(err)}",
            "code": error_code,
            "id": uuid.uuid4().hex,
        }
        log.error("<RUN51231106E>", error_content, exc_info=True)
        return error_content

    def _run_in_thread(self):
        """Run the server in an isolated thread"""
        self._uvicorn_server_thread = threading.Thread(target=self.server.run)
        self._uvicorn_server_thread.start()
        while not self.server.started:
            time.sleep(1e-3)
        log.info("HTTP Server is running in thread")

    def _get_model_id(self, request: Type[pydantic.BaseModel]) -> str:
        """Get the model id from the payload"""
        request_kwargs = dict(request)
        model_id = request_kwargs.get(MODEL_ID)
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

    def _get_request_openapi(
        self, pydantic_model: Union[pydantic.BaseModel, Type, Type[pydantic.BaseModel]]
    ):
        """Helper to generate the openapi schema for a given request"""

        # Get the json schema from the pydantic model or TypeAdapter
        if inspect.isclass(pydantic_model) and issubclass(
            pydantic_model, pydantic.BaseModel
        ):
            raw_schema = pydantic_model.model_json_schema()
        else:
            raw_schema = pydantic.TypeAdapter(pydantic_model).json_schema()

        # Update openapi defs with defs from raw schema
        for def_name, schema in raw_schema.pop("$defs", {}).items():
            self._openapi_defs[def_name] = schema

        multipart_schema = convert_json_schema_to_multipart(
            raw_schema, self._openapi_defs
        )

        return {
            "requestBody": {
                "content": {
                    "multipart/form-data": {"schema": multipart_schema},
                    "application/json": {"schema": raw_schema},
                },
                "required": True,
            }
        }

    def _get_response_openapi(
        self,
        dm_class: Type[DataBase],
        pydantic_model: Union[Type, Type[pydantic.BaseModel]],
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

            for def_name, schema in json_schema.pop("$defs", {}).items():
                self._openapi_defs[def_name] = schema

            response_schema = {"application/json": json_schema}

        output = {200: {"content": response_schema}}
        return output

    @contextmanager
    def _tls_files(self) -> Iterable[_TlsFiles]:
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

    def _patch_openapi_spec(self):
        """
        FastAPI does not have a way to dynamically add openapi defs
        for specific paths. This means we must wait till the very end
        to update the def values. This does allow for adding context
        specific fields though which is beneficial.

        """
        # Parse the library name into a more human readable version
        library_name = "FastAPI"
        if get_config().runtime.library:
            library_name = snake_to_upper_camel(get_config().runtime.library)

        # Attempt to load in the runtime library to fetch the module's docstring. This
        # is safe to do in _patch_openapi_spec because the runtime service generation
        # has already ocurred during super().__init__()
        try:
            imported_module = get_dynamic_module(get_config().runtime.library)
            openapi_description = getattr(imported_module, "__doc__", "")
        except ImportError:
            log.debug(
                "Unable to import runtime library %s when trying to fetch module description",
                get_config().runtime.library,
            )
            openapi_description = ""

        # Construct openapi schema from fastapi routes
        openapi_schema = get_openapi(
            title=library_name,
            version=get_config().runtime.version_info.runtime_image or "",
            description=openapi_description,
            routes=self.app.routes,
        )
        openapi_schema.setdefault("components", {}).setdefault("schemas", {}).update(
            self._openapi_defs
        )

        def _recursively_update_defs_to_component(obj: Any) -> dict:
            """Helper function to replace $defs references with components/schemas"""
            if isinstance(obj, dict):
                return {
                    key: _recursively_update_defs_to_component(val)
                    for key, val in obj.items()
                }
            elif isinstance(obj, list):
                return [_recursively_update_defs_to_component(val) for val in obj]
            elif isinstance(obj, str):
                return obj.replace("$defs", "components/schemas")
            else:
                return obj

        # Update $def references to components/schemas
        openapi_schema = _recursively_update_defs_to_component(openapi_schema)
        self.app.openapi_schema = openapi_schema

    def _patch_exit_handler(self):
        """
        🌶️🌶️🌶️ Here there are dragons! 🌶️🌶️🌶️
        uvicorn will explicitly set the interrupt handler to `server.handle_exit` when
        `server.run()` is called. That will override any other signal handlers that we
        may have tried to set.

        To work around this, we:
        1. Register `server.handle_exit` as a SIGINT/SIGTERM signal handler ourselves, so that it
          is invoked on interrupt and terminate
        2. Set `server.handle_exit` to the existing SIGINT signal handler, so that when the uvicorn
          server explicitly overrides the signal handler for SIGINT and SIGTERM to this, it has no
          effect.

        Since uvicorn overrides SIGINT and SIGTERM with a single common handler, any special
            handlers added for SIGTERM but not SIGINT will not be invoked.
        """
        original_exit_handler = self.server.handle_exit
        self._add_signal_handler(signal.SIGINT, original_exit_handler)
        self._add_signal_handler(signal.SIGTERM, original_exit_handler)
        self.server.handle_exit = signal.getsignal(signal.SIGINT)


## Main ########################################################################


def main(blocking: bool = True):
    server = RuntimeHTTPServer()
    server.start(blocking)


if __name__ == "__main__":
    main()
