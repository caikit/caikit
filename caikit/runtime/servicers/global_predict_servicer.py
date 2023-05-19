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
from importlib.metadata import version
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
from grpc import StatusCode
from prometheus_client import Counter, Summary

# First Party
import alog

# Local
from caikit import get_config
from caikit.runtime.metrics.rpc_meter import RPCMeter
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import clean_lib_names
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    build_proto_response,
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

    def Predict(self, request, context):
        """Global predict RPC -- Mocks the invocation of a Caikit Library module.run()
        method for a loaded Caikit Library model

        Args:
            request(object):
                A deserialized RPC request message
            context(grpc.ServicerContext): Context object (contains request metadata, etc)

        Returns:
            response (object):
                A Caikit Library data model response object
        """
        desc_name = request.DESCRIPTOR.name
        outer_scope_name = "GlobalPredictServicerMock.Predict:%s" % desc_name
        inner_scope_name = (
            "GlobalPredictServicerImpl.Predict.caikit_library_run:%s" % desc_name
        )

        try:
            with alog.ContextLog(log.debug, outer_scope_name):
                # Make sure the request has a model before doing anything
                model_id = get_metadata(context, self.MODEL_MESH_MODEL_ID_KEY)
                # Retrieve the model from the model manager
                log.debug("<RUN52259029D>", "Retrieving model '%s'", model_id)
                model = self._model_manager.retrieve_model(model_id)

                # Unmarshall the request object into the required module run argument(s)
                with PREDICT_FROM_PROTO_SUMMARY.labels(
                    grpc_request=desc_name, model_id=model_id
                ).time():
                    caikit_library_request = build_caikit_library_request_dict(
                        request, model.run
                    )

                # NB: we previously recorded the size of the request, and timed this module to
                # provide a rudimentary throughput metric of size / time
                with alog.ContextLog(log.debug, inner_scope_name):
                    with PREDICT_CAIKIT_LIBRARY_SUMMARY.labels(
                        grpc_request=desc_name, model_id=model_id
                    ).time():
                        if self.use_abortable_threads:
                            work = AbortableAction(
                                CallAborter(context),
                                model.run,
                                **caikit_library_request,
                            )
                            response = work.do()
                        else:
                            response = model.run(**caikit_library_request)

                # Marshall the response to the necessary return type
                with PREDICT_TO_PROTO_SUMMARY.labels(
                    grpc_request=desc_name, model_id=model_id
                ).time():
                    response_proto = build_proto_response(response)

            # Update Prometheus metrics
            PREDICT_RPC_COUNTER.labels(
                grpc_request=desc_name, code=StatusCode.OK.name, model_id=model_id
            ).inc()
            if get_config().runtime.metering.enabled:
                self.rpc_meter.update_metrics(str(type(model)))
            return response_proto

        except CaikitRuntimeException as e:
            log_dict = {
                "log_code": "<RUN50530380W>",
                "message": e.message,
                "model_id": get_metadata(context, self.MODEL_MESH_MODEL_ID_KEY),
                "error_id": e.id,
            }
            log.warning({**log_dict, **e.metadata})
            PREDICT_RPC_COUNTER.labels(
                grpc_request=desc_name, code=e.status_code.name, model_id=model_id
            ).inc()
            raise e

        # Duplicate code in global_train_servicer
        # pylint: disable=duplicate-code
        except (TypeError, ValueError) as e:
            log_dict = {
                "log_code": "<RUN490439039W>",
                "message": repr(e),
                "model_id": get_metadata(context, self.MODEL_MESH_MODEL_ID_KEY),
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            PREDICT_RPC_COUNTER.labels(
                grpc_request=desc_name,
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
                "model_id": get_metadata(context, self.MODEL_MESH_MODEL_ID_KEY),
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            PREDICT_RPC_COUNTER.labels(
                grpc_request=desc_name, code=StatusCode.INTERNAL.name, model_id=model_id
            ).inc()
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "Unhandled exception during prediction"
            ) from e
