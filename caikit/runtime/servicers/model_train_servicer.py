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
import json
import traceback

# Third Party
from google.protobuf.json_format import ParseDict
import grpc

# First Party
import alog

# Local
from caikit.runtime.protobufs import process_pb2, process_pb2_grpc
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.servicer_util import snake_to_upper_camel
import caikit.core

log = alog.use_channel("MT-SERVICR-I")


class ModelTrainServicerImpl(process_pb2_grpc.ProcessServicer):
    """This class contains the implementation of the Model Train process.proto
    interface (a.k.a. the Process Servicer).
    """

    def __init__(self, training_service: ServicePackage):
        self._training_service = training_service
        self._gts = GlobalTrainServicer(self._training_service)

    def Run(self, request, context=None):
        """`Run` RPC -- launches a training job.
        Args:
            request(process_pb2.ProcessRequest):
                Generated from process.proto
            context(grpc.ServicerContext): Context object (contains request metadata, etc)
        Returns:
            process_pb2.ProcessResponse:
                Generated from process.proto
        """
        log.info("<RUN02584562I>", "Calling ModelTrainServicer.Run!")
        log.debug("<RUN91039903D>", "ProcessRequest: %s", request)

        try:
            request_dict = request.request_dict

            # get the model to train
            train_module = caikit.core.registries.module_registry().get(
                request_dict["train_module"]
            )
            log.debug("<RUN76043064D>", "train_module: %s", train_module)
            if train_module is None:
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Model Train not able to parse module for this Train Request",
                )

            # prepare the model's train request
            training_params = json.loads(request_dict["training_params"])
            module_split = train_module.__module__.split(".")
            request_name = snake_to_upper_camel(
                f"{module_split[1]}_{module_split[2]}_{train_module.__name__}_TrainRequest"
            )
            log.debug("<RUN22972949D>", "request_name: %s", request_name)

            if not hasattr(self._training_service.messages, request_name):
                raise CaikitRuntimeException(
                    grpc.StatusCode.INTERNAL,
                    f"Model Train not able to create a request for {request_name}",
                )
            req_name_func = getattr(self._training_service.messages, request_name)
            # construct the protobufs message request with train params
            train_message_request = ParseDict(
                training_params,
                req_name_func(),
            )
            # make the train call
            log.debug("Training output dir: %s", request.training_output_dir)
            training_response = self._gts.run_training_job(
                request=train_message_request,
                model=train_module,
                training_id=request.trainingID,
                training_output_dir=request.training_output_dir,
                wait=True,
            )
            log.debug("<RUN00837184D>", "training_response: %s", training_response)
            # return response
            process_response = process_pb2.ProcessResponse(
                trainingID=training_response.training_id,
                customTrainingID=request.customTrainingID,
            )
            return process_response

        except CaikitRuntimeException as e:
            # pylint: disable=R0801
            log.warning(
                {
                    "log_code": "<RUN50531303W>",
                    "message": e.message,
                    "error_id": e.id,
                    **e.metadata,
                }
            )
            raise e

        except (TypeError, ValueError) as e:
            log.warning(
                {
                    "log_code": "<RUN490449039W>",
                    "message": repr(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Exception raised during inference. This may be a problem with your input: {e}",
            ) from e

        except Exception as e:
            log.warning(
                {
                    "log_code": "<RUN48349070W>",
                    "message": repr(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL, "Unhandled exception during prediction"
            ) from e
