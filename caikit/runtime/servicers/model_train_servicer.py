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
import os
import traceback

# Third Party
from google.protobuf.json_format import ParseDict
from google.protobuf.message import Message as ProtoMessageType
import grpc

# First Party
import alog

# Local
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    File,
    ListOfFiles,
)
from caikit.runtime.protobufs import process_pb2, process_pb2_grpc
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.data_stream_source import DataStreamSourceBase
from caikit.runtime.service_generation.rpcs import ModuleClassTrainRPC
from caikit.runtime.servicers.global_train_servicer import GlobalTrainServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit.core

log = alog.use_channel("MT-SERVICR-I")


class ModelTrainServicerImpl(process_pb2_grpc.ProcessServicer):
    """This class contains the implementation of the Model Train process.proto
    interface (a.k.a. the Process Servicer).
    """

    def __init__(self, training_service: ServicePackage):
        self._training_service = training_service
        self._gts = GlobalTrainServicer(self._training_service)

    def Run(self, request, context):
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
            request_name = ModuleClassTrainRPC.module_class_to_req_name(train_module)
            log.debug("<RUN22972949D>", "request_name: %s", request_name)

            if not hasattr(self._training_service.messages, request_name):
                raise CaikitRuntimeException(
                    grpc.StatusCode.INTERNAL,
                    f"Model Train not able to create a request for {request_name}",
                )
            request_class = getattr(self._training_service.messages, request_name)
            # construct the protobufs message request with train params
            train_message_request = ParseDict(
                training_params,
                request_class(),
            )
            # update any file references relative to the training input dir
            self._update_file_references(
                train_message_request, request.training_input_dir
            )

            # make the train call
            log.debug("Training output dir: %s", request.training_output_dir)

            training_response = self._gts.run_training_job(
                request=train_message_request,
                module=train_module,
                training_output_dir=request.training_output_dir,
                context=context,
                wait=True,
                # TODO: This usage of the server will sit behind Model Train and
                #   will therefore always use a local trainer which supports the
                #   external_training_id override. If that is ever not the case
                #   this will be brokenly passing that kwarg through to the
                #   module's train function.
                external_training_id=request.trainingID,
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
                f"Exception raised during training. This may be a problem with your input: {e}",
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
                grpc.StatusCode.INTERNAL, "Unhandled exception during training"
            ) from e

    def _update_file_references(
        self, train_message_request: ProtoMessageType, training_input_dir: str
    ):
        # Check through request and redirect file references to the training input dir
        # This requires a bit of fancy footwork:
        # 1. Get the real data model for the request
        dm_class = caikit.core.data_model.DataBase.get_class_for_proto(
            train_message_request
        )
        training_request_data_model = dm_class.from_proto(train_message_request)
        # 2. Find any data streams
        for attr_name in training_request_data_model.__annotations__.keys():
            val = getattr(training_request_data_model, attr_name)
            if isinstance(val, DataStreamSourceBase):
                # 3. Look for file pointers and update them
                source = val.data_stream
                if isinstance(source, File):
                    source.filename = self._update_file_ref(
                        source.filename, training_input_dir
                    )
                if isinstance(source, ListOfFiles):
                    source.files = [
                        self._update_file_ref(filename, training_input_dir)
                        for filename in source.files
                    ]
                if isinstance(source, Directory):
                    source.dirname = self._update_file_ref(
                        source.dirname, training_input_dir
                    )
                # Importantly we must set the attribute directly back on the proto message
                # Re-serializing the entire request wouldn't properly propagate the existence
                # of unset fields
                proto_val = getattr(train_message_request, attr_name)
                proto_val.CopyFrom(val.to_proto())

    @staticmethod
    def _update_file_ref(file_path: str, training_input_dir: str) -> str:
        """Append the training_input_dir to the file_path if the resulting path exists"""
        updated_path = os.path.join(training_input_dir, file_path)
        if os.path.exists(updated_path):
            return updated_path
        return file_path
