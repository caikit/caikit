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
from typing import Optional, Type, Union
import os
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtoMessageType
from grpc import ServicerContext, StatusCode

# First Party
import alog

# Local
from caikit import get_config
from caikit.core import MODEL_MANAGER, ModuleBase
from caikit.interfaces.common.data_model.stream_sources import S3Path
from caikit.interfaces.runtime.data_model import TrainingJob
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import ModuleClassTrainRPC
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import clean_lib_names, get_data_model
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    validate_data_model,
)
import caikit.core

log = alog.use_channel("GT-SERVICR-I")
error = caikit.core.toolkit.errors.error_handler.get(log)

# Protobuf non primitives
# Ref: https://developers.google.com/protocol-buffers/docs/reference/cpp/google.protobuf.descriptor
NON_PRIMITIVE_TYPES = [FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.TYPE_ENUM]

# pylint: disable=too-many-instance-attributes
class GlobalTrainServicer:
    """Something something about the train servicer"""

    def __init__(self, training_service: ServicePackage):
        self._training_service = training_service
        self._model_manager = ModelManager.get_instance()
        caikit_config = get_config()
        self.training_output_dir = caikit_config.runtime.training.output_dir
        self.save_with_id = caikit_config.runtime.training.save_with_id

        # TODO: think about if we really want to do this here:
        self.cdm = get_data_model()

        # Validate that the Caikit Library CDM is compatible with our service descriptor
        validate_data_model(self._training_service.descriptor)
        log.info("<RUN76773777I>", "Validated Caikit Library CDM successfully")

        # TODO: support multiple libs? `caikit_config.libraries` dict
        # Or grab the `libraries` off of the `training_service` instead of config here?
        # Duplicate code in global_train_servicer
        # pylint: disable=duplicate-code
        self.library = clean_lib_names(caikit_config.runtime.library)[0]
        try:
            lib_version = version(self.library)
        except Exception:  # pylint: disable=broad-exception-caught
            lib_version = "unknown"

        log.info(
            "<RUN76884779I>",
            "Constructed train service for library: %s, version: %s",
            self.library,
            lib_version,
        )
        super()

    def Train(
        self,
        request: ProtoMessageType,
        context: ServicerContext,
        *_,
        **__,
    ) -> TrainingJob:
        """Global predict RPC -- Mocks the invocation of a Caikit Library module.train()
        method for a loaded Caikit Library model
        Args:
            request (object): A deserialized RPC request message
            context (ServicerContext): Context object (contains request metadata, etc)
        Returns:
            caikit.interfaces.runtime.data_model.TrainingJob:
                A TrainingJob data model response object
        """
        desc_name = request.DESCRIPTOR.name
        outer_scope_name = "GlobalTrainServicer.Train:%s" % desc_name

        try:
            with alog.ContextLog(log.debug, outer_scope_name):
                module = None
                for mod in caikit.core.registries.module_registry().values():
                    if mod.TASK_CLASS:
                        train_request_for_mod = (
                            ModuleClassTrainRPC.module_class_to_req_name(mod)
                        )
                        if train_request_for_mod == desc_name:
                            module = mod
                            break

                # At this point, if model is still None, we don't know the module this request
                # is for
                if module is None:
                    raise CaikitRuntimeException(
                        StatusCode.INTERNAL,
                        "Global Train not able to parse module for this Train Request",
                    )
                return self.run_training_job(
                    request=request,
                    module=module,
                    training_output_dir=self.training_output_dir,
                    wait=False,
                    context=context,
                ).to_proto()

        except CaikitRuntimeException as e:
            log_dict = {
                "log_code": "<RUN50530380W>",
                "message": e.message,
                "error_id": e.id,
            }
            log.warning({**log_dict, **e.metadata})
            raise e

        # Duplicate code in global_predict_servicer
        # pylint: disable=duplicate-code
        except (TypeError, ValueError) as e:
            log_dict = {
                "log_code": "<RUN490439039W>",
                "message": repr(e),
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            raise CaikitRuntimeException(
                StatusCode.INVALID_ARGUMENT,
                f"Exception raised during training. This may be a problem with your input: {e}",
            ) from e

        except Exception as e:
            log_dict = {
                "log_code": "<RUN49049070W>",
                "message": repr(e),
                "stack_trace": traceback.format_exc(),
            }
            log.warning(log_dict)
            raise CaikitRuntimeException(
                StatusCode.INTERNAL, "Unhandled exception during training"
            ) from e

    def run_training_job(
        self,
        request: ProtoMessageType,
        module: Type[ModuleBase],
        training_output_dir: str,
        *,
        context: Optional[ServicerContext] = None,
        wait: bool = False,
        **kwargs,
    ) -> TrainingJob:
        """Builds the request dict for calling the train function asynchronously,
        then returns the thread id

        Args:
            request (ProtoMessageType): The message that stimulated this request
            module (Type[ModuleBase]): The module class to train
            training_output_dir (str): The base directory where trained models
                should be saved

        Kwargs:
            context (Optional[ServicerContext]): The grpc context for the
                request if called from a grpc handler
            wait (bool): Whether or not to block until the training is complete

        Returns:
            training_job (TrainingJob): The job handle for the training with the
                job's ID and the model's name
        """
        request_data_model = caikit.core.data_model.DataBase.get_class_for_proto(
            request
        ).from_proto(request)

        # Figure out where this model will be saved
        model_path: Union[str, S3Path]
        if request_data_model.output_path:
            # If we got an S3 storage link, just pass that along to the trainer
            model_path: S3Path = request_data_model.output_path
        else:
            # Otherwise, append the model name to the specified output directory
            model_path: str = self._get_model_path(
                training_output_dir, request_data_model.model_name
            )

        # Build the full set of kwargs for the train call
        kwargs.update(
            {
                "module": module,
                "save_path": model_path,
                "save_with_id": self.save_with_id,
                **build_caikit_library_request_dict(request, module.TRAIN_SIGNATURE),
            }
        )

        # Submit the request to the model manager
        model_future = MODEL_MANAGER.train(**kwargs)

        # if requested, wait for training to complete, thus
        # allowing different servicers to cancel the request
        # in case needed. This does make this call synchronous,
        # but that is the intent of this function, since for async request
        # we have separate function below returning futures.
        # TODO: In future, for the case where we want to execute the training
        # in async manner, we would implement a separate "cancel" / "delete"
        # API which would integrate with different training backends
        # as per their interface requirements.
        if wait:

            # Register the cancellation callback if given a context
            if context is not None:

                # Create a callback to register termination of training
                def rpc_termination_callback():
                    """Cancel the model future if it has not yet completed"""
                    if not model_future.get_info().status.is_terminal:
                        log.warning(
                            "<RUN36361257W>", "Canceling training %s", model_future.id
                        )
                        model_future.cancel()

                # NOTE: callback registration needs to be before waiting for the
                #   future, otherwise request will wait before registering
                #   callback.
                callback_registered = context.add_callback(rpc_termination_callback)
                if not callback_registered:
                    log.warning(
                        "<RUN54118242W>",
                        "Failed to register rpc termination callback, aborting rpc",
                    )
                    raise CaikitRuntimeException(
                        StatusCode.ABORTED,
                        "Could not register RPC callback, call has likely terminated.",
                    )

            with alog.ContextTimer(
                log.debug, "Training %s complete in: ", model_future.id
            ):
                model_future.wait()
                training_info = model_future.get_info()
                if training_info.errors:
                    log.info(
                        "Training %s failed with error: %s. "
                        "Re-raising exception for synchronous response",
                        model_future.id,
                        str(training_info.errors[0]),
                    )
                    raise training_info.errors[0]

        # return TrainingJob object
        return TrainingJob(
            model_name=request.model_name,
            training_id=model_future.id,
        )

    def _get_model_path(
        self,
        training_output_dir: Optional[str],
        model_name: str,
    ) -> str:
        """Get the right output path for a given model"""
        base_dir = (
            training_output_dir
            if training_output_dir is not None
            else self.training_output_dir
        )
        return os.path.join(base_dir, model_name)

    def _load_trained_model(self, model_name: str, model_path: str):
        log.debug("Autoloading trained model %s", model_name)
        self._model_manager.load_model(
            model_id=model_name,
            local_model_path=model_path,
            model_type="standalone",
        )
        return self._model_manager.retrieve_model(model_name)
