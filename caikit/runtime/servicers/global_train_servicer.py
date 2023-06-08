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
from typing import Optional
from uuid import uuid4
import concurrent.futures
import multiprocessing
import os
import threading
import traceback

# Third Party
from google.protobuf.descriptor import FieldDescriptor
from grpc import ServicerContext, StatusCode

# First Party
import alog

# Local
from caikit import get_config
from caikit.interfaces.runtime.data_model import TrainingJob
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.model_management.training_manager import TrainingManager
from caikit.runtime.service_factory import ServicePackage
from caikit.runtime.service_generation.rpcs import ModuleClassTrainRPC
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import clean_lib_names, get_data_model
from caikit.runtime.utils.servicer_util import (
    build_caikit_library_request_dict,
    validate_data_model,
)
from caikit.runtime.work_management.train_executors import (
    LocalTrainSaveExecutor,
    SubProcessTrainSaveExecutor,
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
        self.training_manager = TrainingManager.get_instance()
        # NOTE: we are using ThreadPoolExecutor for simplicity of the
        # the API with an intent to handle the training job
        # in an async fashion with "Futures".
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # store the map of model ids to job ids
        self.training_map = self.training_manager.training_futures
        caikit_config = get_config()
        self.training_output_dir = caikit_config.runtime.training.output_dir
        self.auto_load_trained_model = (
            caikit_config.runtime.training.auto_load_trained_model
        )

        self.use_subprocess = caikit_config.runtime.training.use_subprocess

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

    def Train(self, request, context, *_, **__) -> TrainingJob:
        """Global predict RPC -- Mocks the invocation of a Caikit Library module.train()
        method for a loaded Caikit Library model
        Args:
            request(object):
                A deserialized RPC request message
            context(grpc.ServicerContext): Context object (contains request metadata, etc)
        Returns:
            caikit.interfaces.runtime.data_model.TrainingJob:
                A TrainingJob data model response object
        """
        desc_name = request.DESCRIPTOR.name
        outer_scope_name = "GlobalTrainServicer.Train:%s" % desc_name

        try:
            with alog.ContextLog(log.debug, outer_scope_name):
                for mod in caikit.core.registries.module_registry().values():
                    if mod.TASK_CLASS:
                        train_request_for_mod = (
                            ModuleClassTrainRPC.module_class_to_req_name(mod)
                        )
                        if train_request_for_mod == desc_name:
                            model = mod
                            break

                # At this point, if model is still None, we don't know the module this request
                # is for
                if model is None:
                    raise CaikitRuntimeException(
                        StatusCode.INTERNAL,
                        "Global Train not able to parse module for this Train Request",
                    )
                # generate a unique training id
                training_id = str(uuid4())
                return self.run_training_job(
                    request=request,
                    model=model,
                    training_id=training_id,
                    training_output_dir=self.training_output_dir,
                    context=context,
                )

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
                f"Exception raised during inference. This may be a problem with your input: {e}",
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
        request,
        model,
        training_id,
        training_output_dir,
        context: ServicerContext,
        wait=False,
    ) -> TrainingJob:
        """Builds the request dict for calling the train function asynchronously,
        then returns the thread id"""

        # Figure out where this model will be saved
        model_name = request.model_name
        model_path = self._get_model_path(training_output_dir, model_name, training_id)

        # Build the full set of kwargs for the train and save call
        kwargs = {
            "module_class": model,
            "model_path": model_path,
            **build_caikit_library_request_dict(request, model.TRAIN_SIGNATURE),
        }

        # If running with a subprocess, set the target, events and args accordingly
        if self.use_subprocess:
            cancel_event = multiprocessing.Event()
            target = SubProcessTrainSaveExecutor(cancel_event)
        else:
            cancel_event = threading.Event()
            target = LocalTrainSaveExecutor(cancel_event)

        log.debug2(
            "Training with %s",
            "SUBPROCESS" if self.use_subprocess else "MAIN PROCESS",
        )

        # start training asynchronously
        thread_future = self.run_async(
            runnable_executor=target,
            kwargs=kwargs,
            model_name=model_name,
            model_path=model_path,
        )

        self.training_map[training_id] = thread_future

        # Create a callback to register termination of training
        def rpc_termination_callback():
            """Function to be called when the RPC is terminated.
            This can happen when the training is completed or
            when we receive a cancellation request.
            """
            for event in target.events:
                if not event.is_set():
                    event.set()

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
            # NOTE: callback registration needs to be before
            # waiting for the future, otherwise request will wait before registering
            # callback
            # Add callback for termination of request
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

            with alog.ContextTimer(log.debug, "Training %s complete in: ", training_id):
                thread_future.result()

        # return TrainingJob object
        return TrainingJob(
            model_name=request.model_name, training_id=training_id
        ).to_proto()

    def run_async(
        self,
        runnable_executor,
        kwargs,
        model_name,
        model_path,
    ) -> concurrent.futures.Future:
        """Runs the train function in a thread and saves the trained model in a callback"""

        if self.auto_load_trained_model:

            def target(*args, **kwargs):
                runnable_executor.train_and_save_model(*args, **kwargs)
                return self._load_trained_model(model_name, model_path)

        else:
            target = runnable_executor.train_and_save_model

        return self.executor.submit(target, **kwargs)

    def _get_model_path(
        self,
        training_output_dir: Optional[str],
        model_name: str,
        training_id: str,
    ) -> str:
        """Get the right output path for a given model"""

        # make sure we create the right path for saving the trained model
        # The path depends on training_output_dir. If it's provided, use it
        # otherwise use the default
        if training_output_dir is not None:
            if training_id in training_output_dir:
                model_path = os.path.join(training_output_dir, model_name)
            else:  # create a subdir with training_id
                model_path = os.path.join(training_output_dir, training_id, model_name)
        else:
            model_path = os.path.join(self.training_output_dir, training_id, model_name)
        return model_path

    def _load_trained_model(self, model_name: str, model_path: str):
        log.debug("Autoloading trained model %s", model_name)
        self._model_manager.load_model(
            model_id=model_name,
            local_model_path=model_path,
            model_type="standalone",
        )
        return self._model_manager.retrieve_model(model_name)
