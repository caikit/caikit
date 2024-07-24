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

# Have pylint ignore Class XXXX has no YYYY member so that we can use gRPC enums.
# pylint: disable=E1101

# Third Party
import grpc

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER, DataObjectBase
from caikit.core.data_model import BackgroundInferenceStatus
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.interfaces.runtime.data_model import BackgroundInferenceStatusResponse, BackgroundInferenceInfoRequest
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.servicer_util import raise_caikit_runtime_exception

log = alog.use_channel("TM-SERVICR-I")


class InferenceManagementServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to run a
    service in Model Mesh as a Model-Runtime."""

    #######################
    ## gRPC Service Impl ##
    #######################

    def GetBackgroundInferenceResult(self, request:BackgroundInferenceInfoRequest , context):  # pylint: disable=unused-argument
        """Get the status of a training by ID"""
        return self.get_inference_result(request.inference_id).to_proto()

    def GetBackgroundInferenceStatus(self, request: BackgroundInferenceInfoRequest, context):  # pylint: disable=unused-argument
        """Get the status of a training by ID"""
        return self.get_inference_status(request.inference_id).to_proto()

    def CancelBackgroundInference(self, request: BackgroundInferenceInfoRequest, context):  # pylint: disable=unused-argument
        """Cancel a training future."""
        return self.cancel_inference(request.inference_id).to_proto()

    ####################################
    ## Interface-agnostic entrypoints ##
    ####################################

    def get_inference_result(self, inference_id: str) -> DataObjectBase:
        """Get the status of a training by ID"""
        model_future = self._get_model_future(inference_id, operation="get_status")
        try:
            return model_future.result()
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to get result for inference id {}".format(
                    inference_id,
                ),
            ) from err
            
    def get_inference_status(self, inference_id: str) -> BackgroundInferenceStatusResponse:
        """Get the status of a training by ID"""
        model_future = self._get_model_future(inference_id, operation="get_status")
        try:
            reasons = []
            training_info = model_future.get_info()
            if training_info.errors:
                reasons = [str(error) for error in training_info.errors]

            return BackgroundInferenceStatusResponse(
                inference_id=inference_id,
                state=training_info.status,
                reasons=reasons,
                submission_timestamp=training_info.submission_time,
                completion_timestamp=training_info.completion_time,
            )
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to get status for inference id {}".format(
                    inference_id,
                ),
            ) from err

    def cancel_inference(self, inference_id: str) -> BackgroundInferenceStatusResponse:
        """Cancel a training future."""
        model_future = self._get_model_future(inference_id, operation="cancel")
        try:
            model_future.cancel()
            training_info = model_future.get_info()

            reasons = []
            if training_info.errors:
                reasons = [str(error) for error in training_info.errors]

            return BackgroundInferenceStatusResponse(
                inference_id=model_future.id,
                state=training_info.status,
                reasons=reasons,
            )
        except CaikitCoreException as err:
            # In the case that we get a `NOT_FOUND`, we assume that the training was canceled.
            # This is to handle stateful trainers that implement `cancel` by fully deleting
            # the training. NB: Future `GetTrainingStatus` calls for these canceled trainings
            # would raise a not found error to the user.
            if err.status_code == CaikitCoreStatusCode.NOT_FOUND:
                return BackgroundInferenceStatusResponse(
                    inference_id=inference_id,
                    state=BackgroundInferenceStatus.CANCELED,
                )
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Unexpected error trying to cancel inference id %s: [%s]",
                inference_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to cancel inference id {}".format(
                    inference_id,
                ),
            ) from err

    ############################
    ## Implementation Details ##
    ############################

    @staticmethod
    def _get_model_future(inference_id: str, operation: str):
        """Returns a model future, or raises 404 caikit runtime exception on error.
        Wrapped here so that we only catch errors directly in the `trainer.get_model_future` call
        """
        try:
            return MODEL_MANAGER.get_model_future(inference_id, future_type="background_inference")
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Caught unexpected exception while trying to look up model future for id %s: [%s]",
                inference_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Unexpected error with inference id {}. Could not perform {}".format(
                    inference_id, operation
                ),
            ) from err
