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

# Standard
from typing import Iterable, Optional, Union

# Third Party
from google.protobuf.message import Message as ProtobufMessage
import grpc

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER, DataObjectBase
from caikit.core.data_model import JobStatus
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.core.model_management.job_base import JobFutureBase
from caikit.interfaces.runtime.data_model import (
    PredictionJobInfoRequest,
    PredictionJobStatusResponse,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.servicer_util import raise_caikit_runtime_exception

log = alog.use_channel("TM-SERVICR-I")


class PredictionJobManagementServicerImpl:
    """This class contains the implementation of all of the RPCs that are required to manage
    a prediction job. This includes fetching status, cancelling jobs, and getting the results.
    """

    #######################
    ## gRPC Service Impl ##
    #######################

    def GetPredictionJobResult(
        self,
        request: PredictionJobInfoRequest,
        context,
        *_,
        **__,
    ) -> Union[
        ProtobufMessage, Iterable[ProtobufMessage]
    ]:  # pylint: disable=unused-argument
        """Get the result of a prediction job by ID"""
        return self.get_prediction_result(request.prediction_id).to_proto()

    def GetPredictionJobStatus(
        self,
        request: PredictionJobInfoRequest,
        context,
        *_,
        **__,
    ):  # pylint: disable=unused-argument
        """Get the status of a prediction job ID"""
        return self.get_prediction_status(request.prediction_id).to_proto()

    def CancelPredictionJob(
        self,
        request: PredictionJobInfoRequest,
        context,
        *_,
        **__,
    ):  # pylint: disable=unused-argument
        """Cancel a prediction job."""
        return self.cancel_prediction(request.prediction_id).to_proto()

    ####################################
    ## Interface-agnostic entrypoints ##
    ####################################

    def get_prediction_result(self, prediction_id: str) -> DataObjectBase:
        """Get the result of a job by ID"""
        model_future: Optional[JobFutureBase] = self._get_prediction_future(
            prediction_id, operation="get_status"
        )
        try:
            model_status = model_future.get_info().status
            if model_status == JobStatus.COMPLETED:
                return model_future.result()

            model_err_message = f"Unable to find {model_future.id} result"
            if model_status == JobStatus.RUNNING:
                model_err_message = f"Prediction {model_future.id} is still in progress"
            if model_status == JobStatus.CANCELED:
                model_err_message = f"Prediction {model_future.id} was cancelled"
            if model_status == JobStatus.ERRORED:
                model_err_message = f"Prediction {model_future.id} encountered an error"
            if model_status == JobStatus.QUEUED:
                model_err_message = f"Prediction {model_future.id} has not started yet"

            raise CaikitCoreException(
                CaikitCoreStatusCode.NOT_FOUND,
                model_err_message,
            )
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to get result for job id {}".format(
                    prediction_id,
                ),
            ) from err

    def get_prediction_status(self, prediction_id: str) -> PredictionJobStatusResponse:
        """Get the status of a job by ID"""
        model_future = self._get_prediction_future(
            prediction_id, operation="get_status"
        )
        try:
            reasons = []
            job_info = model_future.get_info()
            if job_info.errors:
                reasons = [str(error) for error in job_info.errors]

            return PredictionJobStatusResponse(
                prediction_id=prediction_id,
                state=job_info.status,
                reasons=reasons,
                submission_timestamp=job_info.submission_time,
                completion_timestamp=job_info.completion_time,
            )
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to get status for job id {}".format(
                    prediction_id,
                ),
            ) from err

    def cancel_prediction(self, prediction_id: str) -> PredictionJobStatusResponse:
        """Cancel a prediction job."""
        model_future = self._get_prediction_future(prediction_id, operation="cancel")
        try:
            model_future.cancel()
            job_info = model_future.get_info()

            reasons = []
            if job_info.errors:
                reasons = [str(error) for error in job_info.errors]

            return PredictionJobStatusResponse(
                prediction_id=model_future.id,
                state=job_info.status,
                reasons=reasons,
            )
        except CaikitCoreException as err:
            # In the case that we get a `NOT_FOUND`, we assume that the job was canceled.
            # This is to handle stateful predictors that implement `cancel` by fully deleting
            # the prediction.
            if err.status_code == CaikitCoreStatusCode.NOT_FOUND:
                return PredictionJobStatusResponse(
                    inference_id=prediction_id,
                    state=JobStatus.CANCELED,
                )
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Unexpected error trying to cancel job id %s: [%s]",
                prediction_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to cancel job id {}".format(
                    prediction_id,
                ),
            ) from err

    ############################
    ## Implementation Details ##
    ############################

    @staticmethod
    def _get_prediction_future(prediction_id: str, operation: str):
        """Returns a model future, or raises 404 caikit runtime exception on error.
        Wrapped here so that we only catch errors directly in the `predictor.get_prediction_future`
        call
        """
        try:
            return MODEL_MANAGER.get_prediction_future(prediction_id)
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Caught unexpected exception while trying to look up model future for id %s: [%s]",
                prediction_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Unexpected error with job id {}. Could not perform {}".format(
                    prediction_id, operation
                ),
            ) from err
