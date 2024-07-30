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
from typing import Iterable, Union

# Third Party
from google.protobuf.message import Message as ProtobufMessage
import grpc

# First Party
import alog

# Local
from caikit.core import MODEL_MANAGER, DataObjectBase
from caikit.core.data_model import JobStatus, JobType
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
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
        return self.get_job_result(request.job_id).to_proto()

    def GetPredictionJobStatus(
        self,
        request: PredictionJobInfoRequest,
        context,
        *_,
        **__,
    ):  # pylint: disable=unused-argument
        """Get the status of a prediction job ID"""
        return self.get_job_status(request.job_id).to_proto()

    def CancelPredictionJob(
        self,
        request: PredictionJobInfoRequest,
        context,
        *_,
        **__,
    ):  # pylint: disable=unused-argument
        """Cancel a prediction job."""
        return self.cancel_job(request.job_id).to_proto()

    ####################################
    ## Interface-agnostic entrypoints ##
    ####################################

    def get_job_result(self, job_id: str) -> DataObjectBase:
        """Get the result of a job by ID"""
        model_future = self._get_model_future(job_id, operation="get_status")
        try:
            return model_future.result()
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to get result for job id {}".format(
                    job_id,
                ),
            ) from err

    def get_job_status(self, job_id: str) -> PredictionJobStatusResponse:
        """Get the status of a job by ID"""
        model_future = self._get_model_future(job_id, operation="get_status")
        try:
            reasons = []
            job_info = model_future.get_info()
            if job_info.errors:
                reasons = [str(error) for error in job_info.errors]

            return PredictionJobStatusResponse(
                job_id=job_id,
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
                    job_id,
                ),
            ) from err

    def cancel_job(self, job_id: str) -> PredictionJobStatusResponse:
        """Cancel a prediction job."""
        model_future = self._get_model_future(job_id, operation="cancel")
        try:
            model_future.cancel()
            job_info = model_future.get_info()

            reasons = []
            if job_info.errors:
                reasons = [str(error) for error in job_info.errors]

            return PredictionJobStatusResponse(
                job_id=model_future.id,
                state=job_info.status,
                reasons=reasons,
            )
        except CaikitCoreException as err:
            # In the case that we get a `NOT_FOUND`, we assume that the job was canceled.
            # This is to handle stateful trainers that implement `cancel` by fully deleting
            # the training.
            if err.status_code == CaikitCoreStatusCode.NOT_FOUND:
                return PredictionJobStatusResponse(
                    inference_id=job_id,
                    state=JobStatus.CANCELED,
                )
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Unexpected error trying to cancel job id %s: [%s]",
                job_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Failed to cancel job id {}".format(
                    job_id,
                ),
            ) from err

    ############################
    ## Implementation Details ##
    ############################

    @staticmethod
    def _get_model_future(job_id: str, operation: str):
        """Returns a model future, or raises 404 caikit runtime exception on error.
        Wrapped here so that we only catch errors directly in the `predictor.get_model_future` call
        """
        try:
            return MODEL_MANAGER.get_model_future(
                job_id, future_type=JobType.PREDICTION
            )
        except CaikitCoreException as err:
            raise_caikit_runtime_exception(exception=err)
        except Exception as err:
            log.debug2(
                "Caught unexpected exception while trying to look up model future for id %s: [%s]",
                job_id,
                err,
            )
            raise CaikitRuntimeException(
                grpc.StatusCode.INTERNAL,
                "Unexpected error with job id {}. Could not perform {}".format(
                    job_id, operation
                ),
            ) from err
