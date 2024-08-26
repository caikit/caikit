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
A Job Predictor is responsible for managing execution of a prediction jobs running
in the background for a given task method

Configuration for Job Predictors lives under the config as follows:

model_management:
    job_predictors:
        <predictor name>:
            type: <predictor type name>
            config:
                <config option>: <value>
"""
# Standard
from typing import Optional
import abc

# Local
from ..data_model import DataObjectBase
from ..modules import ModuleBase
from .job_base import JobBase, JobFutureBase, JobInfo


class JobPredictorInfo(JobInfo):
    """JobPredictorInfo is a remap of JobInfo but for predictors"""


class JobPredictorFutureBase(JobFutureBase):
    """Subclass of JobFutureBase for Job Predictions"""

    @abc.abstractmethod
    def result(self) -> DataObjectBase:
        """The result of a JobPredictorFutureBase is the result object"""


class JobPredictorBase(JobBase):
    __doc__ = __doc__

    @abc.abstractmethod
    def predict(
        self,
        model_instance: ModuleBase,
        prediction_func_name: str,
        *args,
        external_inference_id: Optional[str] = None,
        **kwargs,
    ) -> JobPredictorFutureBase:
        """Start a prediction with the given model instance and function and return a
        future to the prediction result
        """

    @abc.abstractmethod
    def get_prediction_future(self, future_id: str) -> JobPredictorFutureBase:
        """Look up the jobs future for the given id"""

    ## Shared Utilities ##

    def get_future(self, prediction_id: str) -> JobFutureBase:
        """Look up the job future for the given id"""
        return self.get_prediction_future(prediction_id)

    @classmethod
    def get_predictor_name(cls, predict_id: str) -> str:
        """Un-hash the predictors's instance name from the given prediction id"""
        return cls.get_job_name(predict_id)
