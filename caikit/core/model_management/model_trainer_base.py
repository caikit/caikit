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
A Trainer is responsible for managing execution of a training job for a given
module class

Configuration for ModelTrainers lives under the config as follows:

model_management:
    trainers:
        <trainer name>:
            type: <trainer type name>
            config:
                <config option>: <value>
"""
# Standard
from typing import Optional, Type, Union
import abc

# Local
from ...interfaces.common.data_model.stream_sources import S3Path
from ..modules import ModuleBase
from .job_base import JobBase, JobFutureBase, JobInfo


class TrainingInfo(JobInfo):
    pass


class ModelTrainerFutureBase(JobFutureBase):
    @abc.abstractmethod
    def load(self) -> ModuleBase:
        """A model future must be loadable with no additional arguments. Mainly
        useful in train results"""


class ModelTrainerBase(JobBase):
    __doc__ = __doc__

    ModelFutureBase = ModelTrainerFutureBase

    @abc.abstractmethod
    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[Union[str, S3Path]] = None,
        save_with_id: bool = False,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ModelFutureBase:
        """Start training the given module and return a future to the trained
        model instance
        """

    ## Shared Utilities ##

    @classmethod
    def get_trainer_name(cls, training_id: str) -> str:
        """Un-hash the trainer's instance name from the given training id"""
        return cls.get_job_name(training_id)
