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
from typing import List, Optional, Type, Union
import abc
import dataclasses
import datetime
import os

# Local
from .model_background_base import ModelBackgroundBase, ModelFutureBase, BackgroundInfo
from ...interfaces.common.data_model.stream_sources import S3Path
from ..data_model import TrainingStatus
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible
from ..toolkit.reversible_hasher import ReversibleHasher


class TrainingInfo(BackgroundInfo):
    pass

class ModelTrainerBase(ModelBackgroundBase):
    __doc__ = __doc__
    ModelFutureBase = ModelFutureBase

    def __init__(
        self,
        trainer_name: str,
        training_id: str,
        **kwargs,
    ):
        if "future_name" not in kwargs:
            kwargs["future_name"]=trainer_name
        if "future_id" not in kwargs:
            kwargs["future_id"]=training_id
        super().__init__(**kwargs)
        
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
        return cls.get_background_name(training_id)
