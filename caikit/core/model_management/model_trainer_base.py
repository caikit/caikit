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
from typing import List, Optional, Type
import abc
import dataclasses
import datetime

# Local
from ..data_model import TrainingStatus
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible
from ..toolkit.reversible_hasher import ReversibleHasher
from .model_saver import ModelSaver


@dataclasses.dataclass
class TrainingInfo:
    status: TrainingStatus
    errors: Optional[List[Exception]] = None
    submission_time: Optional[datetime.datetime] = None
    completion_time: Optional[datetime.datetime] = None


class ModelTrainerBase(FactoryConstructible):
    __doc__ = __doc__

    class ModelFutureBase(abc.ABC):
        """Every Trainer must implement a ModelFuture class that can access the
        training job in the infrastructure managed by the trainer.
        """

        ID_DELIMITER = ":"

        def __init__(
            self,
            trainer_name: str,
            training_id: str,
            saver: Optional[ModelSaver] = None,
            model_name: Optional[str] = None,
            use_reversible_hash: bool = True,
        ):
            self._id = (
                self.__class__.ID_DELIMITER.join(
                    [ReversibleHasher.hash(trainer_name), training_id]
                )
                if use_reversible_hash
                else training_id
            )
            self._saver = saver
            self._model_name = model_name

        @property
        def id(self) -> str:
            """Every model future must have a unique ID that can be used to look
            up the in-flight training
            """
            return self._id

        @property
        def name(self) -> str:
            """The user-provided name of the model being trained"""
            return self._model_name

        @property
        def saver(self) -> ModelSaver:
            """Trainers with remote execution must have a ModelSaver that can store
            the trained model somewhere
            """
            return self._saver

        @abc.abstractmethod
        def get_info(self) -> TrainingInfo:
            """Every model future must be able to poll the status of the
            training job
            """

        @abc.abstractmethod
        def cancel(self):
            """Terminate the given training"""

        @abc.abstractmethod
        def wait(self):
            """Block until the job reaches a terminal state"""

        @abc.abstractmethod
        def load(self) -> ModuleBase:
            """A model future must be loadable with no additional arguments"""

        ## Common Impl ##

        def result(self) -> ModuleBase:
            """Support result() to match concurrent.futures.Future"""
            return self.load()

    @abc.abstractmethod
    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        saver: Optional[ModelSaver],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> "ModelFutureBase":
        """Start training the given module and return a future to the trained
        model instance
        """

    @abc.abstractmethod
    def get_model_future(self, training_id: str) -> "ModelFutureBase":
        """Look up the model future for the given id"""

    ## Shared Utilities ##

    @classmethod
    def get_trainer_name(cls, training_id: str) -> str:
        """Un-hash the trainer's instance name from the given training id"""
        return ReversibleHasher.reverse_hash(
            training_id.split(cls.ModelFutureBase.ID_DELIMITER)[0]
        )
