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
import os

# Local
from ...interfaces.common.data_model.stream_sources import S3Path
from ..data_model import TrainingStatus
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible
from ..toolkit.reversible_hasher import ReversibleHasher


@dataclasses.dataclass
class TrainingInfo:
    status: TrainingStatus
    errors: Optional[List[Exception]] = None


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
            save_with_id: bool,
            save_path: Optional[Union[str, S3Path]],
        ):
            # Trainers should deal with an S3 ref first and not pass it along here
            if save_path and isinstance(save_path, S3Path):
                raise ValueError("S3 output path not supported by this runtime")
            self._id = self.__class__.ID_DELIMITER.join(
                [ReversibleHasher.hash(trainer_name), training_id]
            )
            self._save_path = self.__class__._save_path_with_id(
                save_path,
                save_with_id,
                self._id,
            )

        @property
        def id(self) -> str:
            """Every model future must have a unique ID that can be used to look
            up the in-flight training
            """
            return self._id

        @property
        def save_path(self) -> Optional[str]:
            """If created with a save path, the future must expose it, including
            any injected training id
            """
            return self._save_path

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

        @classmethod
        def _save_path_with_id(
            cls,
            save_path: Optional[str],
            save_with_id: bool,
            training_id: str,
        ) -> Optional[str]:
            """If asked to save_with_id, child classes should use this shared
            utility to construct the final save path
            """
            if save_path is None:
                return save_path
            if not save_with_id or training_id in save_path:
                return save_path

            # If told to save with the ID in the path, inject it right before the
            # final portion of the path which is assumed to be the model ID.
            path_parts = os.path.split(save_path)
            return os.path.join(
                *list(path_parts[:-1] + (training_id,) + path_parts[-1:])
            )

    @abc.abstractmethod
    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[Union[str, S3Path]] = None,
        save_with_id: bool = False,
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
