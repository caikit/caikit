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
"""

# Standard
from enum import Enum
from typing import Optional, Type
import abc

# Local
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible


class TrainerBase(FactoryConstructible):
    __doc__ = __doc__

    class TrainingStatus(Enum):
        """A given training job must be in exactly one of these states at all
        times
        """

        QUEUED = 1
        RUNNING = 2
        COMPLETED = 3
        CANCELED = 4
        ERRORED = 5

    class ModelFutureBase(abc.ABC):
        """Every Trainer must implement a ModelFuture class that can access the
        training job in the infrastructure managed by the trainer.
        """

        @property
        @abc.abstractmethod
        def id(self) -> str:
            """Every model future must have a unique ID that can be used to look
            up the in-flight training
            """

        @property
        @abc.abstractmethod
        def save_path(self) -> Optional[str]:
            """If created with a save path, the future must expose it"""

        @abc.abstractmethod
        def get_status(self) -> "TrainingStatus":
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

    @abc.abstractmethod
    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> "ModelFutureBase":
        """Start training the given module and return a future to the trained
        model instance
        """
