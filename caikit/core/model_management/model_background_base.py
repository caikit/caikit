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
A Base class for background model operations. This class is 
used for background training and inferences

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
from ...interfaces.common.data_model.stream_sources import S3Path
from ..data_model import BackgroundStatus
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible
from ..toolkit.reversible_hasher import ReversibleHasher


@dataclasses.dataclass
class BackgroundInfo:
    status: BackgroundStatus
    errors: Optional[List[Exception]] = None
    submission_time: Optional[datetime.datetime] = None
    completion_time: Optional[datetime.datetime] = None


class ModelFutureBase(abc.ABC):
    """Every Background implementation must have a ModelFuture class that can access the
    job information in the infrastructure managed by the task.
    """

    ID_DELIMITER = ":"

    def __init__(
        self,
        future_name: str,
        future_id: str,
        save_with_id: bool,
        save_path: Optional[Union[str, S3Path]],
        model_name: Optional[str] = None,
        use_reversible_hash: bool = True,
        **kwargs,
    ):
        # Trainers should deal with an S3 ref first and not pass it along here
        if save_path and isinstance(save_path, S3Path):
            raise ValueError("S3 output path not supported by this runtime")
        self._id = (
            self.__class__.ID_DELIMITER.join(
                [ReversibleHasher.hash(future_name), future_id]
            )
            if use_reversible_hash
            else future_id
        )

    @property
    def id(self) -> str:
        """Every model future must have a unique ID that can be used to look
        up the in-flight background task
        """
        return self._id

    @property
    def save_path(self) -> Optional[str]:
        """If created with a save path, the future must expose it, including
        any injected background id
        """
        return self._save_path

    @abc.abstractmethod
    def get_info(self) -> BackgroundInfo:
        """Every model future must be able to poll the status of the
        background job
        """

    @abc.abstractmethod
    def cancel(self):
        """Terminate the given training"""

    @abc.abstractmethod
    def wait(self):
        """Block until the job reaches a terminal state"""

    @abc.abstractmethod
    def load(self) -> ModuleBase:
        """A model future must be loadable with no additional arguments. Mainly
        useful in train results"""

    @abc.abstractmethod
    def result(self):
        """Support result() to match concurrent.futures.Future"""


class ModelBackgroundBase(FactoryConstructible):
    @abc.abstractmethod
    def get_model_future(self, training_id: str) -> "ModelFutureBase":
        """Look up the model future for the given id"""

    ## Shared Utilities ##

    @classmethod
    def get_background_name(cls, background_id: str) -> str:
        """Un-hash the background's instance name from the given training id"""
        return ReversibleHasher.reverse_hash(
            background_id.split(ModelFutureBase.ID_DELIMITER)[0]
        )
