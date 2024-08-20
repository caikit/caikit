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
used for background training and predictions
"""
# Standard
from typing import Any, List, Optional
import abc
import dataclasses
import datetime

# Local
from ..data_model import JobStatus
from ..toolkit.factory import FactoryConstructible
from ..toolkit.reversible_hasher import ReversibleHasher


@dataclasses.dataclass
class JobInfo:
    status: JobStatus
    errors: Optional[List[Exception]] = None
    submission_time: Optional[datetime.datetime] = None
    completion_time: Optional[datetime.datetime] = None


class JobFutureBase(abc.ABC):
    """Every JobBase implementation must have a JobFutureBase class that can access the
    job information in the infrastructure managed by the task.
    """

    ID_DELIMITER = ":"

    def __init__(
        self,
        future_name: str,
        future_id: str,
        use_reversible_hash: bool = True,
        **kwargs,
    ):
        self._id = (
            self.__class__.ID_DELIMITER.join(
                [ReversibleHasher.hash(future_name), future_id]
            )
            if use_reversible_hash
            else future_id
        )

    @property
    def id(self) -> str:
        """Every job future must have a unique ID that can be used to look
        up the in-flight background task
        """
        return self._id

    @abc.abstractmethod
    def get_info(self) -> JobInfo:
        """Every model future must be able to poll the status of the
        background job
        """

    @abc.abstractmethod
    def cancel(self):
        """Terminate the given job"""

    @abc.abstractmethod
    def wait(self):
        """Block until the job reaches a terminal state"""

    @abc.abstractmethod
    def result(self) -> Any:
        """Support result() to match concurrent.futures.Future"""


class JobBase(FactoryConstructible):
    @abc.abstractmethod
    def get_future(self, job_id: str) -> JobFutureBase:
        """Look up the model future for the given id"""

    ## Shared Utilities ##

    @classmethod
    def get_job_name(cls, job_id: str) -> str:
        """Un-hash the background's instance name from the given job id"""
        return ReversibleHasher.reverse_hash(
            job_id.split(JobFutureBase.ID_DELIMITER)[0]
        )
