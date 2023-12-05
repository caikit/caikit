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
Common interface for destroyable threads and processes
"""

# Standard
from typing import Any, Optional
import abc


class Destroyable(abc.ABC):
    __doc__ = __doc__

    @property
    @abc.abstractmethod
    def destroyed(self) -> bool:
        """Return True if destroy was called, regardless of whether the
        destroyable was alive at the time
        """

    @property
    @abc.abstractmethod
    def canceled(self) -> bool:
        """Returns True if destroyed while actively working"""

    @property
    @abc.abstractmethod
    def ran(self) -> bool:
        """Return True if the destroyable completed execution in any state"""

    @property
    @abc.abstractmethod
    def threw(self) -> bool:
        """Return True if any exception was raised during execution"""

    @abc.abstractmethod
    def get_or_throw(self) -> Any:
        """Get the result of the execution or raise an error if one occurred"""

    @abc.abstractmethod
    def destroy(self):
        """Cancel any in-progress work"""

    @property
    @abc.abstractmethod
    def error(self) -> Optional[Exception]:
        """Return the error information to user if one occurred"""
