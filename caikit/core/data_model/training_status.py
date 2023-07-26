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
Common data model enum used for reporting training status
"""

# Standard
from enum import Enum

# Local
from .dataobject import dataobject
from .package import PACKAGE_COMMON


@dataobject(PACKAGE_COMMON)
class TrainingStatus(Enum):
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    CANCELED = 4
    ERRORED = 5

    @property
    def is_terminal(self):
        return self in [
            self.__class__.COMPLETED,
            self.__class__.CANCELED,
            self.__class__.ERRORED,
        ]
