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
Caikit Core Exception enum used for reporting Exception status raised in caikit core
"""

# Standard
from enum import Enum
import uuid


class CaikitCoreStatusCode(Enum):
    NOT_FOUND = 1
    INVALID_ARGUMENT = 2
    CONNECTION_ERROR = 3
    UNAUTHORIZED = 4
    FORBIDDEN = 5
    UNKNOWN = 6
    FATAL = 7


class CaikitCoreException(Exception):
    status_code: CaikitCoreStatusCode
    message: str

    def __init__(self, status_code: CaikitCoreStatusCode, message: str) -> None:
        self.status_code = status_code
        self.message = message
        self.id = uuid.uuid4().hex
