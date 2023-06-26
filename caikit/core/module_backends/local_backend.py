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

"""This module implements a LOCAL backend configuration
"""

# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from ..toolkit.wip_decorator import TempDisableWIP
from .backend_types import register_backend_type
from .base import BackendBase

log = alog.use_channel("LCLBKND")
error = error_handler.get(log)


class LocalBackend(BackendBase):
    backend_type = "LOCAL"

    def register_config(self, config) -> None:
        """Function to merge configs with existing configurations"""
        error(
            "<COR86557945E>",
            AssertionError(
                f"{self.backend_type} backend does not support this operation"
            ),
        )

    def start(self):
        """Start local backend. This is a no-op function"""
        self._started = True

    def stop(self):
        """Stop local backend. This is a no-op"""
        self._started = False


# Register local backend
with TempDisableWIP():
    register_backend_type(LocalBackend)
