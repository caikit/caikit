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
"""Base class with common functionality across all caikit servers"""

# Standard
from typing import Optional
import abc

# First Party
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.runtime.model_management.model_manager import ModelManager

log = alog.use_channel("SERVR-BASE")


class RuntimeServerBase(abc.ABC):
    __doc__ = __doc__

    def __init__(self, base_port: int, tls_config_override: Optional[aconfig.Config]):
        self.config = get_config()
        self.port = base_port
        self.tls_config = (
            tls_config_override if tls_config_override else self.config.runtime.tls
        )
        log.debug4("Full caikit config: %s", get_config())

    def _shut_down_model_manager(self):
        """Shared utility for shutting down the model manager"""
        ModelManager.get_instance().shut_down()

    @abc.abstractmethod
    def start(self, blocking: bool = True):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    # Context manager impl
    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, type_, value, traceback):
        self.stop()
