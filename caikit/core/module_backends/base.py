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

"""Base class to initialize backend
"""

# Standard
from contextlib import contextmanager
from threading import Lock
from typing import Optional
import abc

# First Party
import aconfig


class BackendBase(abc.ABC):
    """Interface for creating configuration setup for backends"""

    def __init__(self, config: Optional[aconfig.Config] = None) -> None:
        self.config = config or {}
        self._started = False
        self._start_lock = Lock()

    @property
    @classmethod
    @abc.abstractmethod
    def backend_type(cls):
        """Property storing type of the backend"""

    @property
    def is_started(self):
        return self._started

    @abc.abstractmethod
    def register_config(self, config):
        """Function to allow dynamic merging of configs.
        This can be useful, if there are explicit configurations
        particular implementations (modules) need to register before the starting the backend.
        """
        # NOTE: This function should be implemented in such a way that it can be called multiple
        # times

    @abc.abstractmethod
    def start(self):
        """Function to start a distributed backend. This function
        should set self._started variable to True"""

    @abc.abstractmethod
    def stop(self):
        """Function to stop a distributed backend. This function
        should set self._started variable to False"""

    @contextmanager
    def start_lock(self):
        with self._start_lock:
            yield
