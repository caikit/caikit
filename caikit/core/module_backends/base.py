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
from typing import Optional, Type
import abc

# First Party
import aconfig

# Local
from ..modules import ModuleBase


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


class SharedTrainBackendBase(BackendBase, abc.ABC):
    """Interface for a backend that can perform train on any given module

    A Shared backend is one that treats the given module as a black box and
    delegates the execution of that module's functionality to an alternate
    execution engine.
    """

    @abc.abstractmethod
    def train(self, module_class: Type[ModuleBase], *args, **kwargs) -> ModuleBase:
        """Perform the given module's train operation and return the trained
        module instance.

        TODO: The return type here might be problematic in the case where the
            server performing the train operation is just a proxy for both train
            and inference. Consider some kind of lazy load proxy that would not
            require the model to be held in memory.

        Args:
            module_class (Type[ModuleBase]): The module class to train
            *args, **kwargs: The args to pass through to training

        Returns:
            model (ModuleBase): The in-memory instance of the trained
                module
        """


class SharedLoadBackendBase(BackendBase, abc.ABC):
    """Interface for a backend that can perform load/unload on any given model

    A Shared backend is one that treats the given module as a black box and
    delegates the execution of that module's functionality to an alternate
    execution engine.

    The module returned by a universal manager must be capable of having run
    called locally and delegating the execution of the underlying module to the
    backend's framework.
    """

    @abc.abstractmethod
    def load(self, model_path: str, *args, **kwargs) -> Optional[ModuleBase]:
        """Load the model stored at the given path into the backend

        This function is responsible for loading a model in a way that the
        backend is then able to execute it.

        Shared loaders will be configured in a priority sequence. If a higher
        priority loader fails to load a given model, the next one is attempted
        until the model is loaded or no loaders are left.

        Args:
            model_path (str): Path to directory or zip file holding the model
                with the config.yml and any artifacts
            *args, **kwargs: Additional args to pass through to the module's
                load function

        Returns:
            model (Optional[ModuleBase]): A runnable model if one could be
                loaded, otherwise None. Such a model may be a wrapper that
                delegates execution to the concrete model loaded elsewhere.
        """
