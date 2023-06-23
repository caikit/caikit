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

# Standard
from typing import Optional
import copy

# Third Party
import pytest

# Local
from caikit.core import MODEL_MANAGER

# Add mock backend
# This is set in the base test config's load_priority list
from caikit.core.model_management import ModelLoaderBase, model_loader_factory
from caikit.core.model_management.local_model_loader import LocalModelLoader
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import ModuleBase, ModuleConfig
from caikit.core.registries import (
    module_backend_classes,
    module_backend_registry,
    module_backend_types,
    module_registry,
)


class MockBackend(BackendBase):
    backend_type = "MOCK"

    def __init__(self, config=...) -> None:
        super().__init__(config)
        self._started = False

    def start(self):
        self._started = True

    def register_config(self, config):
        self.config = {**config, **self.config}

    def stop(self):
        self._started = False


backend_types.register_backend_type(MockBackend)


# Add a new shared load backend that tests can use
class TestLoader(ModelLoaderBase):
    name = "TESTLOADER"
    __test__ = False

    def __init__(self, config):
        self.config = config
        self.loaded_models = []
        self.local_loader = model_loader_factory.construct({"type": "LOCAL"})

    def load(self, model_config: ModuleConfig, *args, **kwargs) -> Optional[ModuleBase]:
        # allow config.model_type to control whether this loader barfs
        if "model_type" in self.config and "model_type" in kwargs:
            if self.config["model_type"] != kwargs["model_type"]:
                # Don't load in this loader
                return None
        # use the "Local" loader to actually load the model
        model = self.local_loader.load(model_config)
        self.loaded_models.append(model)
        return model


model_loader_factory.register(TestLoader)


def configured_backends():
    local_loaders = [
        loader
        for loader in MODEL_MANAGER._get_loaders()
        if isinstance(loader, LocalModelLoader)
    ]
    return [backend for loader in local_loaders for backend in loader._backends]


@pytest.fixture
def reset_backend_types():
    """Fixture that will reset the backend types if a test modifies them"""
    base_backend_types = {key: val for key, val in module_backend_types().items()}
    base_backend_classes = {key: val for key, val in module_backend_classes().items()}
    yield
    module_backend_types().clear()
    module_backend_types().update(base_backend_types)
    module_backend_classes().clear()
    module_backend_classes().update(base_backend_classes)


@pytest.fixture
def reset_module_backend_registry():
    """Fixture that will reset the module distribution registry if a test modifies them"""
    orig_module_backend_registry = {
        key: val for key, val in module_backend_registry().items()
    }
    yield
    module_backend_registry().clear()
    module_backend_registry().update(orig_module_backend_registry)


@pytest.fixture
def reset_module_registry():
    """Fixture that will reset caikit.core module registry if a test modifies it"""
    orig_module_registry = {key: val for key, val in module_registry().items()}
    yield
    module_registry().clear()
    module_registry().update(orig_module_registry)


@pytest.fixture
def reset_model_manager():
    prev_finders = MODEL_MANAGER._finders
    prev_loaders = MODEL_MANAGER._loaders
    MODEL_MANAGER._finders = None
    MODEL_MANAGER._loaders = None
    yield
    MODEL_MANAGER._finders = prev_finders
    MODEL_MANAGER._loaders = prev_loaders


@pytest.fixture
def reset_globals(
    reset_backend_types,
    reset_model_manager,
    reset_module_backend_registry,
    reset_module_registry,
):
    """Fixture that will reset the backend types and module registries if a test modifies them"""
