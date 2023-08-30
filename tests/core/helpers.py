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
from typing import Optional, Type
import os
import uuid

# Local
from caikit.core import MODEL_MANAGER
from caikit.core.data_model import TrainingStatus
from caikit.core.model_management import (
    ModelFinderBase,
    ModelInitializerBase,
    ModelTrainerBase,
    TrainingInfo,
    model_finder_factory,
    model_initializer_factory,
    model_trainer_factory,
)
from caikit.core.model_management.local_model_initializer import LocalModelInitializer
from caikit.core.module_backends import BackendBase, backend_types
from caikit.core.modules import ModuleBase, ModuleConfig
from sample_lib.modules import SampleModule


# Add mock backend
# This is set in the base test config's load_priority list
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

# Add a new model finder that tests can use
class TestFinder(ModelFinderBase):
    name = "TestFinder"
    __test__ = False

    def __init__(self, config, instance_name):
        # Fill in expected fields for SampleModule
        config.setdefault("module_id", SampleModule.MODULE_ID)
        config.setdefault("train", {}).setdefault("batch_size", 1)
        config.setdefault("train", {}).setdefault("learning_rate", 0.1)
        self._fail_to_find = config.fail_to_find
        self._raise_on_find = config.raise_on_find
        self._config = config
        self._local_finder = model_finder_factory.construct({"type": "LOCAL"})
        self._instance_name = instance_name

    def find_model(self, model_path, *args, **kwargs):
        if self._raise_on_find:
            raise RuntimeError("You told me to")
        if self._fail_to_find:
            return None
        if os.path.exists(model_path):
            return self._local_finder.find_model(model_path, *args, **kwargs)
        return ModuleConfig(self._config)


model_finder_factory.register(TestFinder)


# Add a new model initializer that tests can use
class TestInitializer(ModelInitializerBase):
    name = "TestInitializer"
    __test__ = False

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name
        self.loaded_models = []
        self.local_initializer = model_initializer_factory.construct({"type": "LOCAL"})

    def init(self, model_config: ModuleConfig, *args, **kwargs) -> Optional[ModuleBase]:
        # allow config.model_type to control whether this loader barfs
        if "model_type" in self.config and "model_type" in kwargs:
            if self.config["model_type"] != kwargs["model_type"]:
                # Don't load in this loader
                return None
        # use the "Local" loader to actually load the model
        model = self.local_initializer.init(model_config)
        self.loaded_models.append(model)
        return model


model_initializer_factory.register(TestInitializer)


# Add a new simple trainer for tests to use
class TestTrainer(ModelTrainerBase):
    name = "TestTrainer"
    __test__ = False

    def __init__(self, config, instance_name):
        self.instance_name = instance_name
        self.canned_status = config.get("canned_status", TrainingStatus.RUNNING)
        self._futures = {}

    class TestModelFuture(ModelTrainerBase.ModelFutureBase):
        __test__ = False

        def __init__(self, parent, trained_model, save_path, save_with_id):
            super().__init__(
                trainer_name=parent.instance_name,
                training_id=str(uuid.uuid4()),
                save_path=save_path,
                save_with_id=save_with_id,
            )
            self._parent = parent
            self._trained_model = trained_model
            self._canceled = False
            self._completed = False

        def get_info(self):
            if self._completed:
                return TrainingInfo(status=TrainingStatus.COMPLETED)
            if self._canceled:
                return TrainingInfo(status=TrainingStatus.CANCELED)
            return TrainingInfo(status=self._parent.canned_status)

        def cancel(self):
            self._canceled = True

        def wait(self):
            self._completed = True
            if self.save_path:
                self._trained_model.save(self.save_path)

        def load(self):
            return self._trained_model

    def train(
        self,
        module_class: Type[ModuleBase],
        *args,
        save_path: Optional[str] = None,
        save_with_id: bool = False,
        **kwargs,
    ):
        trained_model = module_class.train(*args, **kwargs)
        future = self.TestModelFuture(self, trained_model, save_path, save_with_id)
        self._futures[future.id] = future
        return future

    def get_model_future(self, training_id: str) -> ModelTrainerBase.ModelFutureBase:
        if training_id not in self._futures:
            raise ValueError(f"Unknown training id: {training_id}")
        return self._futures[training_id]


model_trainer_factory.register(TestTrainer)


def configured_backends():
    local_initializers = [
        loader
        for loader in MODEL_MANAGER._initializers.values()
        if isinstance(loader, LocalModelInitializer)
    ]
    return [backend for loader in local_initializers for backend in loader._backends]
