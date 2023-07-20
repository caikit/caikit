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
Global factories for model management
"""
# Standard
from typing import Optional
import importlib

# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from ..toolkit.factory import Factory, FactoryConstructible
from .local_model_finder import LocalModelFinder
from .local_model_initializer import LocalModelInitializer
from .local_model_trainer import LocalModelTrainer

log = alog.use_channel("MMFCTRY")
error = error_handler.get(log)


class ImportableFactory(Factory):
    """An ImportableFactory extends the base Factory to allow the construction
    to specify an "import_class" field that will be used to import and register
    the implementation class before attempting to initialize it.
    """

    IMPORT_CLASS_KEY = "import_class"

    def construct(
        self,
        instance_config: dict,
        instance_name: Optional[str] = None,
    ):
        # Look for an import_class and import and register it if found
        import_class_val = instance_config.get(self.__class__.IMPORT_CLASS_KEY)
        if import_class_val:
            error.type_check(
                "<COR85108801E>",
                str,
                **{self.__class__.IMPORT_CLASS_KEY: import_class_val},
            )
            module_name, class_name = import_class_val.rsplit(".", 1)
            try:
                imported_module = importlib.import_module(module_name)
            except ImportError:
                error(
                    "<COR46837141E>",
                    ValueError(
                        "Invalid {}: Module cannot be imported [{}]".format(
                            self.__class__.IMPORT_CLASS_KEY,
                            module_name,
                        )
                    ),
                )
            try:
                imported_class = getattr(imported_module, class_name)
            except AttributeError:
                error(
                    "<COR46837142E>",
                    ValueError(
                        "Invalid {}: No such class [{}] on module [{}]".format(
                            self.__class__.IMPORT_CLASS_KEY,
                            class_name,
                            module_name,
                        )
                    ),
                )
            error.subclass_check("<COR52306423E>", imported_class, FactoryConstructible)

            self.register(imported_class)
        return super().construct(instance_config, instance_name)


# Model trainer factory. A trainer is responsible for performing the train
# operation against a configured framework connection.
model_trainer_factory = ImportableFactory("ModelTrainer")
model_trainer_factory.register(LocalModelTrainer)

# Model finder factory. A finder is responsible for locating a well defined
# configuration for a model based on a unique path or id.
model_finder_factory = ImportableFactory("ModelFinder")
model_finder_factory.register(LocalModelFinder)

# Model initializer factory. An initializer is responsible for taking a model
# configuration and preparing the model to be run in a configured runtime
# location.
model_initializer_factory = ImportableFactory("ModelInitializer")
model_initializer_factory.register(LocalModelInitializer)
