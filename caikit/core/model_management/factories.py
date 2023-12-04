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

# Local
from ..toolkit.factory import ImportableFactory
from .local_model_finder import LocalModelFinder
from .local_model_initializer import LocalModelInitializer
from .local_model_trainer import LocalModelTrainer
from .multi_model_finder import MultiModelFinder
from .multi_model_initializer import MultiModelInitializer

# Model trainer factory. A trainer is responsible for performing the train
# operation against a configured framework connection.
model_trainer_factory = ImportableFactory("ModelTrainer")
model_trainer_factory.register(LocalModelTrainer)

# Model finder factory. A finder is responsible for locating a well defined
# configuration for a model based on a unique path or id.
model_finder_factory = ImportableFactory("ModelFinder")
model_finder_factory.register(LocalModelFinder)
model_finder_factory.register(MultiModelFinder)

# Model initializer factory. An initializer is responsible for taking a model
# configuration and preparing the model to be run in a configured runtime
# location.
model_initializer_factory = ImportableFactory("ModelInitializer")
model_initializer_factory.register(LocalModelInitializer)
model_initializer_factory.register(MultiModelInitializer)
