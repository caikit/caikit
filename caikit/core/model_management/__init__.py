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
The model management abstractions manage the key lifecycle phases of a concrete
model instance and are used by the ModelManager to handle each phase of the
model lifecycle.

For architectural details, see docs/adrs/020-model-management-abstractions.md
"""

# Local
from .factories import (
    model_finder_factory,
    model_initializer_factory,
    model_trainer_factory,
)
from .model_finder_base import ModelFinderBase
from .model_initializer_base import ModelInitializerBase
from .model_trainer_base import ModelTrainerBase, TrainingInfo
