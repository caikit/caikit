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
from caikit.core.toolkit.factory import ImportableFactory
from caikit.runtime.model_management.core_model_loader import CoreModelLoader
from caikit.runtime.model_management.directory_model_sizer import DirectoryModelSizer
from caikit.runtime.model_management.mm_model_sizer import ModelMeshModelSizer

# Model Loader factory. A loader is responsible for constructing
# a LoadedModel instance
model_loader_factory = ImportableFactory("ModelLoader")
model_loader_factory.register(CoreModelLoader)

# Model Sizer factory. A sizer is responsible for estimating
# the size of a model
model_sizer_factory = ImportableFactory("ModelSizer")
model_sizer_factory.register(DirectoryModelSizer)
model_sizer_factory.register(ModelMeshModelSizer)
