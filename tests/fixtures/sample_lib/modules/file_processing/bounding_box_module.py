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
A sample module for sample things!
"""
# Local
from ...data_model.sample import FileDataType, FileTask
from caikit.core.modules import ModuleLoader
import caikit.core


@caikit.core.module(
    "750cad4d-c3b8-4327-b52e-e772f0d6f311", "BoundingBoxModule", "0.0.1", FileTask
)
class BoundingBoxModule(caikit.core.ModuleBase):
    def run(
        self,
        unprocessed: FileDataType,
    ) -> FileDataType:
        filename = f"processed_{unprocessed.filename}"
        data = b"bounding|" + unprocessed.data + b"|box"
        return FileDataType(filename, data)

    @classmethod
    def load(cls, model_path, **kwargs):
        loader = ModuleLoader(model_path)
        config = loader.config
        return cls()
