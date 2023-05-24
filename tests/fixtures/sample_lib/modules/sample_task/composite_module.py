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
# Local
from ...data_model import SampleInputType, SampleOutputType, SampleTask
from ...modules.sample_task.sample_implementation import SampleModule
from caikit.core.modules import ModuleLoader, ModuleSaver
import caikit


@caikit.module(
    "A34E68FA-E5E6-41BD-BAAE-77A880EB6877", "CompositeModule", "0.0.1", SampleTask
)
class CompositeModule(caikit.core.ModuleBase):
    def __init__(self, sample_block: SampleModule = SampleModule()):
        super().__init__()
        self.block = sample_block

    def run(self, sample_input: SampleInputType) -> SampleOutputType:
        """Runs the inner block"""
        return self.block.run(sample_input)

    @classmethod
    def load(cls, model_path: str):
        loader = ModuleLoader(model_path)
        return CompositeModule(loader.load_module("dummy_model"))

    def save(self, model_path):
        saver = ModuleSaver(
            module=self,
            model_path=model_path,
        )
        with saver:
            saver.save_module(self.block, "dummy_model")

    @classmethod
    def train(cls, sample_block: SampleModule) -> "CompositeModule":
        return cls(sample_block)
