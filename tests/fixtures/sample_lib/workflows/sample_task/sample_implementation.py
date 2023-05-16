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
from caikit.core.workflows import WorkflowLoader, WorkflowSaver
import caikit.core


@caikit.core.workflow(
    "A34E68FA-E5E6-41BD-BAAE-77A880EB6877", "SampleWorkflow", "0.0.1", SampleTask
)
class SampleWorkflow(caikit.core.WorkflowBase):
    def __init__(self, sample_block: SampleModule = SampleModule()):
        super().__init__()
        self.block = sample_block

    def run(self, sample_input: SampleInputType) -> SampleOutputType:
        """Runs the inner block"""
        return self.block.run(sample_input)

    @classmethod
    def _load(cls, workflow_loader: WorkflowLoader):
        return SampleWorkflow(workflow_loader.load_module("dummy_model"))

    def save(self, model_path):
        saver = WorkflowSaver(
            module=self,
            model_path=model_path,
        )
        with saver:
            saver.save_module(self.block, "dummy_model")

    @classmethod
    def train(cls, sample_block: SampleModule) -> "SampleWorkflow":
        return cls(sample_block)
