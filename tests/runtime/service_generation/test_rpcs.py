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

"""Tests for the rpc objects that hold our in-memory representation of
what an RPC for a service looks like"""
# Standard
import uuid

# Local
from caikit.core import ModuleBase, TaskBase
from caikit.runtime.service_generation.rpcs import ModuleClassTrainRPC, TaskPredictRPC
from sample_lib.data_model import SampleOutputType
import caikit.core


def test_task_inference_rpc_with_all_optional_params():
    @caikit.core.task(
        required_parameters={"str_val": str}, output_type=SampleOutputType
    )
    class TestTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testest", version="9.9.9", task=TestTask
    )
    class TestModule(ModuleBase):
        def run(self, str_val="I have a default") -> SampleOutputType:
            pass

    rpc = TaskPredictRPC(
        task=TestTask,
        method_signatures=[TestModule.RUN_SIGNATURE],
    )

    data_model = rpc.create_request_data_model(package_name="blah")
    assert data_model is not None

    assert rpc.name == "TestTaskPredict"


def test_module_train_rpc():
    @caikit.core.task(
        required_parameters={"str_val": str}, output_type=SampleOutputType
    )
    class TestTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testest", version="9.9.9", task=TestTask
    )
    class TestModule(ModuleBase):
        def run(self, str_val: str) -> SampleOutputType:
            pass

        @classmethod
        def train(cls, int_val: int, str_val: str) -> "TestModule":
            pass

    rpc = ModuleClassTrainRPC(method_signature=TestModule.TRAIN_SIGNATURE)

    data_model = rpc.create_request_data_model(package_name="blah")
    assert data_model is not None

    assert rpc.name == "TestTaskTestModuleTrain"
