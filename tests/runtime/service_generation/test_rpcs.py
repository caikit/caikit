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
from caikit.runtime.service_generation.rpcs import TaskPredictRPC
from caikit.runtime.service_generation.signature_parsing import (
    CaikitCoreModuleMethodSignature,
)
from sample_lib.data_model import SampleOutputType, SampleTask
import caikit.core


def test_task_inference_rpc_with_all_optional_params():
    @caikit.core.task(
        required_parameters={"str_val": str}, output_type=SampleOutputType
    )
    class TestTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testest", version="9.9.9", task=SampleTask
    )
    class TestModule(ModuleBase):
        def run(self, str_val="I have a default"):
            pass

    rpc = TaskPredictRPC(
        task=("foo", "bar"),
        method_signatures=[CaikitCoreModuleMethodSignature(TestModule, "run")],
        primitive_data_model_types=[],
    )

    data_model = rpc.create_request_data_model(package_name="blah")
    assert data_model is not None
