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
from typing import Iterable, Union
import uuid

# Local
from caikit.core import ModuleBase, TaskBase
from caikit.core.data_model import DataStream
from caikit.runtime.service_generation.rpcs import ModuleClassTrainRPC, TaskPredictRPC
from sample_lib.data_model import SampleInputType, SampleOutputType
import caikit.core


def test_task_inference_multiples_modules_rpc():
    @caikit.core.task(
        unary_parameters={"sample_input": SampleInputType},
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class MultiModTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testmod1", version="9.9.9", task=MultiModTask
    )
    class TestModule1(ModuleBase):
        @MultiModTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testmod2", version="9.9.9", task=MultiModTask
    )
    class TestModule2(ModuleBase):
        @MultiModTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

    rpc = TaskPredictRPC(
        task=MultiModTask,
        method_signatures=[
            TestModule1.get_inference_signature(
                input_streaming=True, output_streaming=False
            ),
            TestModule2.get_inference_signature(
                input_streaming=True, output_streaming=False
            ),
        ],
        input_streaming=True,
        output_streaming=False,
    )
    assert rpc.request.name == "ClientStreamingMultiModTaskRequest"
    assert rpc.request.triples == [(SampleInputType, "sample_inputs", 1)]

    data_model = rpc.create_request_data_model(package_name="blah")
    assert data_model is not None

    assert rpc.name == "ClientStreamingMultiModTaskPredict"


def test_task_inference_rpc_with_client_streaming():
    @caikit.core.task(
        unary_parameters={"sample_input": SampleInputType},
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class TestTask1(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testest", version="9.9.9", task=TestTask1
    )
    class TestModule(ModuleBase):
        @TestTask1.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

    rpc = TaskPredictRPC(
        task=TestTask1,
        method_signatures=[
            TestModule.get_inference_signature(
                input_streaming=True, output_streaming=False
            )
        ],
        input_streaming=True,
        output_streaming=False,
    )
    assert rpc.request.name == "ClientStreamingTestTask1Request"
    assert rpc.request.triples == [(SampleInputType, "sample_inputs", 1)]

    data_model = rpc.create_request_data_model(package_name="blah")
    assert data_model is not None

    assert rpc.name == "ClientStreamingTestTask1Predict"


def test_task_inference_rpc_with_streaming():
    @caikit.core.task(
        unary_parameters={"sample_input": SampleInputType},
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class TestTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="testest", version="9.9.9", task=TestTask
    )
    class TestModule(ModuleBase):
        @TestTask.taskmethod()
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

        @TestTask.taskmethod(output_streaming=True)
        def run_stream_out(
            self, sample_input: SampleInputType
        ) -> DataStream[SampleOutputType]:
            pass

        @TestTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

        @TestTask.taskmethod(input_streaming=True, output_streaming=True)
        def run_stream_bidi(
            self, sample_inputs: Iterable[SampleInputType]
        ) -> DataStream[SampleOutputType]:
            pass

    # Unary
    rpc = TaskPredictRPC(
        task=TestTask,
        method_signatures=[
            TestModule.get_inference_signature(
                input_streaming=False, output_streaming=False
            )
        ],
    )

    assert rpc.name == "TestTaskPredict"

    # # Server streaming
    rpc = TaskPredictRPC(
        task=TestTask,
        method_signatures=[
            TestModule.get_inference_signature(
                input_streaming=False, output_streaming=True
            )
        ],
        input_streaming=False,
        output_streaming=True,
    )

    assert rpc.name == "ServerStreamingTestTaskPredict"

    # Client streaming
    rpc = TaskPredictRPC(
        task=TestTask,
        method_signatures=[
            TestModule.get_inference_signature(
                input_streaming=True, output_streaming=False
            )
        ],
        input_streaming=True,
        output_streaming=False,
    )

    assert rpc.name == "ClientStreamingTestTaskPredict"

    # Bidi streaming
    rpc = TaskPredictRPC(
        task=TestTask,
        method_signatures=[
            TestModule.get_inference_signature(
                input_streaming=True, output_streaming=True
            )
        ],
        input_streaming=True,
        output_streaming=True,
    )

    assert rpc.name == "BidiStreamingTestTaskPredict"


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
