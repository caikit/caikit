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
# Standard
from typing import Iterable
import uuid

# Local
from caikit.runtime.service_generation.create_service import (
    create_inference_rpcs,
    create_training_rpcs,
)
from sample_lib.data_model import (
    OtherOutputType,
    SampleInputType,
    SampleOutputType,
    SampleTask,
)
from sample_lib.modules import SampleModule
import caikit
import sample_lib

## Setup ########################################################################

widget_class = sample_lib.modules.sample_task.SampleModule

## Tests ########################################################################

### create_inference_rpcs tests #################################################


def test_create_inference_rpcs_uses_task_from_module_decorator():
    # make a new module with SampleTask
    @caikit.module(
        id=str(uuid.uuid4()), name="something", version="0.0.0", task=SampleTask
    )
    class NewModule(caikit.core.ModuleBase):
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

    # SampleModule also implements `SampleTask`
    rpcs = create_inference_rpcs([NewModule, SampleModule])
    assert len(rpcs) == 3  # SampleModule has 3 streaming flavors
    assert NewModule in rpcs[0].module_list
    assert SampleModule in rpcs[0].module_list


def test_create_inference_rpcs_uses_task_from_module_decorator_with_streaming():
    @caikit.module(
        id=str(uuid.uuid4()),
        name="NewStreamingModule1",
        version="0.0.0",
        task=SampleTask,
    )
    class NewStreamingModule1(caikit.core.ModuleBase):
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

    @caikit.module(
        id=str(uuid.uuid4()),
        name="NewStreamingModule2",
        version="0.0.0",
        task=SampleTask,
    )
    class NewStreamingModule2(caikit.core.ModuleBase):
        @SampleTask.taskmethod()
        def run_unary(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

        @SampleTask.taskmethod(input_streaming=True, output_streaming=True)
        def run_stream_bidi(
            self, sample_inputs: caikit.core.data_model.DataStream[SampleInputType]
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass

    @caikit.module(
        id=str(uuid.uuid4()),
        name="NewStreamingModule3",
        version="0.0.0",
        task=SampleTask,
    )
    class NewStreamingModule3(caikit.core.ModuleBase):
        @SampleTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: caikit.core.data_model.DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

    @caikit.core.task(
        unary_parameters={"text": str},
        streaming_parameters={"texts": Iterable[str]},
        unary_output_type=OtherOutputType,
        streaming_output_type=Iterable[OtherOutputType],
    )
    class OtherStreamingTask(caikit.core.TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="TestModule1",
        version="0.0.0",
        task=OtherStreamingTask,
    )
    class TestModule1(caikit.core.ModuleBase):
        @OtherStreamingTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, texts: caikit.core.data_model.DataStream[str]
        ) -> OtherOutputType:
            pass

        @OtherStreamingTask.taskmethod(output_streaming=True)
        def run_stream_out(
            self, text: str
        ) -> caikit.core.data_model.DataStream[OtherOutputType]:
            pass

    # Not including NewStreamingModule3 to check that we don't get ClientStreaming RPC generated
    rpcs = create_inference_rpcs(
        [NewStreamingModule1, NewStreamingModule2, SampleModule]
    )
    assert len(rpcs) == 3
    _test_rpc(
        rpcs,
        task=SampleTask,
        input_streaming=False,
        output_streaming=False,
        expected_name="SampleTaskPredict",
        expected_module_list=[NewStreamingModule1, NewStreamingModule2, SampleModule],
    )  # unary
    _test_rpc(
        rpcs,
        task=SampleTask,
        input_streaming=True,
        output_streaming=True,
        expected_name="BidiStreamingSampleTaskPredict",
        expected_module_list=[NewStreamingModule2, SampleModule],
    )  # bidi stream
    _test_rpc(
        rpcs,
        task=SampleTask,
        input_streaming=False,
        output_streaming=True,
        expected_name="ServerStreamingSampleTaskPredict",
        expected_module_list=[SampleModule],
    )  # out stream

    rpcs = create_inference_rpcs(
        [
            NewStreamingModule1,
            NewStreamingModule2,
            NewStreamingModule3,
            SampleModule,
            TestModule1,
        ]
    )
    assert len(rpcs) == 6
    # only checking the new rpcs here
    _test_rpc(
        rpcs,
        task=SampleTask,
        input_streaming=True,
        output_streaming=False,
        expected_name="ClientStreamingSampleTaskPredict",
        expected_module_list=[NewStreamingModule3],
    )  # in stream
    _test_rpc(
        rpcs,
        task=OtherStreamingTask,
        input_streaming=True,
        output_streaming=False,
        expected_name="ClientStreamingOtherStreamingTaskPredict",
        expected_module_list=[TestModule1],
    )  # OtherStreamingTask's in stream
    _test_rpc(
        rpcs,
        task=OtherStreamingTask,
        input_streaming=False,
        output_streaming=True,
        expected_name="ServerStreamingOtherStreamingTaskPredict",
        expected_module_list=[TestModule1],
    )  # OtherStreamingTask's out stream


def _test_rpc(
    rpcs, task, input_streaming, output_streaming, expected_name, expected_module_list
):
    rpc_list = [
        rpc
        for rpc in rpcs
        if rpc.task == task
        and rpc._input_streaming == input_streaming
        and rpc._output_streaming == output_streaming
    ]
    assert len(rpc_list) == 1
    rpc = rpc_list[0]
    assert rpc.name == expected_name
    assert set(rpc.module_list) == set(expected_module_list)


def test_create_inference_rpcs():
    rpcs = create_inference_rpcs([widget_class])
    assert len(rpcs) == 3  # SampleModule has inference methods for 3 streaming flavors
    assert widget_class in rpcs[0].module_list


def test_create_inference_rpcs_for_multiple_modules_of_same_type():
    module_list = [
        sample_lib.modules.sample_task.SampleModule,
        sample_lib.modules.sample_task.SamplePrimitiveModule,
        sample_lib.modules.other_task.OtherModule,
    ]
    rpcs = create_inference_rpcs(module_list)

    # 4 RPCs, SampleModule and SamplePrimitiveModule have task SampleTask with 3 flavors for
    # streaming, OtherModule has task OtherTask
    assert len(rpcs) == 4
    assert sample_lib.modules.sample_task.SampleModule in rpcs[0].module_list
    assert sample_lib.modules.sample_task.SamplePrimitiveModule in rpcs[0].module_list
    assert sample_lib.modules.sample_task.SampleModule in rpcs[1].module_list
    assert sample_lib.modules.sample_task.SampleModule in rpcs[2].module_list
    assert sample_lib.modules.other_task.OtherModule in rpcs[-1].module_list


def test_create_inference_rpcs_removes_modules_with_no_task():
    module_list = [
        sample_lib.modules.sample_task.SampleModule,  # has a task, has 3 streaming flavors
        sample_lib.modules.sample_task.InnerModule,  # does not have a task
    ]
    rpcs = create_inference_rpcs(module_list)

    assert len(rpcs) == 3
    assert sample_lib.modules.sample_task.SampleModule in rpcs[0].module_list
    assert sample_lib.modules.sample_task.InnerModule not in rpcs[0].module_list


### create_training_rpcs tests #################################################


def test_no_training_rpcs_module_with_no_train_function():
    @caikit.module(
        id=str(uuid.uuid4()), name="something", version="0.0.0", task=SampleTask
    )
    class Foo(caikit.core.ModuleBase):
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

        def train_in_progress(self):
            pass

    rpcs = create_training_rpcs([Foo])
    assert len(rpcs) == 0


def test_no_training_rpcs_for_module_with_no_task():
    @caikit.module(id=str(uuid.uuid4()), name="something", version="0.0.0")
    class Foo(caikit.core.ModuleBase):
        def train(self, foo: int) -> "Foo":
            pass

    rpcs = create_training_rpcs([Foo])
    assert len(rpcs) == 0


def test_create_training_rpcs():
    rpcs = create_training_rpcs([widget_class])
    assert len(rpcs) == 1
    assert widget_class in rpcs[0].module_list
