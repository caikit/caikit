## Tests #######################################################################
# Standard
from typing import Iterable, Union
import uuid

# Third Party
import pytest

# Local
from caikit.core import TaskBase, task
from sample_lib import SampleModule
from sample_lib.data_model.sample import SampleInputType, SampleOutputType, SampleTask
from sample_lib.modules.multi_task import FirstTask, MultiTaskModule, SecondTask
import caikit.core


def test_task_decorator_has_streaming_types():
    @task(
        unary_parameters={"text": str},
        streaming_parameters={"tokens": Iterable[str]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class SampleTask(TaskBase):
        pass

    assert (
        SampleTask.get_output_type(output_streaming=True) == Iterable[SampleOutputType]
    )
    assert SampleTask.get_required_parameters(input_streaming=True) == {
        "tokens": Iterable[str]
    }


def test_task_decorator_validates_class_extends_task_base():
    with pytest.raises(TypeError, match="is not a subclass of .*TaskBase"):

        @task()
        class SampleTask:
            pass


def test_task_decorator_validates_output_is_data_model():
    with pytest.raises(TypeError, match=".*str.* is not a subclass of .*DataBase"):

        @task(unary_parameters={"text": str}, unary_output_type=str)
        class SampleTask(TaskBase):
            pass


def test_task_decorator_can_have_iterable_output():
    """This test covers tasks + modules with streaming output"""

    @task(
        unary_parameters={"sample_input": SampleInputType},
        streaming_output_type=Iterable[SampleOutputType],
    )
    class StreamingTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="StreamingModule",
        version="0.0.1",
        task=StreamingTask,
    )
    class StreamingModule(caikit.core.ModuleBase):
        @StreamingTask.taskmethod(output_streaming=True)
        def run(
            self, sample_input: SampleInputType
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass


def test_task_decorator_validates_streaming_output_is_iterable():
    with pytest.raises(TypeError, match="not a subclass of .*Iterable"):

        @task(streaming_parameters={"text": Iterable[str]}, streaming_output_type=str)
        class StreamingTask(TaskBase):
            pass


def test_task_decorator_validates_streaming_input_is_iterable():
    with pytest.raises(TypeError, match="not a subclass of .*Iterable"):

        @task(
            streaming_parameters={"text": str},
            streaming_output_type=Iterable[SampleOutputType],
        )
        class StreamingTask(TaskBase):
            pass


def test_task_validator_raises_on_wrong_streaming_type():
    @task(
        unary_parameters={"sample_input": SampleInputType},
        streaming_output_type=Iterable[SampleOutputType],
    )
    class StreamingTask(TaskBase):
        pass

    with pytest.raises(TypeError, match="Wrong output type for module"):

        @caikit.core.module(
            id=str(uuid.uuid4()),
            name="InvalidStreamingModule",
            version="0.0.1",
            task=StreamingTask,
        )
        class InvalidStreamingModule(caikit.core.ModuleBase):
            @StreamingTask.taskmethod(output_streaming=True)
            def run(
                self, sample_input: SampleInputType
            ) -> caikit.core.data_model.DataStream[SampleInputType]:
                pass


def test_task_is_set_on_module_classes():
    assert hasattr(SampleModule, "tasks")
    assert len(SampleModule.tasks) == 1
    assert SampleTask in SampleModule.tasks


def test_multiple_tasks_are_set_on_module_class():
    assert hasattr(MultiTaskModule, "tasks")
    assert FirstTask in MultiTaskModule.tasks
    assert SecondTask in MultiTaskModule.tasks


def test_task_can_be_inferred_from_parent_module():
    @caikit.core.modules.module(id="foobar", name="Stuff", version="0.0.1")
    class Stuff(SampleModule):
        pass

    assert Stuff.tasks == SampleModule.tasks


def test_multiple_tasks_inherited_from_parent_module():
    @caikit.core.modules.module(
        id="multichild", name="MultiTaskChildModule", version="0.0.1"
    )
    class MultiTaskChildModule(MultiTaskModule):
        pass

    assert FirstTask in MultiTaskChildModule.tasks
    assert SecondTask in MultiTaskChildModule.tasks


def test_tasks_added_from_parent_and_child_module():
    @task(unary_parameters={"foo": int}, unary_output_type=SampleOutputType)
    class ThirdTask(TaskBase):
        pass

    @caikit.core.modules.module(
        id="taskfamily", name="MultiTaskChildModule", version="0.0.1", tasks=[ThirdTask]
    )
    class MultiTaskChildModule(MultiTaskModule):
        @ThirdTask.taskmethod()
        def run_third_task(self, foo: int) -> SampleOutputType:
            pass

    for t in [FirstTask, SecondTask, ThirdTask]:
        assert t in MultiTaskChildModule.tasks

    # Make sure no tasks are double-counted
    assert len(MultiTaskChildModule.tasks) == len(MultiTaskChildModule._TASK_CLASSES)


def test_task_is_not_required_for_modules():
    @caikit.core.modules.module(id=str(uuid.uuid4()), name="Stuff", version="0.0.1")
    class Stuff(caikit.core.ModuleBase):
        pass

    assert Stuff.tasks == set()


def test_raises_if_tasks_not_list():
    with pytest.raises(
        TypeError,
        match=".*tasks.*list.*",
    ):

        @caikit.core.modules.module(
            id=str(uuid.uuid4()),
            name="BadTypeModule",
            version="0.0.1",
            tasks=set([FirstTask, SecondTask]),
        )
        class Stuff(caikit.core.ModuleBase):
            pass


def test_task_and_tasks_are_mutually_exclusive():
    with pytest.raises(
        ValueError,
        match=".*Specify either task or tasks parameter, not both",
    ):

        @caikit.core.modules.module(
            id=str(uuid.uuid4()),
            name="Stuff",
            version="0.0.1",
            task=SampleTask,
            tasks=[FirstTask, SecondTask],
        )
        class Stuff(caikit.core.ModuleBase):
            pass


def test_task_validation_throws_on_no_params():
    @task(unary_parameters={"foo": int}, unary_output_type=SampleOutputType)
    class SomeTask(TaskBase):
        pass

    with pytest.raises(
        ValueError,
        match="Task could not be validated, no .run parameters were provided",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self) -> SampleOutputType:
                pass


def test_task_validation_throws_on_no_return_type():
    @task(unary_parameters={"foo": int}, unary_output_type=SampleOutputType)
    class SomeTask(TaskBase):
        pass

    with pytest.raises(
        ValueError,
        match="Task could not be validated, no .run return type was provided",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self, foo: int):
                pass


def test_task_validation_throws_on_wrong_return_type():
    @task(
        unary_parameters={"foo": int},
        unary_output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    with pytest.raises(
        TypeError,
        match="Wrong output type for module",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self, foo: int) -> SampleInputType:
                pass


def test_task_validation_accepts_union_outputs():
    @task(
        unary_parameters={"foo": int},
        unary_output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
    )
    class Stuff(caikit.core.ModuleBase):
        def run(self, foo: int) -> Union[SampleOutputType, int, str]:
            pass


def test_task_validation_throws_on_missing_parameter():
    @task(
        unary_parameters={"foo": int},
        unary_output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    with pytest.raises(TypeError, match="Required parameters .*foo.* not in signature"):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self, bar: str) -> SampleOutputType:
                pass


def test_task_validation_throws_on_wrong_parameter_type():
    @task(
        unary_parameters={"foo": int},
        unary_output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    with pytest.raises(
        TypeError,
        match="Parameter foo has type .*str.* but type .*int.* is required",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self, foo: str) -> SampleOutputType:
                pass


def test_task_validation_passes_when_module_has_correct_run_signature():
    @task(
        unary_parameters={"foo": int},
        unary_output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
    )
    class SomeModule(caikit.core.ModuleBase):
        def run(self, foo: int, bar: str) -> SampleOutputType:
            pass


def test_task_decorator_adds_taskmethods_to_modules():
    """This test covers tasks + modules with streaming output"""

    @task(
        unary_parameters={"sample_input": SampleInputType},
        unary_output_type=SampleOutputType,
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        streaming_output_type=Iterable[SampleOutputType],
    )
    class StreamingTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="StreamingModule",
        version="0.0.1",
        task=StreamingTask,
    )
    class StreamingModule(caikit.core.ModuleBase):
        @StreamingTask.taskmethod()
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

        @StreamingTask.taskmethod(output_streaming=True)
        def run_stream_out(
            self, sample_input: SampleInputType
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass

        @StreamingTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: caikit.core.data_model.DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

    assert (
        StreamingModule.get_inference_signature(
            input_streaming=False, output_streaming=False
        ).method_name
        == "run"
    )
    assert (
        StreamingModule.get_inference_signature(
            input_streaming=False, output_streaming=True
        ).method_name
        == "run_stream_out"
    )

    assert (
        StreamingModule.get_inference_signature(
            input_streaming=True, output_streaming=False
        ).method_name
        == "run_stream_in"
    )


def test_inference_signatures_returned_for_multiple_tasks():
    first_task_signatures = MultiTaskModule.get_inference_signatures(FirstTask)

    assert first_task_signatures is not None
    assert len(first_task_signatures) == 1
    (in_stream, out_stream, signature) = first_task_signatures[0]
    assert in_stream is False
    assert out_stream is False
    assert signature.method_name == "run_some_task"

    second_task_signatures = MultiTaskModule.get_inference_signatures(SecondTask)

    assert second_task_signatures is not None
    assert len(second_task_signatures) == 1
    (in_stream, out_stream, signature) = second_task_signatures[0]
    assert in_stream is False
    assert out_stream is False
    assert signature.method_name == "run_other_task"


def test_task_decorator_datastream_throws_wrong_type():
    @task(
        unary_parameters={"sample_input": SampleInputType},
        unary_output_type=SampleOutputType,
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        streaming_output_type=Iterable[SampleOutputType],
    )
    class DataStreamStreamingTask(TaskBase):
        pass

    with pytest.raises(
        TypeError,
        match=".*expected .*SampleInputType.* but got .*SampleOutputType",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()),
            name="DataStreamStreamingModule",
            version="0.0.1",
            task=DataStreamStreamingTask,
        )
        class DataStreamStreamingModule(caikit.core.ModuleBase):
            @DataStreamStreamingTask.taskmethod(input_streaming=True)
            def run_stream_in(
                self, sample_inputs: caikit.core.data_model.DataStream[SampleOutputType]
            ) -> SampleOutputType:
                pass


def test_task_decorator_datastream_params():
    @task(
        unary_parameters={"sample_input": SampleInputType},
        unary_output_type=SampleOutputType,
        streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
        streaming_output_type=Iterable[SampleOutputType],
    )
    class DataStreamStreamingTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="DataStreamStreamingModule",
        version="0.0.1",
        task=DataStreamStreamingTask,
    )
    class DataStreamStreamingModule(caikit.core.ModuleBase):
        @DataStreamStreamingTask.taskmethod()
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

        @DataStreamStreamingTask.taskmethod(output_streaming=True)
        def run_stream_out(
            self, sample_input: SampleInputType
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass

        @DataStreamStreamingTask.taskmethod(input_streaming=True)
        def run_stream_in(
            self, sample_inputs: caikit.core.data_model.DataStream[SampleInputType]
        ) -> SampleOutputType:
            pass

        @DataStreamStreamingTask.taskmethod(input_streaming=True, output_streaming=True)
        def run_stream_bidi(
            self, sample_inputs: caikit.core.data_model.DataStream[SampleInputType]
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass

    signatures = DataStreamStreamingModule.get_inference_signatures(
        DataStreamStreamingTask
    )

    stream_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=False, output_streaming=False
    )
    assert (False, False, stream_stream_method_signature) in signatures

    out_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=False, output_streaming=True
    )
    assert (False, True, out_stream_method_signature) in signatures

    in_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=True, output_streaming=False
    )
    assert (True, False, in_stream_method_signature) in signatures

    bidi_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=True, output_streaming=True
    )
    assert (True, True, bidi_stream_method_signature) in signatures

    assert stream_stream_method_signature.method_name == "run"
    assert stream_stream_method_signature.parameters == {
        "sample_input": SampleInputType
    }
    assert stream_stream_method_signature.return_type == SampleOutputType

    assert out_stream_method_signature.method_name == "run_stream_out"
    assert out_stream_method_signature.parameters == {"sample_input": SampleInputType}
    assert (
        out_stream_method_signature.return_type
        == caikit.core.data_model.streams.data_stream.DataStream[SampleOutputType]
    )

    assert in_stream_method_signature.method_name == "run_stream_in"
    assert in_stream_method_signature.parameters == {
        "sample_inputs": caikit.core.data_model.streams.data_stream.DataStream[
            SampleInputType
        ]
    }
    assert in_stream_method_signature.return_type == SampleOutputType

    assert bidi_stream_method_signature.method_name == "run_stream_bidi"
    assert bidi_stream_method_signature.parameters == {
        "sample_inputs": caikit.core.data_model.streams.data_stream.DataStream[
            SampleInputType
        ]
    }
    assert (
        bidi_stream_method_signature.return_type
        == caikit.core.data_model.streams.data_stream.DataStream[SampleOutputType]
    )


def test_decorator_adds_default_run_method_to_modules():
    @task(
        unary_parameters={"sample_input": SampleInputType},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class SomeTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="SomeModule",
        version="0.0.1",
        task=SomeTask,
    )
    class SomeModule(caikit.core.ModuleBase):
        def run(self, sample_input: SampleInputType) -> SampleOutputType:
            pass

    assert (
        SomeModule.get_inference_signature(
            input_streaming=False, output_streaming=False
        ).method_name
        == "run"
    )


def test_validation_allows_union_subsets():
    """Validate that a task can take a union type that is a subset of implementing module types."""
    # Task param types need to correctly map to proto types since they're used by runtime
    @task(
        unary_parameters={"sample_input": Union[str, int]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class SomeTask(TaskBase):
        pass

    # But a module may consume types that are not backed by proto, e.g., PIL images
    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="SomeModule",
        version="0.0.1",
        task=SomeTask,
    )
    class SomeModule(caikit.core.ModuleBase):
        def run(self, sample_input: Union[str, int, bytes]) -> SampleOutputType:
            pass


def test_validation_does_not_allow_union_supersets():
    """Ensure that an implementing module cannot take a subset of param types of the task."""

    @task(
        unary_parameters={"sample_input": Union[str, int, bytes]},
        unary_output_type=SampleOutputType,
        streaming_output_type=Iterable[SampleOutputType],
    )
    class SomeTask(TaskBase):
        pass

    # If the task says bytes are okay, the module needs to be able to handle bytes also
    with pytest.raises(TypeError):

        @caikit.core.module(
            id=str(uuid.uuid4()),
            name="SomeModule",
            version="0.0.1",
            task=SomeTask,
        )
        class SomeModule(caikit.core.ModuleBase):
            def run(self, sample_input: Union[str, int]) -> SampleOutputType:
                pass


# ----------- BACKWARDS COMPATIBILITY ------------------------------------------- ##


def test_task_backwards_compatibility():
    """The old 'required_parameters' and 'output_type' should continue to function"""

    @task(
        required_parameters={"foo": int},
        output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    @caikit.core.module(
        id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
    )
    class SomeModule(caikit.core.ModuleBase):
        def run(self, foo: int, bar: str) -> SampleOutputType:
            pass

    with pytest.raises(TypeError):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class SomeModule(caikit.core.ModuleBase):
            def run(self, foo: str) -> SampleOutputType:
                pass

    with pytest.raises(TypeError):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class SomeModule(caikit.core.ModuleBase):
            def run(self, foo: int) -> SampleInputType:
                pass
