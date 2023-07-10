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
    assert hasattr(SampleModule, "TASK_CLASS")
    assert SampleModule.TASK_CLASS == SampleTask


def test_task_can_be_inferred_from_parent_module():
    @caikit.core.modules.module(id="foobar", name="Stuff", version="0.0.1")
    class Stuff(SampleModule):
        pass

    assert Stuff.TASK_CLASS == SampleModule.TASK_CLASS


def test_task_cannot_conflict_with_parent_module():
    @task(unary_parameters={"foo": SampleInputType}, unary_output_type=SampleOutputType)
    class SomeTask(TaskBase):
        pass

    with pytest.raises(TypeError, match="but superclass has"):

        @caikit.core.modules.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(SampleModule):
            pass


def test_task_is_not_required_for_modules():
    @caikit.core.modules.module(id=str(uuid.uuid4()), name="Stuff", version="0.0.1")
    class Stuff(caikit.core.ModuleBase):
        pass

    assert Stuff.TASK_CLASS is None


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

    stream_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=False, output_streaming=False
    )

    out_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=False, output_streaming=True
    )

    in_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=True, output_streaming=False
    )

    bidi_stream_method_signature = DataStreamStreamingModule.get_inference_signature(
        input_streaming=True, output_streaming=True
    )

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
