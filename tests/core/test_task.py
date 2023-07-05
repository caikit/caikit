## Tests #######################################################################
# Standard
from typing import Iterable, Union
import uuid

# Third Party
import pytest

# Local
from caikit.core import TaskBase, task
from caikit.core.task import StreamingFlavor
from sample_lib import SampleModule
from sample_lib.data_model.sample import SampleInputType, SampleOutputType, SampleTask
import caikit.core


def test_task_decorator_has_streaming_types():
    @task()
    class SampleTask(TaskBase):
        unary_params: {"text": str}
        streaming_params: {"tokens": Iterable[str]}
        unary_output_type: SampleOutputType
        streaming_output_type: Iterable[SampleOutputType]

    assert (
        SampleTask.get_output_type(flavor=StreamingFlavor.STREAM_STREAM)
        == Iterable[SampleOutputType]
    )
    assert SampleTask.get_required_parameters(flavor=StreamingFlavor.STREAM_STREAM) == {
        "tokens": Iterable[str]
    }


def test_task_decorator_validates_class_extends_task_base():
    with pytest.raises(TypeError):

        @task()
        class SampleTask:
            pass


def test_task_decorator_validates_output_is_data_model():
    with pytest.raises(TypeError, match=".*str.* is not a subclass"):

        @task()
        class SampleTask(TaskBase):
            unary_params: {"text": str}
            unary_output_type: str


def test_task_decorator_can_have_iterable_output():
    """This test covers tasks + modules with streaming output"""

    @task()
    class StreamingTask(TaskBase):
        unary_params: {"sample_input": SampleInputType}
        streaming_output_type: Iterable[SampleOutputType]

    @caikit.core.module(
        id=str(uuid.uuid4()),
        name="StreamingModule",
        version="0.0.1",
        task=StreamingTask,
    )
    class StreamingModule(caikit.core.ModuleBase):
        @StreamingTask.taskmethod(streaming_flavor=StreamingFlavor.UNARY_STREAM)
        def run(
            self, sample_input: SampleInputType
        ) -> caikit.core.data_model.DataStream[SampleOutputType]:
            pass


def test_task_decorator_validates_streaming_output_is_iterable():
    with pytest.raises(TypeError, match="not a subclass of .*Iterable"):

        @task()
        class StreamingTask(TaskBase):
            streaming_params: {"text": Iterable[str]}
            streaming_output_type: str


def test_task_decorator_validates_streaming_input_is_iterable():
    with pytest.raises(TypeError, match="not a subclass of .*Iterable"):

        @task()
        class StreamingTask(TaskBase):
            streaming_params: {"text": str}
            streaming_output_type: Iterable[SampleOutputType]


def test_task_iterator_raises_on_wrong_streaming_type():
    @task()
    class StreamingTask(TaskBase):
        unary_params: {"sample_input": SampleInputType}
        streaming_output_type: Iterable[SampleOutputType]

    with pytest.raises(TypeError, match="Wrong output type for module"):

        @caikit.core.module(
            id=str(uuid.uuid4()),
            name="InvalidStreamingModule",
            version="0.0.1",
            task=StreamingTask,
        )
        class InvalidStreamingModule(caikit.core.ModuleBase):
            @StreamingTask.taskmethod(streaming_flavor=StreamingFlavor.UNARY_STREAM)
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
    @task()
    class SomeTask(TaskBase):
        unary_params: {"foo": SampleInputType}
        unary_output_type: SampleOutputType

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
    @task()
    class SomeTask(TaskBase):
        unary_params: {"foo": int}
        unary_output_type: SampleOutputType

    with pytest.raises(
        ValueError,
        match=".* failed validation for task .*",
    ):

        @caikit.core.module(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(caikit.core.ModuleBase):
            def run(self) -> SampleOutputType:
                pass


# def test_task_validation_throws_on_no_return_type():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     with pytest.raises(
#         ValueError,
#         match="Task could not be validated, no .run return type was provided",
#     ):

#         @caikit.core.module(
#             id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#         )
#         class Stuff(caikit.core.ModuleBase):
#             def run(self, foo: int):
#                 pass


# def test_task_validation_throws_on_wrong_return_type():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     with pytest.raises(
#         TypeError,
#         match="Wrong output type for module",
#     ):

#         @caikit.core.module(
#             id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#         )
#         class Stuff(caikit.core.ModuleBase):
#             def run(self, foo: int) -> SampleInputType:
#                 pass


# def test_task_validation_accepts_union_outputs():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     @caikit.core.module(
#         id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#     )
#     class Stuff(caikit.core.ModuleBase):
#         def run(self, foo: int) -> Union[SampleOutputType, int, str]:
#             pass


# def test_task_validation_throws_on_missing_parameter():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     with pytest.raises(TypeError, match="Required parameters .*foo.* not in signature"):

#         @caikit.core.module(
#             id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#         )
#         class Stuff(caikit.core.ModuleBase):
#             def run(self, bar: str) -> SampleOutputType:
#                 pass


# def test_task_validation_throws_on_wrong_parameter_type():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     with pytest.raises(
#         TypeError,
#         match="Parameter foo has type .*str.* but type .*int.* is required",
#     ):

#         @caikit.core.module(
#             id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#         )
#         class Stuff(caikit.core.ModuleBase):
#             def run(self, foo: str) -> SampleOutputType:
#                 pass


# def test_task_validation_passes_when_module_has_correct_run_signature():
#     @task(
#         required_parameters={"foo": int},
#         output_type=SampleOutputType,
#     )
#     class SomeTask(TaskBase):
#         pass

#     @caikit.core.module(
#         id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
#     )
#     class SomeModule(caikit.core.ModuleBase):
#         def run(self, foo: int, bar: str) -> SampleOutputType:
#             pass
