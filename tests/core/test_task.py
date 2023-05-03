## Tests #######################################################################
# Standard
import uuid

# Third Party
import pytest

# Local
from caikit.core import TaskBase, TaskGroupBase, task, taskgroup
from sample_lib import SampleBlock
from sample_lib.data_model.sample import (
    SampleInputType,
    SampleOutputType,
    SampleTask,
    SampleTaskGroup,
)
import caikit.core


@taskgroup(input_types={SampleInputType})
class TestTaskGroup(TaskGroupBase):
    pass


def test_task_decorator_has_required_inputs_and_output_type():
    # TODO: whoops, the DataBase classes _don't_ have DataBase base class at static check time
    @task(
        task_group=TestTaskGroup,
        required_inputs={"sample_input": SampleInputType},
        output_type=SampleOutputType,
    )
    class SampleTask(TaskBase):
        pass

    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}
    assert SampleTask.get_output_type() == SampleOutputType

    # assert Immutable properties
    SampleTask.required_inputs = {"sample_input": str}
    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}


def test_task_decorator_validates_class_extends_task_base():
    with pytest.raises(TypeError):

        @task(
            task_group=TestTaskGroup,
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask:
            pass


def test_task_decorator_validates_output_is_data_model():
    with pytest.raises(TypeError):

        @task(
            task_group=TestTaskGroup,
            required_inputs={"sample_input": SampleInputType},
            output_type=str,
        )
        class SampleTask(TaskBase):
            pass


def test_task_decorator_validates_input_is_in_domain():
    with pytest.raises(TypeError):
        # `str` is not in TestDomain.input_types
        @task(
            task_group=TestTaskGroup,
            required_inputs={"sample_input": str},
            output_type=SampleOutputType,
        )
        class SampleTask(TaskBase):
            pass


def test_task_decorator_validates_domain_is_a_domain():
    with pytest.raises(TypeError):
        # `str` is not in TestDomain.input_types
        @task(
            task_group=str,
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask(TaskBase):
            pass


def test_domain_has_input_type_set():
    @taskgroup(input_types={int, float})
    class SampleTaskGroup(TaskGroupBase):
        pass

    assert SampleTaskGroup.get_input_type_set() == {int, float}


def test_domain_validates_inputs_are_protoabletypes():
    with pytest.raises(TypeError):

        @taskgroup(input_types={1, 2.3})
        class SampleTaskGroup(TaskGroupBase):
            pass

    with pytest.raises(TypeError):

        @taskgroup(input_types={dict})
        class SampleTaskGroup(TaskGroupBase):
            pass

    with pytest.raises(TypeError):

        @taskgroup(input_types={caikit.core.ModuleBase})
        class SampleTaskGroup(TaskGroupBase):
            pass


def test_task_is_set_on_module_classes():
    assert hasattr(SampleBlock, "TASK_CLASS")
    assert SampleBlock.TASK_CLASS == SampleTask


def test_task_can_be_inferred_from_parent_block():
    @caikit.core.blocks.block(id="foobar", name="Stuff", version="0.0.1")
    class Stuff(SampleBlock):
        pass

    assert Stuff.TASK_CLASS == SampleBlock.TASK_CLASS


def test_task_cannot_conflict_with_parent_block():
    @task(
        task_group=SampleTaskGroup,
        required_inputs={"foo": SampleInputType},
        output_type=SampleOutputType,
    )
    class SomeTask(TaskBase):
        pass

    with pytest.raises(TypeError, match="but superclass has"):

        @caikit.core.blocks.block(
            id=str(uuid.uuid4()), name="Stuff", version="0.0.1", task=SomeTask
        )
        class Stuff(SampleBlock):
            pass


def test_task_is_temporarily_not_required_for_blocks():
    # TODO: remove (or assert failure in) this test once task is required
    @caikit.core.blocks.block(id=str(uuid.uuid4()), name="Stuff", version="0.0.1")
    class Stuff(caikit.core.blocks.base.BlockBase):
        pass
