## Tests #######################################################################
# Standard
import uuid

# Third Party
import pytest

# Local
from caikit.core import TaskBase, task
from sample_lib import SampleBlock
from sample_lib.data_model.sample import SampleInputType, SampleOutputType, SampleTask
import caikit.core


def test_task_decorator_has_required_inputs_and_output_type():
    @task(
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
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask:
            pass


def test_task_decorator_validates_output_is_data_model():
    with pytest.raises(TypeError):

        @task(
            required_inputs={"sample_input": SampleInputType},
            output_type=str,
        )
        class SampleTask(TaskBase):
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


def test_task_is_not_required_for_blocks():
    @caikit.core.blocks.block(id=str(uuid.uuid4()), name="Stuff", version="0.0.1")
    class Stuff(caikit.core.blocks.base.BlockBase):
        pass

    assert Stuff.TASK_CLASS is None
