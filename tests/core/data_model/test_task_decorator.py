## Tests #######################################################################
# Third Party
import pytest

# Local
from caikit.core.data_model import task
from caikit.core.data_model.task_decorator import TaskBase
from sample_lib.data_model.sample import SampleInputType, SampleOutputType


def test_task_decorator_has_required_inputs_and_output_type():
    # TODO: whoops, the DataBase classes _don't_ have DataBase base class at static check time
    @task(
        required_inputs={"sample_input": SampleInputType}, output_type=SampleOutputType
    )
    class SampleTask(TaskBase):
        pass

    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}
    assert SampleTask.get_output_type() == SampleOutputType

    # assert Immutable properties
    SampleTask.required_inputs = {"sample_input": str}
    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}


def test_task_decorator_implodes_if_class_does_not_extend_task_base():
    with pytest.raises(TypeError):

        @task(
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask:
            pass
