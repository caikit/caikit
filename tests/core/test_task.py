## Tests #######################################################################
# Third Party
import pytest

# Local
from caikit.core import DomainBase, TaskBase, domain, task
from sample_lib.data_model.sample import SampleInputType, SampleOutputType
import caikit.core


@domain(input_types={SampleInputType})
class TestDomain(DomainBase):
    pass


def test_task_decorator_has_required_inputs_and_output_type():
    # TODO: whoops, the DataBase classes _don't_ have DataBase base class at static check time
    @task(
        domain=TestDomain,
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
    with pytest.raises(ValueError):

        @task(
            domain=TestDomain,
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask:
            pass


def test_task_decorator_validates_output_is_data_model():
    with pytest.raises(ValueError):

        @task(
            domain=TestDomain,
            required_inputs={"sample_input": SampleInputType},
            output_type=str,
        )
        class SampleTask(TaskBase):
            pass


def test_task_decorator_validates_input_is_in_domain():
    with pytest.raises(ValueError):
        # `str` is not in TestDomain.input_types
        @task(
            domain=TestDomain,
            required_inputs={"sample_input": str},
            output_type=SampleOutputType,
        )
        class SampleTask(TaskBase):
            pass


def test_task_decorator_validates_domain_is_a_domain():
    with pytest.raises(ValueError):
        # `str` is not in TestDomain.input_types
        @task(
            domain=str,
            required_inputs={"sample_input": SampleInputType},
            output_type=SampleOutputType,
        )
        class SampleTask(TaskBase):
            pass


def test_domain_has_input_type_set():
    @domain(input_types={int, float})
    class SampleDomain(DomainBase):
        pass

    assert SampleDomain.get_input_type_set() == {int, float}


def test_domain_validates_inputs_are_protoabletypes():
    with pytest.raises(ValueError):

        @domain(input_types={1, 2.3})
        class SampleDomain(DomainBase):
            pass

    with pytest.raises(ValueError):

        @domain(input_types={dict})
        class SampleDomain(DomainBase):
            pass

    with pytest.raises(ValueError):

        @domain(input_types={caikit.core.ModuleBase})
        class SampleDomain(DomainBase):
            pass
