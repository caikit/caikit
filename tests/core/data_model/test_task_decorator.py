## Tests #######################################################################

import pytest
import caikit
from caikit.core.data_model import task
from tests.fixtures.sample_lib.data_model.sample import SampleInputType, SampleOutputType


def test_task_decorator_something():
    @task(required_inputs={"sample_input": SampleInputType}, output_type=SampleOutputType)
    class SampleTask:
        pass

    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}
    assert SampleTask.get_output_type() == SampleOutputType

    # assert Immutable properties
    SampleTask.required_inputs = {"sample_input": str}
    assert SampleTask.get_required_inputs() == {"sample_input": SampleInputType}

    # assert 