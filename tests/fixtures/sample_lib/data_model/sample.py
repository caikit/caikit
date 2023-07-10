"""
Dummy data model object for testing
"""
# Standard
from typing import Iterable
import typing

# Local
from caikit.core import DataObjectBase, TaskBase, dataobject, task


@dataobject(package="caikit_data_model.sample_lib")
class SampleInputType(DataObjectBase):
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""

    name: str


@dataobject(package="caikit_data_model.sample_lib")
class SampleOutputType(DataObjectBase):
    """A simple return type for the `sample_task` task"""

    greeting: str


@dataobject(package="caikit_data_model.sample_lib")
class OtherOutputType(DataObjectBase):
    """A simple return type for the `other_task` task"""

    farewell: str


@dataobject(package="caikit_data_model.sample_lib")
class SampleTrainingType(DataObjectBase):
    """A sample `training data` type for the `sample_task` task."""

    number: int


@task(
    unary_parameters={"sample_input": SampleInputType},
    streaming_parameters={"sample_inputs": Iterable[SampleInputType]},
    unary_output_type=SampleOutputType,
    streaming_output_type=Iterable[SampleOutputType],
)
class SampleTask(TaskBase):
    """A sample `task` for our test models"""


@task(
    unary_parameters={"sample_input": SampleInputType},
    unary_output_type=OtherOutputType,
)
class OtherTask(TaskBase):
    """Another sample `task` for our test models"""


@task(
    streaming_parameters={"lats": Iterable[float], "lons": Iterable[float]},
    streaming_output_type=Iterable[SampleOutputType],
)
class GeoSpatialTask(TaskBase):
    """A task that flexes streaming capabilities"""


# NB: Backwards compatibility test
@task(
    required_parameters={"sample_input": SampleInputType},
    output_type=typing.Iterable[SampleOutputType],
)
class StreamingTask(TaskBase):
    """A streaming version of a task"""
