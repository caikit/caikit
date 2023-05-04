"""
Dummy data model object for testing
"""

# Local
from caikit.core.data_model import DataObjectBase, dataobject


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
