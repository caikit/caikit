"""
Dummy data model object for testing
"""
# Standard
from dataclasses import dataclass

# Local
import caikit.core


@caikit.core.dataobject(package="caikit_data_model.sample_lib")
@dataclass
class SampleInputType:
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""

    name: str


@caikit.core.dataobject(package="caikit_data_model.sample_lib")
@dataclass
class SampleOutputType:
    """A simple return type for the `sample_task` task"""

    greeting: str


@caikit.core.dataobject(package="caikit_data_model.sample_lib")
@dataclass
class OtherOutputType:
    """A simple return type for the `other_task` task"""

    farewell: str


@caikit.core.dataobject(package="caikit_data_model.sample_lib")
@dataclass
class SampleTrainingType:
    """A sample `training data` type for the `sample_task` task."""

    number: int
