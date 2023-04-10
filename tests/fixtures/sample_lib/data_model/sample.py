"""
Dummy data model object for testing
"""

# Local
import caikit.core


@caikit.core.dataobject({"name": str}, package="caikit_data_model.sample_lib")
class SampleInputType:
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""


@caikit.core.dataobject(
    schema={"greeting": str}, package="caikit_data_model.sample_lib"
)
class SampleOutputType:
    """A simple return type for the `sample_task` task"""


@caikit.core.dataobject(
    schema={"farewell": str}, package="caikit_data_model.sample_lib"
)
class OtherOutputType:
    """A simple return type for the `other_task` task"""


@caikit.core.dataobject({"number": int}, package="caikit_data_model.sample_lib")
class SampleTrainingType:
    """A sample `training data` type for the `sample_task` task."""
