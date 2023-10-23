"""
Dummy data model object for testing
"""
# Standard
from io import IOBase
from typing import Iterable, List, Optional
import json
import typing
import zipfile

# Local
from caikit.core import DataObjectBase, TaskBase, dataobject, task
from caikit.core.data_model import ProducerId
from caikit.interfaces.common.data_model import File, StrSequence


@dataobject(package="caikit_data_model.sample_lib")
class SampleInputType(DataObjectBase):
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""

    name: str


@dataobject(package="caikit_data_model.sample_lib")
class SampleListInputType(DataObjectBase):
    """A sample list input type for this library"""

    inputs: List[SampleInputType]


@dataobject(package="caikit_data_model.sample_lib")
class FileInputType(DataObjectBase):
    """A simple type for tasks that deal with file data"""

    file: File
    metadata: SampleInputType


@dataobject(package="caikit_data_model.sample_lib")
class SampleOutputType(DataObjectBase):
    """A simple return type for the `sample_task` task"""

    greeting: str


@dataobject(package="caikit_data_model.sample_lib")
class OtherOutputType(DataObjectBase):
    """A simple return type for the `other_task` task"""

    farewell: str
    producer_id: ProducerId


@dataobject(package="caikit_data_model.sample_lib")
class FileOutputType(DataObjectBase):
    """A simple type for tasks that deal with file data"""

    file: File
    metadata: SampleOutputType

    def to_file(self, file_obj: IOBase) -> Optional[File]:
        with zipfile.ZipFile(
            file_obj, compression=zipfile.ZIP_DEFLATED, mode="w"
        ) as zip_export:
            zip_export.writestr("metadata.json", json.dumps(self.metadata.to_dict()))
            zip_export.writestr(self.file.filename, self.file.data)

        return File(filename="result.zip", type="application/zip")


@dataobject(package="caikit_data_model.sample_lib")
class SampleTrainingType(DataObjectBase):
    """A sample `training data` type for the `sample_task` task."""

    number: int
    label: str


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
    unary_parameters={"input": FileInputType},
    unary_output_type=FileOutputType,
)
class FileTask(TaskBase):
    """A sample task for processing files"""


@task(
    required_parameters={
        "str_type": str,
        "list_str_type": List[str],
        "int_type": int,
        "list_int_type": List[int],
    },
    unary_output_type=StrSequence,
)
class PrimitiveTask(TaskBase):
    """A sample task for testing generic types"""


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
