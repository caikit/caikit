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
from caikit.core.data_model.json_dict import JsonDict
from caikit.core.exceptions.caikit_core_exception import (
    CaikitCoreException,
    CaikitCoreStatusCode,
)
from caikit.interfaces.common.data_model import File


@dataobject(package="caikit_data_model.sample_lib")
class SampleInputType(DataObjectBase):
    """A sample `domain primitive` input type for this library.
    The analog to a `Raw Document` for the `Natural Language Processing` domain."""

    name: str


@dataobject(package="caikit_data_model.sample_lib")
class SampleListInputType(DataObjectBase):
    """A sample list input type for this library"""

    inputs: List[SampleInputType]


# Test w/ just import and no dataobject
@dataobject(package="caikit_data_model.sample_lib")
class JsonDictInputType(DataObjectBase):
    """A sample `JsonDict` input type for this library.

    This exists because it impacts test_json_dict.py testing under proto3.
    This class is not used, but it affects the descriptor pool behavior.
    """

    jd: JsonDict


@dataobject(package="caikit_data_model.sample_lib")
class FileInputType(DataObjectBase):
    """A simple type for tasks that deal with file data"""

    file: File
    metadata: SampleInputType

    def __post_init__(self):
        if self.file.filename and ".exe" in self.file.filename:
            raise CaikitCoreException(
                status_code=CaikitCoreStatusCode.INVALID_ARGUMENT,
                message="Executables are not a supported File type",
            )


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
    metadata={"extra_openapi": {"description": "An Overridden task description"}},
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
