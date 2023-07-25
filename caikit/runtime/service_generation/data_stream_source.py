# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from glob import glob
from typing import Any, List, Type, Union
import os
import sys

# Third Party
import grpc

# First Party
from py_to_proto.dataclass_to_proto import Annotated, OneofField
import alog

# Local
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import _make_oneof_init, make_dataobject
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.core.toolkit.errors import error_handler
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    File,
    ListOfFiles,
    S3Files,
)
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit

# import common explicitly since this module needs it
import caikit.interfaces.common

# This global holds the mapping of element types to their respective
# DataStreamSource wrappers so that the same message is not recreated
# unnecessarily
_DATA_STREAM_SOURCE_TYPES = {}

log = alog.use_channel("DSTRM-SRC")
error = error_handler.get(log)


class DataStreamSourceBase(DataStream):
    """This base class acts as a sentinel so that dynamically generated data
    stream source classes can be identified programmatically.
    """

    def __init__(self):
        super().__init__(self._generator)

    def _generator(self):
        stream = self.to_data_stream()
        return stream.generator_func(*stream.generator_args, **stream.generator_kwargs)

    def __getstate__(self) -> bytes:
        """A DataStreamSource is pickled by serializing its source
        representation. This is particularly useful when sharing data streams
        across subprocesses to run training in an isolated process.
        """
        return self.to_binary_buffer()

    def __setstate__(self, pickle_bytes: bytes):
        """Unpickling a DataStreamSource basically involves unpacking the
        serialized source representation. The catch is that the oneof is
        represented strangely in __dict__, so we need to explicitly set all
        oneof members.
        """
        new_inst = self.__class__.from_binary_buffer(pickle_bytes)
        setattr(self, new_inst.which_oneof("data_stream"), new_inst.data_stream)

    # pylint: disable=too-many-return-statements
    def to_data_stream(self) -> DataStream:
        """Convert to the target data stream type based on the source type"""

        # Determine which of the value types is set
        set_field = None
        for field_name in self.get_proto_class().DESCRIPTOR.fields_by_name:
            if getattr(self, field_name) is not None:
                assert (
                    set_field is None
                ), "Found DataStreamSource with multiple sources set"
                set_field = field_name

        # If no field is set, return an empty DataStream
        if set_field is None:
            log.debug3("Returning empty data stream")
            return DataStream.from_iterable([])

        # If a S3 pointer is given, raise not implemented
        if set_field == "s3files":
            error(
                "<COR80419785E>",
                NotImplementedError(
                    "S3Files are not implemented as stream sources in this runtime."
                ),
            )

        # If jsondata, pull from the data elements directly
        if set_field == "jsondata":
            log.debug3("Pulling data stream from inline json")
            return DataStream.from_iterable(self.jsondata.data)

        # If jsonfile, attempt to read the file and pull in the data from there
        if set_field == "file":
            return self._create_data_stream_from_file(fname=self.file.filename)

        # If list of files, attempt to read all files and combine datastreams from all
        if set_field == "listoffiles":
            # combined list of data streams that we will chain and send back
            data_stream_list = []
            for fname in self.listoffiles.files:
                data_stream_list.append(self._create_data_stream_from_file(fname=fname))

            return DataStream.chain(data_stream_list).flatten()

        # If directory, attempt to read an element from each file in the dir
        if set_field == "directory":
            dirname = self.directory.dirname
            full_dirname = self._get_resolved_source_path(dirname)
            extension = self.directory.extension or "json"
            if not dirname or not os.path.isdir(full_dirname):
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid {extension} directory source file: {full_dirname}",
                )
            files_with_ext = list(glob(os.path.join(full_dirname, "*." + extension)))
            # make sure at least 1 file with the given extension exists
            if len(files_with_ext) == 0:
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"directory {dirname} contains no source files with extension {extension}",
                )
            if extension == "json":
                return DataStream.from_json_collection(full_dirname, extension).map(
                    self._to_element_type
                )
            if extension == "csv":
                return DataStream.from_csv_collection(full_dirname).map(
                    self._to_element_type
                )
            if extension == "jsonl":
                return DataStream.from_jsonl_collection(full_dirname).map(
                    self._to_element_type
                )
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Extension not supported! {extension}",
            )

    @classmethod
    def _create_data_stream_from_file(cls, fname: str) -> DataStream:
        """Create a data stream object by deducing file extension
        and reading the file accordingly"""

        full_fname = cls._get_resolved_source_path(fname)
        _, extension = os.path.splitext(fname)
        log.debug3("Pulling data stream from %s file [%s]", extension, full_fname)

        if not fname or not os.path.isfile(full_fname):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid {extension} data source file: {fname}",
            )
        if extension == ".json":
            return DataStream.from_json_array(full_fname).map(cls._to_element_type)
        if extension == ".csv":
            return DataStream.from_header_csv(full_fname).map(cls._to_element_type)
        if extension == ".jsonl":
            return DataStream.from_jsonl(full_fname).map(cls._to_element_type)
        raise CaikitRuntimeException(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Extension not supported! {extension}",
        )

    @classmethod
    def _to_element_type(cls, raw_element: Any) -> "ElementType":
        """Stream adapter to adapt from the raw json object data representations
        to the underlying data objects
        """
        if issubclass(cls.ELEMENT_TYPE, DataBase):
            return cls.ELEMENT_TYPE.from_json(raw_element)
        return raw_element

    @staticmethod
    def _get_resolved_source_path(input_path: str) -> str:
        """Get a fully resolved path, including any shared prefix"""
        # Get any configured prefix
        source_pfx = caikit.get_config().data_streams.file_source_base
        # If a prefix is configured, use it, otherwise return the path as is
        # NOTE: os.path.join will ignore the prefix if input_path is absolute
        return os.path.join(source_pfx, input_path) if source_pfx else input_path


def make_data_stream_source(data_element_type: Type) -> Type[DataBase]:
    """Dynamically create a data stream source message type that supports
    pulling an iterable of the given type from all valid data stream sources
    """
    log.debug2("Looking for DataStreamSource[%s]", data_element_type)
    if data_element_type not in _DATA_STREAM_SOURCE_TYPES:
        cls_name = _make_data_stream_source_type_name(data_element_type)
        log.debug("Creating DataStreamSource[%s] -> %s", data_element_type, cls_name)

        # Set up the "sub class." In python, this is the same as creating a
        # standalone class with a __qualname__ that nests it under a parent
        # class. We do this outside the declaration of the parent class so that
        # this class can be referenced within the Union annotation for the
        # outer class itself. This class needs to be created dynamically because
        # it encapsulates type information about the elements of the data stream.
        package = "caikit_data_model.runtime"
        JsonData = make_dataobject(
            package=package,
            proto_name=f"{cls_name}JsonData",
            name="JsonData",
            attrs={"__qualname__": f"{cls_name}.JsonData"},
            annotations={"data": List[data_element_type]},
        )

        # Create the outer class that encapsulates the Union (oneof) or the
        # various types of input sources
        data_object = make_dataobject(
            package=package,
            name=cls_name,
            bases=(DataStreamSourceBase,),
            attrs={
                "ELEMENT_TYPE": data_element_type,
                JsonData.__name__: JsonData,
                File.__name__: File,
                ListOfFiles.__name__: ListOfFiles,
                Directory.__name__: Directory,
                S3Files.__name__: S3Files,
            },
            annotations={
                "data_stream": Union[
                    Annotated[JsonData, OneofField(JsonData.__name__.lower())],
                    Annotated[File, OneofField(File.__name__.lower())],
                    Annotated[ListOfFiles, OneofField(ListOfFiles.__name__.lower())],
                    Annotated[Directory, OneofField(Directory.__name__.lower())],
                    Annotated[S3Files, OneofField(S3Files.__name__.lower())],
                ],
            },
        )

        # Add this data stream source to the common data model and the module
        # where it was declared
        setattr(
            caikit.interfaces.common.data_model,
            cls_name,
            data_object,
        )
        setattr(
            sys.modules[data_object.__module__],
            cls_name,
            data_object,
        )

        # Add an init that sequences the initialization so that
        # DataStreamSourceBase is initialized after DataBase
        orig_init = _make_oneof_init(data_object)

        def __init__(self, *args, **kwargs):
            try:
                orig_init(self, *args, **kwargs)
            except TypeError as err:
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT, str(err)
                ) from err
            DataStreamSourceBase.__init__(self)

        setattr(data_object, "__init__", __init__)

        _DATA_STREAM_SOURCE_TYPES[data_element_type] = data_object

    # Return the global stream source object for this element type
    return _DATA_STREAM_SOURCE_TYPES[data_element_type]


def _make_data_stream_source_type_name(data_element_type: Type) -> str:
    """Make the name for data stream source class that wraps the given type"""
    element_name = data_element_type.__name__
    return "DataStreamSource{}".format(element_name[0].upper() + element_name[1:])
