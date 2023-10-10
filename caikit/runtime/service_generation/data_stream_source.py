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
from functools import partial
from glob import glob
from typing import Any, List, Optional, Type, Union
import abc
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
from caikit.core.exceptions import error_handler
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    File,
    ListOfFiles,
    S3Files,
)
from caikit.runtime.service_generation.proto_package import get_runtime_service_package
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


## Scratch!! ###################################################################


class DataStreamSourcePlugin(abc.ABC):
    """A DataStreamSourcePlugin is a pluggable source that defines the shape of
    the data object needed as well as the code for accessing the data from some
    source type.
    """

    ## Abstract Interface ##

    @abc.abstractmethod
    def get_stream_message_type(self) -> Type[DataBase]:
        """Get the type of the dataobject class that will be used as the source
        information
        """

    @abc.abstractmethod
    def to_data_stream(
        self, source_message: Type[DataBase], element_type: type
    ) -> DataStream:
        """Convert an instance of the source message type into a DataStream"""

    ## Public Methods ##

    def get_field_name(self) -> str:
        """Common logic for creating the name of the plugin's field in the oneof
        for the top-level source message
        """
        return self.get_stream_message_type().__name__.lower()

    def get_field_number(self) -> Optional[int]:
        """Allow plugins to return a static field number"""

    ## Shared Impl ##

    @staticmethod
    def _to_element_type(element_type: type, raw_element: Any) -> Any:
        if issubclass(element_type, DataBase):
            # To allow for extra fields (e.g. in training data) that may not
            # be needed by the data objects, we ignore unknown fields
            return element_type.from_json(raw_element, ignore_unknown_fields=True)
        return raw_element


class FilePluginBase(DataStreamSourcePlugin):
    """Intermediate base class for file-based plugins with helper utilities"""

    @classmethod
    def _create_data_stream_from_file(
        cls, fname: str, element_type: type
    ) -> DataStream:
        """Create a data stream object by deducing file extension
        and reading the file accordingly"""

        _, extension = os.path.splitext(fname)
        if not extension:
            return cls._load_from_file_without_extension(fname)

        full_fname = cls._get_resolved_source_path(fname)
        log.debug3("Pulling data stream from %s file [%s]", extension, full_fname)

        if not fname or not os.path.isfile(full_fname):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid {extension} data source file: {fname}",
            )
        to_element_type = partial(cls._to_element_type, element_type)
        if extension == ".json":
            stream = DataStream.from_json_array(full_fname).map(to_element_type)
            # Iterate once to make sure this is a json array
            stream.peek()
            return stream
        if extension == ".csv":
            return DataStream.from_header_csv(full_fname).map(to_element_type)
        if extension == ".jsonl":
            return DataStream.from_jsonl(full_fname).map(to_element_type)
        raise CaikitRuntimeException(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Extension not supported! {extension}",
        )

    @classmethod
    def _load_from_file_without_extension(cls, fname) -> DataStream:
        """Similar to _create_data_stream_from_file, but we don't have a file extension to work
        with. Attempt to create a data stream using one of a few well-known formats.
        🌶🌶🌶️ on ordering here:
        File formats are loosely arranged in order of least-to-most-sketchy format validation.
        1. .json/.jsonl are pretty straightforward
        2. multipart files are a little iffy- the content-type header line can be omitted, in
            which case we check for a `--` string and roll our own boundary parser. This could
            cause problems in the future for multi-yaml files that begin with `---`
        3. CSV support simply assumes the first line of the file has the column headers, and may
            confidently return a stream even if that's not the case.
        """
        full_fname = cls._get_resolved_source_path(fname)
        log.debug3("Attempting to guess file type for file: %s", full_fname)
        for factory_method in (
            DataStream.from_json_array,
            DataStream.from_jsonl,
            DataStream.from_multipart_file,
            DataStream.from_header_csv,
        ):
            try:
                stream = factory_method(full_fname).map(cls._to_element_type)
                # Iterate once and assume we have the correct file type if this works
                stream.peek()
                return stream
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Catch any exception: it's hard to know which all could be thrown by any of the
                # formatters
                log.debug3(
                    "Failed to load file %s using data stream factory method %s: %s",
                    full_fname,
                    factory_method,
                    e,
                    exc_info=True,
                )
        raise CaikitRuntimeException(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Could not load input file with no extension: {full_fname}",
        )

    @staticmethod
    def _get_resolved_source_path(input_path: str) -> str:
        """Get a fully resolved path, including any shared prefix"""
        # Get any configured prefix
        source_pfx = caikit.get_config().data_streams.file_source_base
        # If a prefix is configured, use it, otherwise return the path as is
        # NOTE: os.path.join will ignore the prefix if input_path is absolute
        return os.path.join(source_pfx, input_path) if source_pfx else input_path


class FileDataStreamSourcePlugin(FilePluginBase):
    """Plugin for a single file"""

    @staticmethod
    def get_stream_message_type() -> Type[DataBase]:
        return File

    @classmethod
    def to_data_stream(cls, source_message: File, element_type: type) -> DataStream:
        return cls._create_data_stream_from_file(
            fname=source_message.filename, element_type=element_type
        )


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
        self.generator_func = self._generator
        self.generator_args = tuple()
        self.generator_kwargs = {}

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

    # @classmethod
    # def _create_data_stream_from_file(cls, fname: str) -> DataStream:
    #     """Create a data stream object by deducing file extension
    #     and reading the file accordingly"""

    #     _, extension = os.path.splitext(fname)
    #     if not extension:
    #         return cls._load_from_file_without_extension(fname)

    #     full_fname = cls._get_resolved_source_path(fname)
    #     log.debug3("Pulling data stream from %s file [%s]", extension, full_fname)

    #     if not fname or not os.path.isfile(full_fname):
    #         raise CaikitRuntimeException(
    #             grpc.StatusCode.INVALID_ARGUMENT,
    #             f"Invalid {extension} data source file: {fname}",
    #         )
    #     if extension == ".json":
    #         stream = DataStream.from_json_array(full_fname).map(cls._to_element_type)
    #         # Iterate once to make sure this is a json array
    #         stream.peek()
    #         return stream
    #     if extension == ".csv":
    #         return DataStream.from_header_csv(full_fname).map(cls._to_element_type)
    #     if extension == ".jsonl":
    #         return DataStream.from_jsonl(full_fname).map(cls._to_element_type)
    #     raise CaikitRuntimeException(
    #         grpc.StatusCode.INVALID_ARGUMENT,
    #         f"Extension not supported! {extension}",
    #     )

    # @classmethod
    # def _load_from_file_without_extension(cls, fname) -> DataStream:
    #     """Similar to _create_data_stream_from_file, but we don't have a file extension to work
    #     with. Attempt to create a data stream using one of a few well-known formats.
    #     🌶🌶🌶️ on ordering here:
    #     File formats are loosely arranged in order of least-to-most-sketchy format validation.
    #     1. .json/.jsonl are pretty straightforward
    #     2. multipart files are a little iffy- the content-type header line can be omitted, in
    #         which case we check for a `--` string and roll our own boundary parser. This could
    #         cause problems in the future for multi-yaml files that begin with `---`
    #     3. CSV support simply assumes the first line of the file has the column headers, and may
    #         confidently return a stream even if that's not the case.
    #     """
    #     full_fname = cls._get_resolved_source_path(fname)
    #     log.debug3("Attempting to guess file type for file: %s", full_fname)
    #     for factory_method in (
    #         DataStream.from_json_array,
    #         DataStream.from_jsonl,
    #         DataStream.from_multipart_file,
    #         DataStream.from_header_csv,
    #     ):
    #         try:
    #             stream = factory_method(full_fname).map(cls._to_element_type)
    #             # Iterate once and assume we have the correct file type if this works
    #             stream.peek()
    #             return stream
    #         except Exception as e:  # pylint: disable=broad-exception-caught
    #             # Catch any exception: it's hard to know which all could be thrown by any of the
    #             # formatters
    #             log.debug3(
    #                 "Failed to load file %s using data stream factory method %s: %s",
    #                 full_fname,
    #                 factory_method,
    #                 e,
    #                 exc_info=True,
    #             )
    #     raise CaikitRuntimeException(
    #         grpc.StatusCode.INVALID_ARGUMENT,
    #         f"Could not load input file with no extension: {full_fname}",
    #     )

    # @classmethod
    # def _to_element_type(cls, raw_element: Any) -> "ElementType":
    #     """Stream adapter to adapt from the raw json object data representations
    #     to the underlying data objects
    #     """
    #     if issubclass(cls.ELEMENT_TYPE, DataBase):
    #         # To allow for extra fields (e.g. in training data) that may not
    #         # be needed by the data objects, we ignore unknown fields
    #         return cls.ELEMENT_TYPE.from_json(raw_element, ignore_unknown_fields=True)
    #     return raw_element

    # @staticmethod
    # def _get_resolved_source_path(input_path: str) -> str:
    #     """Get a fully resolved path, including any shared prefix"""
    #     # Get any configured prefix
    #     source_pfx = caikit.get_config().data_streams.file_source_base
    #     # If a prefix is configured, use it, otherwise return the path as is
    #     # NOTE: os.path.join will ignore the prefix if input_path is absolute
    #     return os.path.join(source_pfx, input_path) if source_pfx else input_path


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
        package = get_runtime_service_package()
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
