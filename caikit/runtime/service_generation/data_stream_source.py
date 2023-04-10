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
from typing import Any, Optional, Type
import os
import sys

# Third Party
from google.protobuf.message import Message as ProtoMessageType
import grpc

# First Party
import alog

# Local
from caikit.core.data_model import dataobject
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import _NATIVE_TYPE_TO_JTD
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit

# This global holds the mapping of element types to their respective
# DataStreamSource wrappers so that the same message is not recreated
# unnecessarily
_DATA_STREAM_SOURCE_TYPES = {}

log = alog.use_channel("DSTRM-SRC")


class DataStreamSourceBase(DataStream):
    """This base class acts as a sentinel so that dynamically generated data
    stream source classes can be identified programmatically.
    """

    def __init__(self):
        """Validate oneof semantics"""
        super().__init__(lambda: self.to_data_stream().generator_func())
        self._set_fields = {
            f.name: getattr(self, f.name)
            for f in self.get_proto_class().DESCRIPTOR.oneofs[0].fields
            if getattr(self, f.name) is not None
        }
        if len(self._set_fields) > 1:
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Multiple oneof fields set: {self._set_fields.keys()}",
            )

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
        new_inst_dict = new_inst.__dict__
        self.__dict__.update(new_inst_dict)
        for field_descriptor in self.get_proto_class().DESCRIPTOR.oneofs[0].fields:
            field_name = field_descriptor.name
            setattr(self, field_name, getattr(new_inst, field_name))

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
            extension = self.directory.extension or "json"
            if not dirname or not os.path.isdir(dirname):
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid {extension} directory source file: {dirname}",
                )
            files_with_ext = list(glob(os.path.join(dirname, "*." + extension)))
            # make sure at least 1 file with the given extension exists
            if len(files_with_ext) == 0:
                raise CaikitRuntimeException(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"directory {dirname} contains no source files with extension {extension}",
                )
            if extension == "json":
                return DataStream.from_json_collection(dirname, extension).map(
                    self._to_element_type
                )
            if extension == "csv":
                return DataStream.from_csv_collection(dirname).map(
                    self._to_element_type
                )
            if extension == "jsonl":
                return DataStream.from_jsonl_collection(dirname).map(
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

        _, extension = os.path.splitext(fname)
        log.debug3("Pulling data stream from %s file [%s]", extension, fname)

        if not fname or not os.path.isfile(fname):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid {extension} data source file: {fname}",
            )
        if extension == ".json":
            return DataStream.from_json_array(fname).map(cls._to_element_type)
        if extension == ".csv":
            return DataStream.from_header_csv(fname).map(cls._to_element_type)
        if extension == ".jsonl":
            return DataStream.from_jsonl(fname).map(cls._to_element_type)
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


def make_data_stream_source(data_element_type: Type) -> Type[DataBase]:
    """Dynamically create a data stream source message type that supports
    pulling an iterable of the given type from all valid data stream sources
    """
    log.debug2("Looking for DataStreamSource[%s]", data_element_type)
    if data_element_type not in _DATA_STREAM_SOURCE_TYPES:
        cls_name = _make_data_stream_source_type_name(data_element_type)
        log.debug("Creating DataStreamSource[%s] -> %s", data_element_type, cls_name)
        element_type = (
            data_element_type.get_proto_class().DESCRIPTOR
            if isinstance(data_element_type, type)
            and issubclass(data_element_type, DataBase)
            else _NATIVE_TYPE_TO_JTD[data_element_type]
        )
        data_object = dataobject(
            schema={
                "properties": {
                    "data_stream": {
                        "discriminator": "data_reference_type",
                        "mapping": {
                            "JsonData": {
                                "properties": {
                                    "data": {
                                        "elements": {"type": element_type},
                                    },
                                },
                            },
                            "File": {"properties": {"filename": {"type": "string"}}},
                            "ListOfFiles": {
                                "properties": {
                                    "files": {
                                        "elements": {"type": "string"},
                                    },
                                },
                            },
                            "Directory": {
                                "properties": {
                                    "dirname": {"type": "string"},
                                    "extension": {"type": "string"},
                                }
                            },
                        },
                    }
                }
            },
            package="caikit_data_model.runtime",
        )(
            type(
                cls_name,
                (DataStreamSourceBase,),
                {"ELEMENT_TYPE": data_element_type},
            )
        )
        setattr(
            caikit.interfaces.common.data_model,
            cls_name,
            data_object,
        )
        setattr(
            sys.modules[__name__],
            cls_name,
            data_object,
        )

        # Add an init that sequences the initialization so that
        # DataStreamSourceBase is initialized after DataBase
        orig_init = data_object.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            DataStreamSourceBase.__init__(self)

        setattr(data_object, "__init__", __init__)

        _DATA_STREAM_SOURCE_TYPES[data_element_type] = data_object

    # Return the global stream source object for this element type
    return _DATA_STREAM_SOURCE_TYPES[data_element_type]


def get_data_stream_source(message: Any) -> Optional[Type[DataStreamSourceBase]]:
    """Get the data stream source from the given message if possible"""
    # If it's a protobuf message, alias to the corresponding DM class if
    # possible
    if isinstance(message, ProtoMessageType):
        # NOTE: This is the _very_ naive implementation and is potentially quite
        #   inefficient with the current from_proto implementation
        message = DataBase.get_class_for_proto(message).from_proto(message)
    if isinstance(message, DataStreamSourceBase):
        return message


def _make_data_stream_source_type_name(data_element_type: Type) -> str:
    """Make the name for data stream source class that wraps the given type"""
    element_name = data_element_type.__name__
    return "DataStreamSource{}".format(element_name[0].upper() + element_name[1:])
