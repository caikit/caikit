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
from functools import cached_property, partial
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Type, Union
import abc
import os
import sys

# Third Party
import grpc

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import _make_oneof_init, make_dataobject
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.core.exceptions import error_handler
from caikit.core.toolkit.factory import FactoryConstructible, ImportableFactory
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    FileReference,
    ListOfFileReferences,
    S3Files,
)
from caikit.runtime.names import get_service_package_name
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


## Plugin Bases ################################################################


class DataStreamSourcePlugin(FactoryConstructible):
    """A DataStreamSourcePlugin is a pluggable source that defines the shape of
    the data object needed as well as the code for accessing the data from some
    source type.
    """

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Construct with the basic factory constructible interface and store the
        args for use by the child
        """
        self._config = config
        self._instance_name = instance_name

    ## Abstract Interface ##

    @abc.abstractmethod
    def get_stream_message_type(self, element_type: type) -> Type[DataBase]:
        """Get the type of the dataobject class that will be used as the source
        information
        """

    @abc.abstractmethod
    def to_data_stream(
        self, source_message: Type[DataBase], element_type: type
    ) -> DataStream:
        """Convert an instance of the source message type into a DataStream"""

    @abc.abstractmethod
    def get_field_number(self) -> int:
        """Each plugin must define its field number which may be informed by
        self._config
        """

    ## Public Methods ##

    def get_field_name(self, element_type: type) -> str:
        """The name of the field that this plugin will use in the source oneof"""
        return self.get_stream_message_type(element_type).__name__.lower()

    ## Shared Impl ##

    @staticmethod
    def _to_element_type(element_type: type, raw_element: Any) -> Any:
        if issubclass(element_type, DataBase):
            # To allow for extra fields (e.g. in training data) that may not
            # be needed by the data objects, we ignore unknown fields
            return element_type.from_json(raw_element, ignore_unknown_fields=True)
        return raw_element

    @staticmethod
    def _to_element_partial(element_type: type) -> Callable:
        return partial(DataStreamSourcePlugin._to_element_type, element_type)


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
            return cls._load_from_file_without_extension(fname, element_type)

        full_fname = cls._get_resolved_source_path(fname)
        log.debug3("Pulling data stream from %s file [%s]", extension, full_fname)

        if not fname or not os.path.isfile(full_fname):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid {extension} data source file: {fname}",
            )
        to_element_type = cls._to_element_partial(element_type)
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
    def _load_from_file_without_extension(cls, fname, element_type: type) -> DataStream:
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
        to_element_type = cls._to_element_partial(element_type)
        log.debug3("Attempting to guess file type for file: %s", full_fname)
        for factory_method in (
            DataStream.from_json_array,
            DataStream.from_jsonl,
            DataStream.from_multipart_file,
            DataStream.from_header_csv,
        ):
            try:
                stream = factory_method(full_fname).map(to_element_type)
                # Iterate once and assume we have the correct file type if this
                # works
                stream.peek()
                return stream
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Catch any exception: it's hard to know which all could be
                # thrown by any of the formatters
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


## Source Plugins ##############################################################


class FileDataStreamSourcePlugin(FilePluginBase):
    """Plugin for a single file"""

    name = "FileData"

    def get_field_name(self, element_type: type) -> str:
        """Half-Backwards compatibility and half keep FileReference consistent
        with ListofFiles/Directory"""
        return "file"

    def get_stream_message_type(self, *_, **__) -> Type[DataBase]:
        return FileReference

    def to_data_stream(
        self, source_message: FileReference, element_type: type
    ) -> DataStream:
        return self._create_data_stream_from_file(
            fname=source_message.filename, element_type=element_type
        )

    def get_field_number(self) -> int:
        return 2


class ListOfFilesDataStreamSourcePlugin(FilePluginBase):
    """Plugin for a list of files"""

    name = "ListOfFiles"

    def get_field_name(self, element_type: type) -> str:
        """Half-Backwards compatibility and half keep ListOfFile consistent
        with File/Directory"""
        return "list_of_files"

    def get_stream_message_type(self, *_, **__) -> Type[DataBase]:
        return ListOfFileReferences

    def to_data_stream(
        self, source_message: ListOfFileReferences, element_type: type
    ) -> DataStream:
        data_stream_list = []
        for fname in source_message.files:
            data_stream_list.append(
                self._create_data_stream_from_file(
                    fname=fname, element_type=element_type
                )
            )

        return DataStream.chain(data_stream_list).flatten()

    def get_field_number(self) -> int:
        return 3


class DirectoryDataStreamSourcePlugin(FilePluginBase):
    """Plugin for a directory holding files"""

    name = "Directory"

    def get_stream_message_type(self, *_, **__) -> Type[DataBase]:
        return Directory

    def to_data_stream(
        self, source_message: Directory, element_type: type
    ) -> DataStream:
        dirname = source_message.dirname
        full_dirname = self._get_resolved_source_path(dirname)
        extension = source_message.extension or "json"
        if not dirname or not os.path.isdir(full_dirname):
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Invalid {extension} directory source file: {full_dirname}",
            )
        files_with_ext = list(glob(os.path.join(full_dirname, "*." + extension)))
        to_element_type = self._to_element_partial(element_type)
        # make sure at least 1 file with the given extension exists
        if len(files_with_ext) == 0:
            raise CaikitRuntimeException(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"directory {dirname} contains no source files with extension {extension}",
            )
        if extension == "json":
            return DataStream.from_json_collection(full_dirname, extension).map(
                to_element_type
            )
        if extension == "csv":
            return DataStream.from_csv_collection(full_dirname).map(to_element_type)
        if extension == "jsonl":
            return DataStream.from_jsonl_collection(full_dirname).map(to_element_type)
        raise CaikitRuntimeException(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"Extension not supported! {extension}",
        )

    def get_field_number(self) -> int:
        return 4


class JsonDataStreamSourcePlugin(DataStreamSourcePlugin):
    """This plugin is for inline data, elements are provided in a list.

    This plugin has instantiation logic: it needs the stream's element type so that it can
    generate a data model for List[element_type]"""

    name = "JsonData"

    # class-level cache required to avoid creating duplicate data model classes
    stream_source_type_cache: Dict[Type[DataBase], Type[DataBase]] = {}

    def get_stream_message_type(self, element_type: type) -> Type[DataBase]:
        stream_message_type = self.__class__.stream_source_type_cache.get(element_type)
        if stream_message_type:
            return stream_message_type

        package = get_service_package_name()
        cls_name = _make_data_stream_source_type_name(element_type)
        JsonData = make_dataobject(
            package=package,
            proto_name=f"{cls_name}JsonData",
            name="JsonData",
            attrs={"__qualname__": f"{cls_name}.JsonData"},
            annotations={"data": List[element_type]},
        )
        self.__class__.stream_source_type_cache[element_type] = JsonData
        return JsonData

    def to_data_stream(self, source_message: Type[DataBase], *_, **__) -> DataStream:
        """source_message should be of type self.get_stream_message_type
        So it _should_ contain an attribute named `data`, which is a list"""
        return DataStream.from_iterable(source_message.data)

    def get_field_number(self) -> int:
        return 1


class S3FilesDataStreamSourcePlugin(DataStreamSourcePlugin):
    """Unimplemented!"""

    name = "S3Files"

    def get_stream_message_type(self, *_, **__) -> Type[DataBase]:
        return S3Files

    def to_data_stream(self, *_, **__) -> DataStream:
        error(
            "<RUN80419785E>",
            NotImplementedError(
                "S3Files are not implemented as stream sources in this runtime."
            ),
        )

    def get_field_number(self) -> int:
        return 5


## DataStreamPluginFactory #####################################################


class DataStreamPluginFactory(ImportableFactory):
    """The DataStreamPluginFactory is responsible for holding a registry of
    plugin instances that will be used to create and manage data stream sources
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugins = None

    def get_plugins(
        self, plugins_config: Optional[aconfig.Config] = None
    ) -> List[DataStreamSourcePlugin]:
        """Builds the set of plugins to use for a data stream source of type element_type"""
        if self._plugins is None:
            self._plugins = []
            if plugins_config is None:
                plugins_config = get_config().data_streams.source_plugins
            for name, cfg in plugins_config.items():
                self._plugins.append(self.construct(cfg, name))

            # Make sure field numbers are unique
            field_numbers = [plugin.get_field_number() for plugin in self._plugins]
            duplicate_field_number_names = [
                plugin.name
                for plugin in self._plugins
                if field_numbers.count(plugin.get_field_number()) > 1
            ]
            error.value_check(
                "<RUN69189361E>",
                not duplicate_field_number_names,
                "Duplicate plugin field numbers found for plugins: {}",
                duplicate_field_number_names,
            )
        return self._plugins


# Single default instance
PluginFactory = DataStreamPluginFactory("DataStreamSource")
PluginFactory.register(JsonDataStreamSourcePlugin)
PluginFactory.register(FileDataStreamSourcePlugin)
PluginFactory.register(ListOfFilesDataStreamSourcePlugin)
PluginFactory.register(DirectoryDataStreamSourcePlugin)
PluginFactory.register(S3FilesDataStreamSourcePlugin)


## DataStreamSourceBase ########################################################


class DataStreamSourceBase(DataStream):
    """This base class acts as a sentinel so that dynamically generated data
    stream source classes can be identified programmatically.
    """

    def __init__(self):
        super().__init__(self._generator)

    def _generator(self):
        return self._stream.generator_func(
            *self._stream.generator_args, **self._stream.generator_kwargs
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
        setattr(self, new_inst.which_oneof("data_stream"), new_inst.data_stream)
        self.generator_func = self._generator
        self.generator_args = tuple()
        self.generator_kwargs = {}

    @cached_property
    def name_to_plugin_map(self):
        return {
            plugin.get_field_name(self.ELEMENT_TYPE): plugin for plugin in self.PLUGINS
        }

    @cached_property
    def _stream(self):
        """The internal _stream is cached here so that the result of calling to_data_stream can be
        re-read, rather than requiring to_data_stream to be invoked on every read through the
        stream"""
        return self.to_data_stream()

    # pylint: disable=too-many-return-statements
    def to_data_stream(self) -> DataStream:
        """Convert to the target data stream type based on the source type"""

        # Determine which of the value types is set
        set_field = None
        for field_name in self.get_proto_class().DESCRIPTOR.fields_by_name:
            if getattr(self, field_name) is not None:
                error.value_check(
                    "<RUN80421785E>",
                    set_field is None,
                    "Found DataStreamSource with multiple sources set: {} and {}",
                    set_field,
                    field_name,
                )
                error.value_check(
                    "<RUN80420785E>",
                    field_name in self.name_to_plugin_map,
                    "no data stream plugin found for field: {}",
                    field_name,
                )
                set_field = field_name

        # If no field is set, return an empty DataStream
        if set_field is None:
            log.debug3("Returning empty data stream")
            return DataStream.from_iterable([])

        # Get the correct plugin, and pass it the source field + the element
        # type to serialize to
        plugin = self.name_to_plugin_map[set_field]
        return plugin.to_data_stream(getattr(self, set_field), self.ELEMENT_TYPE)


## make_data_stream_source #####################################################


def make_data_stream_source(
    data_element_type: type,
    plugin_factory: DataStreamPluginFactory = PluginFactory,
    plugins_config: Optional[aconfig.Config] = None,
) -> Type[DataBase]:
    """Dynamically create a data stream source message type that supports
    pulling an iterable of the given type from all valid data stream sources
    """
    log.debug2("Looking for DataStreamSource[%s]", data_element_type)
    if data_element_type not in _DATA_STREAM_SOURCE_TYPES:
        cls_name = _make_data_stream_source_type_name(data_element_type)
        package = get_service_package_name()

        log.debug("Creating DataStreamSource[%s] -> %s", data_element_type, cls_name)

        # Get the required plugins
        plugins = plugin_factory.get_plugins(plugins_config)

        # Make sure there are no field name duplicates
        plug_to_name = {
            plugin: plugin.get_field_name(data_element_type) for plugin in plugins
        }
        all_field_names = list(plug_to_name.values())
        duplicates = {
            plugin.name: field_name
            for plugin, field_name in plug_to_name.items()
            if all_field_names.count(field_name) > 1
        }
        error.value_check(
            "<RUN66854455E>",
            not duplicates,
            "Duplicate plugin field names found for type {}: {}",
            data_element_type,
            duplicates,
        )

        # Create the outer class that encapsulates the Union (oneof) of the
        # various types of input sources

        # Determine the type stream message type for each source. This can
        # potentially be expensive, so we do it once
        stream_message_types = {
            plugin.name: plugin.get_stream_message_type(data_element_type)
            for plugin in plugins
        }

        # Build the type annotation for the data model
        # This describes a large oneof containing all the info from each data
        # stream source plugin
        annotation_list = [
            Annotated[
                stream_message_types[plugin.name],
                OneofField(plugin.get_field_name(data_element_type)),
                FieldNumber(plugin.get_field_number()),
            ]
            for plugin in plugins
        ]
        data_stream_type_union = Union[tuple(annotation_list)]

        # Create an attribute dictionary that will expose each of the source
        # types on this datastream class itself. E.g. if I have the `JsonData`
        # plugin enabled, this enables:
        # >>> make_data_stream_source(some_type).JsonData
        # to access the `JsonData` source message directly.
        type_attrs = {
            msg_type.__name__: msg_type for msg_type in stream_message_types.values()
        }

        data_object = make_dataobject(
            package=package,
            name=cls_name,
            bases=(DataStreamSourceBase,),
            attrs={"ELEMENT_TYPE": data_element_type, "PLUGINS": plugins, **type_attrs},
            annotations={"data_stream": data_stream_type_union},
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

        data_object.__init__ = __init__

        _DATA_STREAM_SOURCE_TYPES[data_element_type] = data_object

    # Return the global stream source object for this element type
    return _DATA_STREAM_SOURCE_TYPES[data_element_type]


def _make_data_stream_source_type_name(data_element_type: Type) -> str:
    """Make the name for data stream source class that wraps the given type"""
    element_name = data_element_type.__name__
    return "DataStreamSource{}".format(element_name[0].upper() + element_name[1:])
