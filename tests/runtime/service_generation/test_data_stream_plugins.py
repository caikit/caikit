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
"""
Tests for the plugin mechanism for data stream sources
"""
# Standard
from unittest import mock

# Third Party
import pytest

# First Party
import aconfig

# Local
from caikit.core.data_model.streams.data_stream import DataStream
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    FileReference,
    ListOfFileReferences,
)
from caikit.runtime.service_generation.data_stream_source import (
    DataStreamPluginFactory,
    DataStreamSourcePlugin,
    DirectoryDataStreamSourcePlugin,
    FileDataStreamSourcePlugin,
    JsonDataStreamSourcePlugin,
    ListOfFilesDataStreamSourcePlugin,
    S3FilesDataStreamSourcePlugin,
    make_data_stream_source,
)
from tests.data_model_helpers import reset_global_protobuf_registry, temp_dpool
from tests.runtime.service_generation.test_data_stream_source import (
    validate_data_stream,
)
import sample_lib


def test_file_plugin(sample_json_file):
    # Sample json file has SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = FileReference(filename=sample_json_file)
    stream = FileDataStreamSourcePlugin({}, "").to_data_stream(
        source_message, element_type
    )
    validate_data_stream(stream, 2, element_type)


def test_directory_plugin(sample_json_collection):
    # Sample json collection has SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = Directory(dirname=sample_json_collection, extension="json")
    stream = DirectoryDataStreamSourcePlugin({}, "").to_data_stream(
        source_message, element_type
    )
    validate_data_stream(stream, 3, element_type)


def test_list_of_files_plugin(sample_json_file, sample_csv_file, sample_jsonl_file):
    # Samples all have SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = ListOfFileReferences(
        files=[sample_json_file, sample_jsonl_file, sample_csv_file]
    )
    stream = ListOfFilesDataStreamSourcePlugin({}, "").to_data_stream(
        source_message, element_type
    )
    validate_data_stream(stream, 9, element_type)


def test_json_data_plugin():
    element_type = sample_lib.data_model.SampleTrainingType
    foo = sample_lib.data_model.SampleTrainingType(number=1, label="foo")
    data = [foo] * 5

    # This plugin is actually initialized so that it can build the source message type
    plugin = JsonDataStreamSourcePlugin({}, "")
    source_message_type = plugin.get_stream_message_type(element_type=element_type)
    source_messsage = source_message_type(data=data)
    stream = plugin.to_data_stream(
        source_message=source_messsage, element_type=element_type
    )

    validate_data_stream(stream, 5, element_type)


def test_json_data_plugin_can_be_reinitialized():
    element_type = sample_lib.data_model.SampleTrainingType
    strm_type = JsonDataStreamSourcePlugin({}, "inst1").get_stream_message_type(
        element_type=element_type
    )
    strm_type2 = JsonDataStreamSourcePlugin({}, "inst2").get_stream_message_type(
        element_type=element_type
    )

    assert strm_type is strm_type2


## PluginFactory ###############################################################


@pytest.fixture
def reset_stream_source_types():
    with reset_global_protobuf_registry():
        with mock.patch(
            "caikit.runtime.service_generation.data_stream_source._DATA_STREAM_SOURCE_TYPES",
            {},
        ):
            with temp_dpool(True):
                yield


@pytest.fixture
def plugin_factory():
    fct = DataStreamPluginFactory("TestFactory")
    fct.register(JsonDataStreamSourcePlugin)
    fct.register(FileDataStreamSourcePlugin)
    fct.register(ListOfFilesDataStreamSourcePlugin)
    fct.register(DirectoryDataStreamSourcePlugin)
    fct.register(S3FilesDataStreamSourcePlugin)
    yield fct


class SampleTrainingPlugin(DataStreamSourcePlugin):
    """Dummy plugin that can be registered with a factory"""

    name = "TEST"

    def get_stream_message_type(self, *_, **__):
        return sample_lib.data_model.SampleTrainingType

    def to_data_stream(self, source_message, *_, **__):
        count = self._config.get("count", 5)
        DataStream.from_iterable(count * [source_message])

    def get_field_number(self) -> int:
        return self._config.get("field_number", 42)

    def get_field_name(self, element_type):
        return self._config.get("field_name", super().get_field_name(element_type))


def test_plugin_factory_is_configurable(plugin_factory):
    """Make sure new plugins can be registered and configured"""
    cfg = aconfig.Config(
        {"foo": {"type": SampleTrainingPlugin.name}},
        override_env_vars=False,
    )
    # Before registration, it's a ValueError
    with pytest.raises(ValueError):
        plugin_factory.construct(cfg.foo, "foo")

    # After registration, it's valid
    plugin_factory.register(SampleTrainingPlugin)
    plug = plugin_factory.construct(cfg.foo, "foo")
    assert isinstance(plug, SampleTrainingPlugin)

    # Make sure there's an instance in get_plugins
    all_plugins = plugin_factory.get_plugins(cfg)
    assert any(isinstance(plug, SampleTrainingPlugin) for plug in all_plugins)


def test_plugin_factory_reuses_plugins(plugin_factory):
    """Make sure that the list of plugins gets reused"""
    plugins = plugin_factory.get_plugins()
    assert plugin_factory.get_plugins() is plugins


def test_no_duplicate_field_numbers(plugin_factory):
    """Make sure that duplicate field numbers cause a ValueError"""
    plugin_factory.register(SampleTrainingPlugin)
    cfg = aconfig.Config(
        {
            "foo": {"type": SampleTrainingPlugin.name},
            "bar": {"type": SampleTrainingPlugin.name},
        },
        override_env_vars=False,
    )
    with pytest.raises(ValueError):
        plugin_factory.get_plugins(cfg)


def test_no_duplicate_field_names(plugin_factory, reset_stream_source_types):
    """Make sure that new data stream sources cannot be created with duplicate
    field names
    """
    plugin_factory.register(SampleTrainingPlugin)
    cfg = aconfig.Config(
        {
            "foo": {"type": SampleTrainingPlugin.name, "config": {"field_number": 42}},
            "bar": {"type": SampleTrainingPlugin.name, "config": {"field_number": 43}},
        },
        override_env_vars=False,
    )
    with pytest.raises(ValueError):
        make_data_stream_source(int, plugin_factory, cfg)


def test_multiple_instances(plugin_factory, reset_stream_source_types):
    """Make sure that multiple instances of a type with different configs can be
    instantiated as long as field numbers and names are unique
    """
    plugin_factory.register(SampleTrainingPlugin)
    cfg = aconfig.Config(
        {
            "foo": {
                "type": SampleTrainingPlugin.name,
                "config": {"field_number": 42, "field_name": "foo"},
            },
            "bar": {
                "type": SampleTrainingPlugin.name,
                "config": {"field_number": 43, "field_name": "bar"},
            },
        },
        override_env_vars=False,
    )
    plugin_factory.get_plugins(cfg)

    class LocalFoobar:
        pass

    strm_src_cls = make_data_stream_source(LocalFoobar, plugin_factory, cfg)
    assert strm_src_cls.ELEMENT_TYPE is LocalFoobar
    assert set(strm_src_cls.get_proto_class().DESCRIPTOR.fields_by_name.keys()) == {
        "foo",
        "bar",
    }
