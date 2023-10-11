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
# Local
from caikit.interfaces.common.data_model.stream_sources import (
    Directory,
    File,
    ListOfFiles,
)
from caikit.runtime.service_generation.data_stream_source import (
    DirectoryDataStreamSourcePlugin,
    FileDataStreamSourcePlugin,
    JsonDataStreamSourcePlugin,
    ListOfFilesDataStreamSourcePlugin,
)
from tests.runtime.service_generation.test_data_stream_source import (
    validate_data_stream,
)
import sample_lib


def test_file_plugin(sample_json_file):
    # Sample json file has SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = File(filename=sample_json_file)

    stream = FileDataStreamSourcePlugin.to_data_stream(source_message, element_type)

    validate_data_stream(stream, 2, element_type)


def test_directory_plugin(sample_json_collection):
    # Sample json collection has SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = Directory(dirname=sample_json_collection, extension="json")

    stream = DirectoryDataStreamSourcePlugin.to_data_stream(
        source_message, element_type
    )

    validate_data_stream(stream, 3, element_type)


def test_list_of_files_plugin(sample_json_file, sample_csv_file, sample_jsonl_file):
    # Samples all have SampleTrainingType data
    element_type = sample_lib.data_model.SampleTrainingType
    source_message = ListOfFiles(
        files=[sample_json_file, sample_jsonl_file, sample_csv_file]
    )

    stream = ListOfFilesDataStreamSourcePlugin.to_data_stream(
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
