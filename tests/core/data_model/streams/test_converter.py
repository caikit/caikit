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
from caikit.core.data_model import DataStream
from caikit.core.data_model.streams.converter import DataStreamConverter

# Unit Test Infrastructure
from tests.base import TestCaseBase

LIST_OF_DICTS = [
    {"a": 1, "b": 2, "c": 3},
    {"a": 4, "b": 5, "c": 6},
    {"a": 7, "b": 8, "c": 9},
]

LIST_OF_LISTS = [
    [1, 3],
    [4, 6],
    [7, 9],
]


class TestDataStreamConverter(TestCaseBase):
    def test_errors_with_unsupported_target_type(self):
        with self.assertRaises(ValueError):
            DataStreamConverter(tuple, ["a", "c"])

    def test_it_is_okay_with_empty_streams_to_list(self):
        empty_stream = DataStream.from_iterable([])
        to_list_converter = DataStreamConverter(list, ["a", "c"])
        stream = to_list_converter.convert(empty_stream)
        self.assertEqual([], list(stream))

    def test_it_is_okay_with_empty_streams_to_dict(self):
        empty_stream = DataStream.from_iterable([])
        to_dict_converter = DataStreamConverter(dict, ["a", "c"])
        stream = to_dict_converter.convert(empty_stream)
        self.assertEqual([], list(stream))

    def test_can_convert_dict_stream_to_list_stream(self):
        dict_stream = DataStream.from_iterable(LIST_OF_DICTS)
        converter = DataStreamConverter(list, ["a", "c"])
        list_stream = converter.convert(dict_stream)
        expected_lists = [
            [1, 3],
            [4, 6],
            [7, 9],
        ]
        self.assertEqual(expected_lists, list(list_stream))

    def test_can_convert_list_stream_to_dict_stream(self):
        list_stream = DataStream.from_iterable(LIST_OF_LISTS)
        converter = DataStreamConverter(dict, ["a", "b"])
        dict_stream = converter.convert(list_stream)
        expected_dict = [
            {"a": 1, "b": 3},
            {"a": 4, "b": 6},
            {"a": 7, "b": 9},
        ]
        self.assertEqual(expected_dict, list(dict_stream))

    def test_does_not_convert_list_stream_already_of_target_type(self):
        # this only actually works if the input and target lists are the same length
        list_stream = DataStream.from_iterable(LIST_OF_LISTS)
        converter = DataStreamConverter(list, ["1", "3"])
        converted_stream = converter.convert(list_stream)
        self.assertEqual(list(converted_stream), list(list_stream))

    def test_does_not_convert_dict_stream_already_of_target_type(self):
        # this only actually works if the input dicts have all the expected keys
        dict_stream = DataStream.from_iterable(LIST_OF_DICTS)
        converter = DataStreamConverter(dict, ["b", "a", "c"])
        converted_stream = converter.convert(dict_stream)
        self.assertEqual(list(converted_stream), list(dict_stream))
