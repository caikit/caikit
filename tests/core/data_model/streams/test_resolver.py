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
import csv
import json
import os
import tempfile
import uuid

# Local
from caikit.core.data_model import DataStream
from caikit.core.data_model.streams.resolver import DataStreamResolver
from caikit.core.toolkit.errors import DataValidationError
from tests.base import TestCaseBase


class TestDataStreamResolver(TestCaseBase):
    resolver = DataStreamResolver(list, {"foo": str, "bar": str})

    @staticmethod
    def _randomData():
        lists = [[str(uuid.uuid4()), str(uuid.uuid4())] for _ in range(100)]
        dicts = [{"foo": each_list[0], "bar": each_list[1]} for each_list in lists]
        return lists, dicts

    def test_it_passes_back_a_data_stream(self):
        lists, dicts = self._randomData()
        data_stream = DataStream.from_iterable(lists)
        self.assertEqual(
            list(data_stream), list(self.resolver.as_data_stream(data_stream))
        )

    def test_it_can_convert_streams(self):
        lists, dicts = self._randomData()
        result = self.resolver.as_data_stream(DataStream.from_iterable(dicts))
        self.assertEqual(lists, list(result))

    def test_it_can_handle_json_files(self):
        lists, dicts = self._randomData()
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "foo.json")
            with open(filename, "w") as f:
                json.dump(dicts, f)

            self.assertEqual(lists, list(self.resolver.as_data_stream(filename)))

    def test_it_can_handle_csv_files(self):
        lists, dicts = self._randomData()
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "foo.csv")
            with open(filename, "w") as f:
                csv.writer(f).writerows(lists)
            self.assertEqual(lists, list(self.resolver.as_data_stream(filename)))

    def test_it_can_handle_csv_formats_with_variable_rows(self):
        lists = [
            ["some text", "label 1"],
            ["some more text", "label 1", "label 2", "label 3"],
        ]
        expected_stream = [
            {"text": "some text", "labels": ["label 1"]},
            {"text": "some more text", "labels": ["label 1", "label 2", "label 3"]},
        ]
        resolver = DataStreamResolver(dict, {"text": str, "labels": list})

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "foo.csv")
            with open(filename, "w") as f:
                csv.writer(f).writerows(lists)
            self.assertEqual(expected_stream, list(resolver.as_data_stream(filename)))

    def test_will_throw_data_validation_error_on_bad_data(self):
        sad_resolver = DataStreamResolver(list, {"foo": str, "baz": str})
        lists, dicts = self._randomData()
        result = sad_resolver.as_data_stream(DataStream.from_iterable(dicts))
        with self.assertRaises(DataValidationError):
            list(result)
