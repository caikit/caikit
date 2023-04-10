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
from caikit.core.data_model.streams.validator import DataStreamValidator
from caikit.core.toolkit.errors import DataValidationError

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


class TestDataStreamValidator(TestCaseBase):
    def test_it_throws_validation_error_on_missing_dictionary_keys(self):
        dict_stream = DataStream.from_iterable(LIST_OF_DICTS)
        # `d` does not exist on the input stream
        validator = DataStreamValidator({"a": int, "b": int, "c": int, "d": int})
        dict_stream = validator.validate(dict_stream)
        # Since validation is lazy, we will not error until we evaluate the stream
        with self.assertRaises(DataValidationError):
            list(dict_stream)

    def test_it_throws_validation_error_on_input_lists_that_are_too_small(self):
        list_stream = DataStream.from_iterable(LIST_OF_LISTS)
        # input lists have only 2 elements each
        validator = DataStreamValidator({"a": int, "b": int, "c": int})
        list_stream = validator.validate(list_stream)
        with self.assertRaises(DataValidationError):
            list(list_stream)

    def test_it_throws_validation_error_on_input_lists_that_are_too_long(self):
        list_stream = DataStream.from_iterable(LIST_OF_LISTS)
        # input lists have 2 elements each
        validator = DataStreamValidator({"a": int})
        list_stream = validator.validate(list_stream)
        with self.assertRaises(DataValidationError):
            list(list_stream)

    def test_it_throws_validation_error_if_types_are_wrong(self):
        list_stream = DataStream.from_iterable(LIST_OF_LISTS)
        validator = DataStreamValidator({"a": int, "b": list})
        list_stream = validator.validate(list_stream)
        with self.assertRaises(DataValidationError):
            list(list_stream)

        dict_stream = DataStream.from_iterable(LIST_OF_DICTS)
        validator = DataStreamValidator({"a": int, "b": list, "c": str})
        dict_stream = validator.validate(dict_stream)
        with self.assertRaises(DataValidationError):
            list(dict_stream)
