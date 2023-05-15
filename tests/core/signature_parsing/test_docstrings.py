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
"""These are mostly some older tests that we wrote ad-hoc while throwing the inference proto-generation together

Coverage is probably not the best
"""
# Standard
from typing import List, Optional, Union
from unittest.mock import patch

# Third Party
from docstring_parser import ParseError

# Local
from caikit.core.signature_parsing.docstrings import (
    _extract_nested_type,
    _get_docstring_type,
    get_arg_type,
    get_return_type,
    is_optional,
)
from sample_lib.data_model import SampleInputType
import caikit
import sample_lib


def test_get_docstring_type():
    # TODO: fun edge case where producer word in description is found in types
    # works on a sample_lib type
    assert (
        _get_docstring_type(
            candidate_type_names=["sample_lib.data_model.SampleOutputType"],
        )
        == sample_lib.data_model.SampleOutputType
    )

    # works on nested type
    assert (
        _get_docstring_type(
            candidate_type_names=["List(int)"],
        )
        == List[int]
    )

    # works on a well known type
    assert (
        _get_docstring_type(
            candidate_type_names=["ProducerId"],
        )
        == caikit.core.data_model.ProducerId
    )

    # returns a Union on valid types
    assert (
        _get_docstring_type(
            candidate_type_names=[
                "SampleInputType",
                "sample_lib.data_model.SampleOutputType",
            ],
        )
        == Union[
            sample_lib.data_model.SampleOutputType,
            sample_lib.data_model.SampleInputType,
        ]
    )


def test_extract_nested_type():
    """
    Test that this function returns the type of List[T], or None if it's not a nested type
    """
    assert _extract_nested_type("List[str]") == List[str]
    assert _extract_nested_type("list(str)") == List[str]
    assert _extract_nested_type("List(SampleInputType)") == List[SampleInputType]
    assert _extract_nested_type("int") == None


def test_is_optional_works_on_corner_cases_docstrings():
    """
    Test that is_optional works on different corner cases of docstrings for an optional field
    """
    # test docstring with multiple params of the same name still works, but
    # evaluated whether "optional" on the first description
    def _fn(self, some_input: Optional[int]):
        """
        Args:
            some_input: yadda yadda blah int
                optional int for input
            some_input: I repeat myself

        Returns:
            None
        """
        pass

    assert is_optional(_fn, "some_input") is True

    # test if docstring_parser.parse throws an exception, we return False
    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert is_optional(_fn, "some_input") is False

    def _fn(input):
        pass

    # test is_optional works on empty docstring
    assert is_optional(_fn, "input") is False


def test_get_arg_type_works_on_corner_cases_docstrings():
    """
    Test that get_arg_type works on different corner cases of docstrings
    """
    # test docstring with multiple params of the same name still works, but
    # evaluated the type on the first description
    def _fn(self, some_input: str):
        """

        Args:
            some_input: yadda yadda blah int
                int for input
            some_input: I repeat myself but type str here
                str for input

        Returns:
            None
        """
        pass

    assert get_arg_type(_fn, "some_input") == int

    # test if docstring_parser.parse throws an exception, we return None
    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert get_arg_type(_fn, "some_input") is None

    # test get_arg_type works on empty docstring
    def _fn_no_docstring(input):
        pass

    assert get_arg_type(_fn_no_docstring, "input") == None


def test_get_return_type_corner_case_with_exception():
    def _fn():
        pass

    # test get_return_type works on empty docstring
    assert get_return_type(_fn) == None

    # test if docstring_parser.parse throws an exception, we swallow and return none
    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert get_return_type(_fn) is None
