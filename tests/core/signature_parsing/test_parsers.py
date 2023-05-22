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
from typing import List, Optional
from unittest.mock import patch
import inspect

# Third Party
from docstring_parser import ParseError

# Local
from caikit.core.signature_parsing.parsers import (
    _get_dm_type_from_name,
    _snake_to_camel,
    get_args_with_defaults,
    get_argument_types,
    get_output_type_name,
)
import caikit.core
import sample_lib

## Tests ########################################################################

# TODO: Add a test that looks for a type in caikit.interfaces.common.data_model


def test_snake_to_camel():
    assert _snake_to_camel("this_is_a_test_str") == "ThisIsATestStr"


def test_get_dm_type_from_name():
    assert _get_dm_type_from_name(None) == None

    assert (
        _get_dm_type_from_name("ProducerId")
        == caikit.core.data_model.producer.ProducerId
    )

    assert (
        _get_dm_type_from_name("SampleOutputType")
        == sample_lib.data_model.sample.SampleOutputType
    )

    assert _get_dm_type_from_name("NonExistentName") == None


def test_get_output_type_name():

    # Test that if there's no function signature, use docstring to deduct output type
    empty_sign = inspect.Signature(return_annotation=inspect.Signature.empty)
    assert (
        get_output_type_name(
            module_class=sample_lib.modules.sample_task.SampleModule,
            fn_signature=empty_sign,
            fn=sample_lib.modules.sample_task.SampleModule.run,
        )
        == sample_lib.data_model.SampleOutputType
    )

    # Test that we use type annotation to deduct output type
    inner_module_run_method_ptr = getattr(
        sample_lib.modules.sample_task.InnerModule, "run"
    )
    fn_sign = inspect.signature(inner_module_run_method_ptr)
    assert (
        get_output_type_name(
            module_class=sample_lib.modules.sample_task.InnerModule,
            fn_signature=fn_sign,
            fn=sample_lib.modules.sample_task.InnerModule.run,
        )
        == sample_lib.data_model.SampleOutputType
    )

    # Test that we use type annotation to deduct output type is return annotation is a string
    def _run(self, some_input: str) -> "InnerModule":
        pass

    fn_sign = inspect.signature(_run)
    assert (
        get_output_type_name(
            module_class=sample_lib.modules.sample_task.InnerModule,
            fn_signature=fn_sign,
            fn=sample_lib.modules.sample_task.InnerModule.run,
        )
        == sample_lib.modules.sample_task.InnerModule
    )

    # Test that we return None if type annotation as a string that doesn't match module class name
    def _run2(self, some_input: str) -> "AStringThatsNotInnerModule":
        pass

    fn_sign = inspect.signature(_run2)
    assert (
        get_output_type_name(
            module_class=sample_lib.modules.sample_task.InnerModule,
            fn_signature=fn_sign,
            fn=sample_lib.modules.sample_task.InnerModule.run,
        )
        == None
    )

    # User doesn't provide any type annotation or docstring, return None
    assert (
        get_output_type_name(
            module_class=sample_lib.modules.sample_task.InnerModule,
            fn_signature=empty_sign,
            fn=sample_lib.modules.sample_task.InnerModule.run,
        )
        == None
    )

    # Test that if a ParseError was raised with docstring.parsers, we have no idea the type and return None
    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert (
            get_output_type_name(
                module_class=sample_lib.modules.sample_task.InnerModule,
                fn_signature=empty_sign,
                fn=sample_lib.modules.sample_task.InnerModule.run,
            )
            == None
        )


def test_get_argument_types_with_real_module():
    """Quick check that we get the right type for our sample module"""
    assert (
        get_argument_types(sample_lib.modules.sample_task.SampleModule.run)[
            "sample_input"
        ]
        == sample_lib.data_model.SampleInputType
    )

    # Test that if a ParseError was raised with docstring.parsers, we could still parse from type annotation
    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert (
            get_argument_types(sample_lib.modules.sample_task.SampleModule.run)[
                "sample_input"
            ]
            == sample_lib.data_model.SampleInputType
        )

    # Test that if a ParseError was raised with docstring.parsers, but no type annotation provided, we cannot deduct type and return None
    def run_fn(some_input: str, some_input_2):
        pass

    with patch("docstring_parser.parse", side_effect=ParseError("mocked error")):
        assert get_argument_types(run_fn)["some_input"] == str
        assert get_argument_types(run_fn)["some_input_2"] == None


def test_optional_type_annotation():
    """Check that we keep the `Optional` wrapping on input types"""

    def _run(sample_input: Optional[int]):
        pass

    assert get_argument_types(_run)["sample_input"] == Optional[int]

    def _run2(sample_input: Optional[str]):
        """
        Args:
            sample_input: str
                optional string input
        """

    assert get_argument_types(_run2)["sample_input"] == Optional[str]


def test_get_argument_type_from_malformed_docstring():
    """This test tests docstring arg type parsing for docstrings in non-conforming styles
    where the actual type name is hidden in the description"""

    def _run(self, foo):
        """

        Args:
            foo: yadda yadda blah sample_lib.data_model.SampleInputType

        Returns:
            None
        """
        pass

    assert get_argument_types(_run)["foo"] == sample_lib.data_model.SampleInputType


def test_get_args_with_no_annotation():
    """Check that we get arguments with no type annotation supplied"""

    def _run(input_1="hello world"):
        pass

    assert get_argument_types(_run)["input_1"] == str

    def _run_with_docstring(input_1):
        """
        Args:
            input_1: str
                Optional str input
        """
        pass

    assert get_argument_types(_run_with_docstring)["input_1"] == Optional[str]

    def _run_with_known_dm_type(sample_input_type):
        pass

    assert (
        get_argument_types(_run_with_known_dm_type)["sample_input_type"]
        == sample_lib.data_model.SampleInputType
    )

    def _run_with_optional_known_dm_type(sample_input_type):
        """
        Args:
            sample_input_type: blah blah
                Optional input
        """
        pass

    assert (
        get_argument_types(_run_with_optional_known_dm_type)["sample_input_type"]
        == Optional[sample_lib.data_model.SampleInputType]
    )

    def _run_with_default_as_list_of_ints(a_list=[1, 2, 3]):
        pass

    assert get_argument_types(_run_with_default_as_list_of_ints)["a_list"] == List[int]

    def _run_with_default_as_list_of_multiple_types(a_list=[True, "hello", 1]):
        pass

    # We parse it as a random type
    assert get_argument_types(_run_with_default_as_list_of_multiple_types)[
        "a_list"
    ] in [List[str], List[bool], List[int]]


def test_get_args_with_defaults():
    """Check that we get arguments with any default value supplied"""

    def _run(
        a, b: bool, c: int = None, d: str = None, e: int = 5, f: float = 0.5, g=None
    ):
        pass

    assert get_args_with_defaults(_run) == {
        "c": None,
        "d": None,
        "e": 5,
        "f": 0.5,
        "g": None,
    }


def test_get_args_with_known_args():
    """Check that we get arguments with a known arg type supplied"""

    def _run(producer_id):
        pass

    assert (
        get_argument_types(_run)["producer_id"]
        == caikit.core.data_model.producer.ProducerId
    )
