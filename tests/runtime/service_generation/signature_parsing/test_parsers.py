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
from typing import List
import inspect

# Third Party
from sample_lib.data_model import SampleInputType
import sample_lib

# Local
from caikit.runtime.service_generation.signature_parsing.docstrings import (
    _extract_nested_type,
    _get_docstring_type,
)
from caikit.runtime.service_generation.signature_parsing.parsers import (
    _get_dm_type_from_name,
    _snake_to_camel,
    get_argument_type,
    get_output_type_name,
)
import caikit.core

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


def test_get_docstring_type():
    # TODO: fun edge case where producer word in description is found in types
    assert (
        _get_docstring_type(
            module_class=sample_lib.blocks.sample_task.SampleBlock,
            candidate_type_names=["sample_lib.data_model.SampleOutputType"],
        )
        == sample_lib.data_model.SampleOutputType
    )


def test_get_argument_type_from_docstring():
    """This tests docstring parsing only (note annotation=inspect.Parameter.empty)
    The 'SampleBlock' has correctly formatted docstrings in Google style
    """
    assert (
        get_argument_type(
            arg=inspect.Parameter(
                name="sample_input",
                kind=inspect.Parameter.POSITIONAL_ONLY,
                annotation=inspect.Parameter.empty,
            ),
            module_class=sample_lib.blocks.sample_task.SampleBlock,
            module_method=sample_lib.blocks.sample_task.SampleBlock.run,
        )
        == sample_lib.data_model.SampleInputType
    )


def test_get_argument_type_from_malformed_docstring():
    """This test tests docstring arg type parsing for docstrings in non-conforming styles
    where the actual type name is hidden in the description"""

    class TestClass:
        def run(self, foo):
            """

            Args:
                foo: yadda yadda blah sample_lib.data_model.SampleInputType

            Returns:
                None
            """
            pass

    assert (
        get_argument_type(
            arg=inspect.Parameter(
                name="foo",
                kind=inspect.Parameter.POSITIONAL_ONLY,
                annotation=inspect.Parameter.empty,
            ),
            module_class=TestClass,
            module_method=TestClass.run,
        )
        == sample_lib.data_model.SampleInputType
    )


def test_get_output_type_name():
    run_sign = inspect.Signature(return_annotation=inspect.Signature.empty)
    assert (
        get_output_type_name(
            module_class=sample_lib.blocks.sample_task.SampleBlock,
            fn_signature=run_sign,
            fn=getattr(sample_lib.blocks.sample_task.SampleBlock, "run"),
        )
        == sample_lib.data_model.SampleOutputType
    )


def test_extract_nested_type():
    """
    Test that this function returns the type of List[T], or None if it's not a nested type
    """
    # TODO: I don't really get why the block ref has to be passed here :hmm:
    assert (
        _extract_nested_type(sample_lib.blocks.sample_task.SampleBlock, "List[str]")
        == List[str]
    )
    assert (
        _extract_nested_type(sample_lib.blocks.sample_task.SampleBlock, "list(str)")
        == List[str]
    )
    assert (
        _extract_nested_type(
            sample_lib.blocks.sample_task.SampleBlock, "List(SampleInputType)"
        )
        == List[SampleInputType]
    )
    assert (
        _extract_nested_type(sample_lib.blocks.sample_task.SampleBlock, "int") == None
    )
