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

# Local
from caikit.runtime.service_generation.signature_parsing.docstrings import (
    _extract_nested_type,
    _get_docstring_type,
)
from sample_lib.data_model import SampleInputType
import sample_lib

def test_get_docstring_type():
    # TODO: fun edge case where producer word in description is found in types
    assert (
        _get_docstring_type(
            candidate_type_names=["sample_lib.data_model.SampleOutputType"],
        )
        == sample_lib.data_model.SampleOutputType
    )


def test_extract_nested_type():
    """
    Test that this function returns the type of List[T], or None if it's not a nested type
    """
    assert _extract_nested_type("List[str]") == List[str]
    assert _extract_nested_type("list(str)") == List[str]
    assert _extract_nested_type("List(SampleInputType)") == List[SampleInputType]
    assert _extract_nested_type("int") == None
