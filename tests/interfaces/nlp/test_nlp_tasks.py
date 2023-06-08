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
"""Tests for the NLP task definitions"""

# Third Party
import pytest

# Local
from caikit.core import ModuleBase, module
from caikit.interfaces.nlp import data_model as nlp_dm
from caikit.interfaces.nlp import tasks as nlp_tasks
from tests.core.helpers import *


def test_text_generation_valid_no_extra_arguments(reset_globals):
    """Make sure that the text generation task binds to a module with the
    required I/O types and no extra arguments
    """

    @module(id="foo", name="Foo", version="0.0.0", task=nlp_tasks.TextGenerationTask)
    class Foo(ModuleBase):
        def run(self, inputs: str) -> nlp_dm.GeneratedResult:
            return nlp_dm.GeneratedResult()


def test_text_generation_valid_with_extra_arguments(reset_globals):
    """Make sure that the text generation task binds to a module with the
    required I/O types plus some extra input arguments
    """

    @module(id="foo", name="Foo", version="0.0.0", task=nlp_tasks.TextGenerationTask)
    class Foo(ModuleBase):
        def run(
            self,
            inputs: str,
            workit: bool,
            makeit: bool,
            doit: bool,
        ) -> nlp_dm.GeneratedResult:
            return nlp_dm.GeneratedResult()


def test_text_generation_missing_arguments(reset_globals):
    """Make sure that the text generation task fails to bind without the
    required input argument
    """

    with pytest.raises(TypeError):

        @module(
            id="foo", name="Foo", version="0.0.0", task=nlp_tasks.TextGenerationTask
        )
        class Foo(ModuleBase):
            def run(self, text: str) -> nlp_dm.GeneratedResult:
                return nlp_dm.GeneratedResult()


def test_text_generation_bad_argument_type(reset_globals):
    """Make sure that the text generation task fails to bind when the required
    input argument has the wrong type
    """

    with pytest.raises(TypeError):

        @module(
            id="foo", name="Foo", version="0.0.0", task=nlp_tasks.TextGenerationTask
        )
        class Foo(ModuleBase):
            def run(self, inputs: int) -> nlp_dm.GeneratedResult:
                return nlp_dm.GeneratedResult()


def test_text_generation_invalid_return_type(reset_globals):
    """Make sure that the text generation task fails to bind without the
    required return type
    """

    with pytest.raises(TypeError):

        @module(
            id="foo", name="Foo", version="0.0.0", task=nlp_tasks.TextGenerationTask
        )
        class Foo(ModuleBase):
            def run(self, inputs: str) -> str:
                return "hi there"
