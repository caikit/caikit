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
# Standard
from typing import Type

# Third Party
import pytest

# Local
from caikit.core import ModuleBase, TaskBase, module
from caikit.interfaces.nlp import tasks as nlp_tasks
from tests.core.helpers import *

## Helpers #####################################################################


## Tests #######################################################################


class InvalidType:
    pass


@pytest.mark.parametrize(
    "task", (nlp_tasks.TextGenerationTask, nlp_tasks.TextGenerationStreamTask)
)
def test_tasks(reset_globals, task: Type[TaskBase]):
    """Common tests for all tasks"""
    # Only support single required param named "inputs"
    assert set(task.get_required_parameters().keys()) == {"inputs"}
    input_type = task.get_required_parameters()["inputs"]
    output_type = task.get_output_type()

    # Version with the right signature and nothing else
    @module(id="foo1", name="Foo", version="0.0.0", task=task)
    class Foo1(ModuleBase):
        def run(self, inputs: input_type) -> output_type:
            return output_type()

    # Version with the right signature plus extra args
    @module(id="foo2", name="Foo", version="0.0.0", task=task)
    class Foo2(ModuleBase):
        def run(
            self,
            inputs: input_type,
            workit: bool,
            makeit: bool,
            doit: bool,
        ) -> output_type:
            return output_type()

    # Version with missing required argument
    with pytest.raises(TypeError):

        @module(id="foo3", name="Foo", version="0.0.0", task=task)
        class Foo3(ModuleBase):
            def run(self, other_name: str) -> output_type:
                return output_type()

    # Version with bad required argument type
    with pytest.raises(TypeError):

        @module(id="foo4", name="Foo", version="0.0.0", task=task)
        class Foo4(ModuleBase):
            def run(self, inputs: InvalidType) -> output_type:
                return output_type()

    # Version with bad return type
    with pytest.raises(TypeError):

        @module(id="foo", name="Foo", version="0.0.0", task=task)
        class Foo(ModuleBase):
            def run(self, inputs: input_type) -> InvalidType:
                return "hi there"
