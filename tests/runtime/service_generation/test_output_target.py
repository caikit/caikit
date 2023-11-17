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
"""
Tests for the
- output_field data model for runtime training APIs
- model saver plugins that populate that field
"""
# Standard
from typing import Callable

# Third Party
import pytest

# Local
from caikit.core import DataObjectBase, ModuleBase, dataobject
from caikit.core.data_model import DataBase
from caikit.core.model_management.model_saver_base import (
    ModelSaveFunctor,
    ModelSaverBase,
)
from caikit.runtime.service_generation.output_target import make_output_target_message
from tests.conftest import temp_config
from tests.data_model_helpers import temp_dpool
import caikit

## Test model saver


@dataobject(package="test.foo")
class MockTargetType(DataObjectBase):
    url: str


class MockModelSaver(ModelSaverBase[MockTargetType]):

    name = "TEST"

    functor = "test_functor_please_ignore"

    def __init__(self, *args, **kwargs):
        pass

    def save_functor(self, output_target: MockTargetType) -> ModelSaveFunctor:
        """Not actually functional"""
        return self.functor


## Tests


def test_output_target_message_class():
    output_target_class = make_output_target_message()

    # Inherits from DataBase
    assert issubclass(output_target_class, DataBase)

    # Should have one `oneof`
    assert len(output_target_class.get_proto_class().DESCRIPTOR.oneofs) == 1
    oneof = output_target_class.get_proto_class().DESCRIPTOR.oneofs[0]

    # For now: only has the `local` field. Can add more test model savers for more
    assert len(oneof.fields) == 1

    assert hasattr(output_target_class, "local")


def test_output_targets_can_be_used_to_build_model_save_functors():
    output_target_class = make_output_target_message()

    target_field = output_target_class(local="foo")

    save_functor = caikit.core.make_save_functor(target_field.output_target)

    assert isinstance(save_functor, Callable)
    assert "LocalModelSaver" in str(type(save_functor.func.__self__))


def test_multiple_model_savers(reset_model_manager):
    with temp_dpool():
        with temp_config(
            config_overrides={
                "model_management": {
                    "savers": {
                        "test": {
                            "type": MockModelSaver.name,
                            "import_class": "tests.runtime.service_generation.test_output_target.MockModelSaver",
                        }
                    }
                }
            },
            merge_strategy="merge",
        ):
            # Note the "merge" above, local save should still be configured

            output_target_class = make_output_target_message()
            assert len(output_target_class.get_proto_class().DESCRIPTOR.oneofs) == 1
            oneof = output_target_class.get_proto_class().DESCRIPTOR.oneofs[0]

            # Now there should be both a `local` and `test` field
            assert len(oneof.fields) == 2

            assert hasattr(output_target_class, "local")
            assert hasattr(output_target_class, "test")

            # Assert we can set the new output target type
            target_message = output_target_class(
                output_target=MockTargetType(url="foo")
            )
            # And get a save_functor for it
            functor = caikit.make_save_functor(target_message.output_target)
            assert functor == MockModelSaver.functor
