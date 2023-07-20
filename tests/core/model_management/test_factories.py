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
"""Unit tests for ImportableFactory"""

# Third Party
import pytest

# Local
from caikit.core.model_management.factories import ImportableFactory
from caikit.core.model_management.local_model_finder import LocalModelFinder


def test_importable_factory_auto_register():
    """Test that a constructible gets auto-registered with a factory if given
    the import_class key
    """
    fact = ImportableFactory("TestFact")
    assert not fact._registered_types
    inst = fact.construct(
        {
            ImportableFactory.IMPORT_CLASS_KEY: "{}.{}".format(
                LocalModelFinder.__module__,
                LocalModelFinder.__name__,
            ),
            ImportableFactory.TYPE_KEY: LocalModelFinder.name,
        }
    )
    assert inst
    assert fact._registered_types


@pytest.mark.parametrize(
    "params",
    [
        (f"not.valid.{LocalModelFinder.__name__}", ValueError),
        (f"{LocalModelFinder.__module__}.FooBar", ValueError),
        (f"{LocalModelFinder.__module__}.alog", TypeError),
        (12345, TypeError),
    ],
)
def test_importable_factory_error_cases(params):
    """Make sure that all forms of bad value result in a ValueError or TypeError
    is raised
    """
    import_class_val, exc_type = params
    fact = ImportableFactory("TestFact")
    assert not fact._registered_types
    with pytest.raises(exc_type):
        fact.construct(
            {
                ImportableFactory.IMPORT_CLASS_KEY: import_class_val,
                ImportableFactory.TYPE_KEY: LocalModelFinder.name,
            }
        )
