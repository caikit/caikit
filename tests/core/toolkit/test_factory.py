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
Tests for the common factory implementation
"""

# Third Party
import pytest

# First Party
import aconfig

# Local
from caikit.core.toolkit import factory

## Helpers #####################################################################


class TypeOne(factory.FactoryConstructible):
    name = "one"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


class TypeTwo(factory.FactoryConstructible):
    name = "two"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


class TypeOtherOne(factory.FactoryConstructible):
    name = "one"

    def __init__(self, config, instance_name):
        self.config = config
        self.instance_name = instance_name


## Tests #######################################################################


def test_factory_happy_path():
    """Make sure that a factory works when used correctly"""
    fact = factory.Factory("Test")
    fact.register(TypeOne)
    fact.register(TypeTwo)

    inst_one = fact.construct({"type": "one"})
    assert isinstance(inst_one, TypeOne)
    assert isinstance(inst_one.config, aconfig.Config)

    inst_two = fact.construct({"type": "two", "config": {"foo": 1}})
    assert isinstance(inst_two, TypeTwo)
    assert isinstance(inst_two.config, aconfig.Config)
    assert inst_two.config.foo == 1


def test_factory_unregistered_error():
    """Make sure that asking to instantiate an unregistered type raises a
    ValueError
    """
    fact = factory.Factory("Test")
    with pytest.raises(ValueError):
        fact.construct({"type": "one"})


def test_factory_duplicate_registration():
    """Make sure that double registering a type is ok, but conflicting
    registration is not
    """
    fact = factory.Factory("Test")
    fact.register(TypeOne)
    fact.register(TypeOne)
    with pytest.raises(ValueError):
        fact.register(TypeOtherOne)


def test_factory_construct_with_instance_name():
    """Make sure that double registering a type is ok, but conflicting
    registration is not
    """
    fact = factory.Factory("Test")
    fact.register(TypeOne)

    inst_no_name = fact.construct({"type": "one"})
    assert isinstance(inst_no_name, TypeOne)
    assert inst_no_name.instance_name == TypeOne.name

    inst_name = "the-instance"
    inst_with_name = fact.construct({"type": "one"}, inst_name)
    assert isinstance(inst_with_name, TypeOne)
    assert inst_with_name.instance_name == inst_name
