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
"""Unit tests for common enums functionality"""

# Standard
from enum import Enum
import json

# Third Party
from munch import Munch
import pytest
import yaml

# First Party
import py_to_proto

# Local
from caikit.core.data_model import enums

## Helpers #####################################################################


@pytest.fixture(autouse=True)
def reset_enums():
    """Helper to purge global attributes from the enums"""
    old_attrs = [attr for attr in dir(enums)]
    yield
    added_attrs = [attr for attr in dir(enums) if attr not in old_attrs]
    for added_attr in added_attrs:
        delattr(enums, added_attr)


## Tests #######################################################################


def test_import_enum_no_enum_class():
    """Make sure that importing an enum without an Enum class correctly creates
    one on the fly
    """

    class MyEnum(Enum):
        FOO = 1
        BAR = 2

    enum_name = MyEnum.__name__
    proto_enum = py_to_proto.descriptor_to_message_class(
        py_to_proto.dataclass_to_proto("foo", MyEnum)
    )

    assert not hasattr(enums, enum_name)
    enums.import_enum(proto_enum)
    assert hasattr(enums, enum_name)
    assert hasattr(enums, enum_name + "Rev")
    imported_enum = getattr(enums, enum_name)
    assert imported_enum.__name__ == enum_name
    assert issubclass(imported_enum, Enum)
    assert imported_enum.FOO.value == MyEnum.FOO.value
    assert imported_enum.BAR.value == MyEnum.BAR.value


def test_import_enum_with_enum_class():
    """Make sure that importing an enum with an Enum class does not create one
    on the fly
    """

    class MyEnum(Enum):
        FOO = 1
        BAR = 2

    enum_name = MyEnum.__name__
    proto_enum = py_to_proto.descriptor_to_message_class(
        py_to_proto.dataclass_to_proto("foo", MyEnum)
    )

    assert not hasattr(enums, enum_name)
    enums.import_enum(proto_enum, MyEnum)
    assert hasattr(enums, enum_name)
    assert hasattr(enums, enum_name + "Rev")
    imported_enum = getattr(enums, enum_name)
    assert imported_enum is MyEnum


def test_import_enum_extra_serialization():
    """Make sure that an imported enum gets correctly decorated with extra
    serialization methods
    """

    class MyEnum(Enum):
        FOO = 1
        BAR = 2

    enum_name = MyEnum.__name__
    proto_enum = py_to_proto.descriptor_to_message_class(
        py_to_proto.dataclass_to_proto("foo", MyEnum)
    )

    assert not hasattr(enums, enum_name)
    enums.import_enum(proto_enum, MyEnum)
    assert hasattr(enums, enum_name)
    assert hasattr(enums, enum_name + "Rev")

    # to_dict
    exp_dict = {"FOO": 1, "BAR": 2}
    assert MyEnum.to_dict() == exp_dict

    # to_munch
    munch_repr = MyEnum.to_munch()
    assert munch_repr == Munch(exp_dict)
    assert json.loads(munch_repr.toJSON()) == exp_dict
    assert yaml.safe_load(munch_repr.toYAML()) == exp_dict


def test_import_enum_rev():
    """Make sure that the reverse enum is created correctly"""

    class MyEnum(Enum):
        FOO = 1
        BAR = 2

    enum_name = MyEnum.__name__
    proto_enum = py_to_proto.descriptor_to_message_class(
        py_to_proto.dataclass_to_proto("foo", MyEnum)
    )
    assert not hasattr(enums, enum_name)
    enums.import_enum(proto_enum, MyEnum)
    assert hasattr(enums, enum_name)
    assert hasattr(enums, enum_name + "Rev")
    rev = getattr(enums, enum_name + "Rev")
    assert rev[1] == "FOO"
    assert rev[2] == "BAR"


def test_import_enum_invalid_proto_class():
    """Make sure that import_enum checks the proto class"""
    with pytest.raises(AttributeError):
        enums.import_enum("not a proto enum")


def test_import_enums_global():
    """Test that the import_enums utility pulls in all enums from the
    'current_globals' dict
    """

    class MyEnum(Enum):
        FOO = 1
        BAR = 2

    enum_name = MyEnum.__name__
    rev_name = enum_name + "Rev"
    proto_enum = py_to_proto.descriptor_to_message_class(
        py_to_proto.dataclass_to_proto("foo", MyEnum)
    )

    assert not hasattr(enums, enum_name)
    current_globals = {
        "protobufs": Munch(
            {
                "all_enum_names": [enum_name],
                enum_name: proto_enum,
            }
        )
    }
    enums.import_enums(current_globals)
    assert hasattr(enums, enum_name)
    assert hasattr(enums, rev_name)
    assert current_globals[enum_name] is getattr(enums, enum_name)
    assert current_globals[rev_name] is getattr(enums, rev_name)
