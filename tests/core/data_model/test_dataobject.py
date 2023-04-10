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

"""Tests for the @dataobject decorator"""

# Standard
import json
import os
import tempfile

# Third Party
from google.protobuf import descriptor_pool, message
import pytest

# Local
from caikit.core import dataobject  # NOTE: Imported from the top to validate
from caikit.core.data_model import enums
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.dataobject import (
    _AUTO_GEN_PROTO_CLASSES,
    render_dataobject_protos,
)
from caikit.core.toolkit.isa import isprotobufenum

## Helpers #####################################################################


@pytest.fixture(autouse=True)
def temp_dpool():
    """Fixture to isolate the descriptor pool used in each test"""
    dpool = descriptor_pool.DescriptorPool()
    global_dpool = descriptor_pool._DEFAULT
    descriptor_pool._DEFAULT = dpool
    yield dpool
    # pylint: disable=duplicate-code
    descriptor_pool._DEFAULT = global_dpool


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset the global registry of generated protos"""
    _AUTO_GEN_PROTO_CLASSES.clear()


def check_field_type(proto_class, field_name, exp_type):
    field = proto_class.DESCRIPTOR.fields_by_name[field_name]
    return field.type == getattr(field, exp_type)


def check_field_label(proto_class, field_name, exp_label):
    field = proto_class.DESCRIPTOR.fields_by_name[field_name]
    return field.label == getattr(field, exp_label)


def check_field_message_type(proto_class, field_name, exp_msg_descriptor):
    field = proto_class.DESCRIPTOR.fields_by_name[field_name]
    return field.message_type is exp_msg_descriptor


def check_field_enum_type(proto_class, field_name, exp_enum_descriptor):
    field = proto_class.DESCRIPTOR.fields_by_name[field_name]
    return field.enum_type is exp_enum_descriptor


## Tests #######################################################################


def test_dataobject_native_types():
    """Make sure that a simple usage of dataobject for a flat object works with
    fields declared using  native python types works
    """

    @dataobject({"foo": str, "bar": int})
    class Foo:
        pass

    assert check_field_type(Foo.get_proto_class(), "foo", "TYPE_STRING")
    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_INT64")

    inst = Foo(foo="test", bar=1)
    assert inst.foo == "test"
    assert inst.bar == 1


def test_dataobject_jtd():
    """Make sure that a simple usage of dataobject using a full JTD schema works
    as expected
    """

    @dataobject({"properties": {"foo": {"type": "string"}}})
    class Foo:
        pass

    assert check_field_type(Foo.get_proto_class(), "foo", "TYPE_STRING")
    inst = Foo(foo="test")
    assert inst.foo == "test"


def test_dataobject_nested_objects():
    """Make sure that nested objects are handled correctly"""

    @dataobject({"bar": {"bat": str}})
    class Foo:
        pass

    assert hasattr(Foo, "Bar")
    bar_type = Foo.Bar
    assert issubclass(bar_type, DataBase)
    assert issubclass(bar_type.get_proto_class(), message.Message)
    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_MESSAGE")


def test_dataobject_nested_enums():
    """Make sure enums work as nested fields"""

    @dataobject(
        {
            "bar": {"enum": ["EXAM", "DRINKS"]},
            "bat": {"kind": {"enum": ["BASEBALL", "VAMPIRE"]}},
        }
    )
    class Foo:
        pass

    assert hasattr(Foo, "Bat")

    # Foo.Bar is a nested enum
    bar_type = Foo.Bar
    assert isinstance(bar_type, enums.EnumBase)
    assert getattr(enums, "Bar") is bar_type

    # Foo.Bat is a nested message
    bat_type = Foo.Bat
    assert issubclass(bat_type, DataBase)
    assert issubclass(bat_type.get_proto_class(), message.Message)
    assert check_field_type(Foo.get_proto_class(), "bat", "TYPE_MESSAGE")

    # Foo.Bat.Kind is a nested-nested enum
    kind_type = Foo.Bat.Kind
    assert isinstance(kind_type, enums.EnumBase)
    assert getattr(enums, "Kind") is kind_type


def test_dataobject_top_level_enums():
    """Make sure enums can be created"""

    @dataobject({"enum": ["EXAM", "DRINKS"]})
    class Foo:
        pass

    # Make sure it is in fact a class! This is surprisingly hard to achieve.
    assert isinstance(Foo, type)
    assert Foo.EXAM == 0
    assert Foo.DRINKS == 1
    dict_view = Foo.toDict()
    assert dict_view == {"EXAM": 0, "DRINKS": 1}

    # Make sure it's a singleton around the underlying EnumBase
    foo_inst = Foo()
    assert isinstance(foo_inst, enums.EnumBase)
    assert foo_inst.toDict() == dict_view
    assert Foo() is foo_inst

    # Make sure the underlying proto enum is accessible
    proto_enum = Foo._proto_enum
    assert isprotobufenum(proto_enum)

    # Make sure the str and repr methods of the Foo class show up like the dict
    assert str(Foo) == str(foo_inst)
    assert repr(Foo) == repr(foo_inst)


def test_dataobject_arrays():
    """Make sure arrays work as expected"""

    @dataobject({"bar": {"elements": str}})
    class Foo:
        pass

    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_STRING")
    assert check_field_label(Foo.get_proto_class(), "bar", "LABEL_REPEATED")


def test_dataobject_obj_refs():
    """Make sure that references to other data objects and enums work as
    expected
    """

    @dataobject({"enum": ["EXAM", "METAL"]})
    class BarEnum:
        pass

    @dataobject({"foo": str})
    class Foo:
        pass

    @dataobject({"foo": Foo, "bar": BarEnum, "bare_bar": BarEnum()})
    class FooBar:
        pass

    assert check_field_type(FooBar.get_proto_class(), "foo", "TYPE_MESSAGE")
    assert check_field_message_type(
        FooBar.get_proto_class(), "foo", Foo.get_proto_class().DESCRIPTOR
    )
    assert check_field_enum_type(
        FooBar.get_proto_class(), "bar", BarEnum._proto_enum.DESCRIPTOR
    )
    assert check_field_enum_type(
        FooBar.get_proto_class(), "bare_bar", BarEnum._proto_enum.DESCRIPTOR
    )


def test_dataobject_invalid_schema():
    """Make sure that a ValueError is raised on an invalid schema"""
    with pytest.raises(ValueError):
        # pylint: disable=unused-variable
        @dataobject("Foo")
        class Foo:
            pass


def test_dataobject_additional_methods():
    """Make sure that additional methods on wrapped classes (for messages and
    enums) are preserved
    """

    @dataobject({"enum": ["EXAM", "DRINKS"]})
    class Foo:
        @classmethod
        def is_exam(cls, val: int) -> bool:
            return val == cls.EXAM

    @dataobject({"bar": str})
    class Bar:
        def caps(self) -> str:
            return self.bar.upper()

    assert Foo.is_exam(Foo.EXAM)
    assert Bar("bat").caps() == "BAT"


def test_render_dataobject_protos():
    """Make sure that render_dataobject_protos correctly renders all generated
    protobufs to the target directory
    """

    @dataobject({"enum": ["EXAM", "METAL"]})
    class BarEnum:
        pass

    @dataobject({"foo": str})
    class Foo:
        pass

    @dataobject({"foo": Foo, "bar": BarEnum, "bare_bar": BarEnum()})
    class FooBar:
        pass

    with tempfile.TemporaryDirectory() as workdir:
        render_dataobject_protos(workdir)
        rendered_files = set(os.listdir(workdir))
        assert rendered_files == {
            BarEnum._proto_enum.DESCRIPTOR.file.name,
            Foo.get_proto_class().DESCRIPTOR.file.name,
            FooBar.get_proto_class().DESCRIPTOR.file.name,
        }


def test_dataobject_with_discriminator():
    """Make sure that adding a discriminator works as expected"""

    @dataobject(
        {
            "properties": {
                "data_stream": {
                    "discriminator": "data_reference_type",
                    "mapping": {
                        "Foo": {
                            "properties": {
                                "data": {
                                    "elements": {"type": "string"},
                                },
                            },
                        },
                        "Bar": {"properties": {"data": {"type": "string"}}},
                        "Baz": {
                            "properties": {
                                "data": {
                                    "elements": {"type": "string"},
                                },
                            },
                        },
                        "Bat": {
                            "properties": {
                                "data1": {"type": "string"},
                                "data2": {"type": "string"},
                            }
                        },
                    },
                }
            }
        }
    )
    class BazObj:
        pass

    # proto tests
    foo1 = BazObj(foo=BazObj.Foo(data=["hello"]))
    proto_repr_foo = foo1.to_proto()
    assert proto_repr_foo.foo.data == ["hello"]
    assert BazObj.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo

    bar1 = BazObj(foo=BazObj.Foo(data=["hello"]), bar=BazObj.Bar(data="world"))
    proto_repr_bar = bar1.to_proto()
    assert proto_repr_bar.bar.data == "world"

    # json tests
    foo1 = BazObj(foo=BazObj.Foo(data=["hello"]))
    json_repr_foo = foo1.to_json()
    assert json.loads(json_repr_foo) == {
        "foo": {"data": ["hello"]},
        "bar": None,
        "baz": None,
        "bat": None,
    }


def test_render_dataobject_protos_no_dir():
    """Make sure that render_dataobject_protos correctly renders all generated
    protobufs to the target directory and creates the target dir if it doesn't exist
    """

    @dataobject({"enum": ["EXAM", "METAL"]})
    class BarEnum:
        pass

    @dataobject({"foo": str})
    class Foo:
        pass

    @dataobject({"foo": Foo, "bar": BarEnum, "bare_bar": BarEnum()})
    class FooBar:
        pass

    with tempfile.TemporaryDirectory() as workdir:
        protos_dir_path = os.path.join(workdir, "protos")
        render_dataobject_protos(protos_dir_path)
        rendered_files = set(os.listdir(protos_dir_path))
        assert rendered_files == {
            BarEnum._proto_enum.DESCRIPTOR.file.name,
            Foo.get_proto_class().DESCRIPTOR.file.name,
            FooBar.get_proto_class().DESCRIPTOR.file.name,
        }


def test_dataobject_round_trip_json():
    """Make sure that a dataobject class can serialize to/from json"""

    @dataobject({"foo": str, "bar": int})
    class BazObj:
        pass

    baz1 = BazObj(foo="foo", bar=1)
    js_repr = baz1.to_json()
    assert json.loads(js_repr) == {"foo": "foo", "bar": 1}
    baz2 = BazObj.from_json(js_repr)
    assert baz2.to_json() == js_repr


def test_dataobject_round_trip_proto():
    """Make sure that a dataobject class can serialize to/from proto"""

    @dataobject({"foo": str, "bar": int})
    class BazObj:
        pass

    baz1 = BazObj(foo="foo", bar=1)
    proto_repr = baz1.to_proto()
    assert proto_repr.foo == "foo"
    assert proto_repr.bar == 1
    baz2 = BazObj.from_proto(proto_repr)
    assert baz2.to_proto() == proto_repr


def test_dir_on_instance():
    """This addresses a bug in how the @dataobject decorator binds in the
    DataBase base class which caused dir(x) to fail on instances of wrapped
    classes.
    """

    @dataobject({"foo": str})
    class BazObj:
        pass

    x = BazObj("foobar")
    dir(x)
