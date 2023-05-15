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
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import List, Optional, Union
import copy
import json
import os
import tempfile

# Third Party
from google.protobuf import descriptor_pool, message
import pytest

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField

# Local
from caikit.core import (  # NOTE: Imported from the top to validate
    DataObjectBase,
    dataobject,
)
from caikit.core.data_model import enums
from caikit.core.data_model.base import DataBase, _DataBaseMetaClass
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
def reset_global_protobuf_registry():
    """Reset the global registry of generated protos"""
    prev_auto_gen_proto_classes = copy.copy(_AUTO_GEN_PROTO_CLASSES)
    prev_class_registry = copy.copy(_DataBaseMetaClass.class_registry)
    _AUTO_GEN_PROTO_CLASSES.clear()
    yield
    _AUTO_GEN_PROTO_CLASSES.extend(prev_auto_gen_proto_classes)
    _DataBaseMetaClass.class_registry.clear()
    _DataBaseMetaClass.class_registry.update(prev_class_registry)


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

    @dataobject
    class Foo(DataObjectBase):
        foo: str
        bar: int

    assert check_field_type(Foo.get_proto_class(), "foo", "TYPE_STRING")
    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_INT64")

    inst = Foo(foo="test", bar=1)
    assert inst.foo == "test"
    assert inst.bar == 1
    inst = Foo()
    assert inst.foo is None
    assert inst.bar is None


def test_dataobject_jtd():
    """Make sure that a simple usage of dataobject using a full JTD schema works
    as expected
    """

    @dataobject(schema={"properties": {"foo": {"type": "string"}}})
    class Foo(DataObjectBase):
        foo: str

    assert check_field_type(Foo.get_proto_class(), "foo", "TYPE_STRING")
    inst = Foo(foo="test")
    assert inst.foo == "test"
    inst = Foo()
    assert inst.foo is None


def test_dataobject_nested_objects():
    """Make sure that nested objects are handled correctly"""

    @dataobject
    class Foo(DataObjectBase):
        @dataobject
        class Bar(DataObjectBase):
            bat: str

        bar: Bar

    assert hasattr(Foo, "Bar")
    bar_type = Foo.Bar
    assert issubclass(bar_type, DataBase)
    assert issubclass(bar_type.get_proto_class(), message.Message)
    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_MESSAGE")


def test_dataobject_nested_enums():
    """Make sure enums work as nested fields"""

    @dataobject
    class Foo(DataObjectBase):
        @dataobject
        class Bar(Enum):
            EXAM = 0
            DRINKS = 1

        @dataobject
        class Bat(DataObjectBase):
            @dataobject
            class Kind(Enum):
                BASEBALL = 0
                VAMPIRE = 1

            kind: Kind

        bar: Bar
        bat: Bat

    assert hasattr(Foo, "Bat")

    # Foo.Bar is a nested enum
    bar_type = Foo.Bar
    assert issubclass(bar_type, Enum)
    assert getattr(enums, "Bar") is bar_type

    # Foo.Bat is a nested message
    bat_type = Foo.Bat
    assert issubclass(bat_type, DataBase)
    assert issubclass(bat_type.get_proto_class(), message.Message)
    assert check_field_type(Foo.get_proto_class(), "bat", "TYPE_MESSAGE")

    # Foo.Bat.Kind is a nested-nested enum
    kind_type = Foo.Bat.Kind
    assert issubclass(kind_type, Enum)
    assert getattr(enums, "Kind") is kind_type


def test_dataobject_top_level_enums():
    """Make sure enums can be created"""

    @dataobject
    class Foo(Enum):
        EXAM = 0
        DRINKS = 1

    # Make sure it is in fact a class! This is surprisingly hard to achieve.
    assert isinstance(Foo, type)
    assert Foo.EXAM.value == 0
    assert Foo.DRINKS.value == 1
    dict_view = Foo.to_dict()
    assert dict_view == {"EXAM": 0, "DRINKS": 1}

    # Make sure the underlying proto enum is accessible
    proto_enum = Foo._proto_enum
    assert isprotobufenum(proto_enum)


def test_dataobject_arrays():
    """Make sure arrays work as expected"""

    @dataobject
    class Foo(DataObjectBase):
        bar: List[str]

    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_STRING")
    assert check_field_label(Foo.get_proto_class(), "bar", "LABEL_REPEATED")


def test_dataobject_obj_refs_no_opt_types():
    """Make sure that references to other data objects and enums work as
    expected
    """

    @dataobject
    class BarEnum(Enum):
        EXAM = 0
        METAL = 1

    @dataobject
    class Foo(DataObjectBase):
        foo: str

    @dataobject
    class FooBar(DataObjectBase):
        foo: Foo
        bar: BarEnum

    assert check_field_type(FooBar.get_proto_class(), "foo", "TYPE_MESSAGE")
    assert check_field_message_type(
        FooBar.get_proto_class(), "foo", Foo.get_proto_class().DESCRIPTOR
    )
    assert check_field_enum_type(
        FooBar.get_proto_class(), "bar", BarEnum._proto_enum.DESCRIPTOR
    )


def test_dataobject_obj_refs_with_optional_types():
    """Make sure that references to other data objects and enums work as
    expected
    """

    @dataobject
    class BarEnum(Enum):
        EXAM = 0
        METAL = 1

    @dataobject
    class Foo(DataObjectBase):
        foo: str

    @dataobject
    class FooBar(DataObjectBase):
        foo: Foo
        optionalFoo: Optional[Foo]
        bar: BarEnum
        optionalBar: Optional[BarEnum]

    assert check_field_type(FooBar._proto_class, "foo", "TYPE_MESSAGE")
    for field in ["foo", "optionalFoo"]:
        assert check_field_message_type(
            FooBar._proto_class, field, Foo._proto_class.DESCRIPTOR
        )
    for field in ["bar", "optionalBar"]:
        assert check_field_enum_type(
            FooBar._proto_class, field, BarEnum._proto_enum.DESCRIPTOR
        )


def test_dataobject_invalid_schema():
    """Make sure that a ValueError is raised on an invalid schema"""
    with pytest.raises(ValueError):
        # pylint: disable=unused-variable
        @dataobject(schema="Foo")
        class Foo:
            pass


def test_dataobject_additional_methods():
    """Make sure that additional methods on wrapped classes (for messages and
    enums) are preserved
    """

    @dataobject
    class Foo(Enum):
        EXAM = 0
        DRINKS = 1

        @classmethod
        def is_exam(cls, val: "Foo") -> bool:
            return val == cls.EXAM

    @dataobject
    class Bar(DataObjectBase):
        bar: str

        def caps(self) -> str:
            return self.bar.upper()

    assert Foo.is_exam(Foo.EXAM)
    assert Bar("bat").caps() == "BAT"


def test_render_dataobject_protos_valid_dir():
    """Make sure that render_dataobject_protos correctly renders all generated
    protobufs to the target directory
    """

    @dataobject
    class BarEnum(Enum):
        EXAM = 0
        METAL = 1

    @dataobject
    class Foo(DataObjectBase):
        foo: str

    @dataobject
    class FooBar(DataObjectBase):
        foo: Foo
        bar: BarEnum

    with tempfile.TemporaryDirectory() as workdir:
        render_dataobject_protos(workdir)
        rendered_files = set(os.listdir(workdir))
        assert rendered_files == {
            BarEnum._proto_enum.DESCRIPTOR.file.name,
            Foo.get_proto_class().DESCRIPTOR.file.name,
            FooBar.get_proto_class().DESCRIPTOR.file.name,
        }


def test_render_dataobject_protos_no_dir():
    """Make sure that render_dataobject_protos correctly renders all generated
    protobufs to the target directory and creates the target dir if it doesn't exist
    """

    @dataobject
    class BarEnum(Enum):
        EXAM = 0
        METAL = 1

    @dataobject
    class Foo(DataObjectBase):
        foo: str

    @dataobject
    class FooBar(DataObjectBase):
        foo: Foo
        bar: BarEnum

    with tempfile.TemporaryDirectory() as workdir:
        protos_dir_path = os.path.join(workdir, "protos")
        render_dataobject_protos(protos_dir_path)
        rendered_files = set(os.listdir(protos_dir_path))
        assert rendered_files == {
            BarEnum._proto_enum.DESCRIPTOR.file.name,
            Foo.get_proto_class().DESCRIPTOR.file.name,
            FooBar.get_proto_class().DESCRIPTOR.file.name,
        }


def test_dataobject_with_discriminator():
    """Make sure that adding a discriminator works as expected"""

    @dataobject(
        schema={
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
    class BazObj(DataObjectBase):
        foo: str
        bar: str
        baz: str
        bat: str

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


def test_dataobject_with_oneof():
    """Make sure that using a Union to create a oneof works as expected"""

    @dataobject
    class BazObj(DataObjectBase):
        _private_slots = ("_which_oneof_datastream",)

        @dataobject
        class Foo(DataObjectBase):
            data: List[str]

        @dataobject
        class Bar(DataObjectBase):
            data: str

        data_stream: Union[
            Annotated[Foo, FieldNumber(1), OneofField("foo")],
            Annotated[Bar, FieldNumber(2), OneofField("bar")],
        ]

        def __getattr__(self, name):
            if name == "data_stream":
                if self._which_oneof_datastream == "foo":
                    return self.foo
                elif self._which_oneof_datastream == "bar":
                    return self.bar
                return None
            if name == "_foo":
                if self._which_oneof_datastream == "foo":
                    return self._data_stream
            if name == "_bar":
                if self._which_oneof_datastream == "bar":
                    return self._data_stream

        def __init__(self, *args, **kwargs):
            if "foo" in kwargs:
                self._which_oneof_datastream = "foo"
                self._data_stream = kwargs["foo"]
            if "bar" in kwargs:
                self._which_oneof_datastream = "bar"
                self._data_stream = kwargs["bar"]

    # proto tests
    foo1 = BazObj(foo=BazObj.Foo(data=["hello"]))
    assert isinstance(foo1.data_stream, BazObj.Foo)
    proto_repr_foo = foo1.to_proto()
    assert proto_repr_foo.foo.data == ["hello"]
    assert BazObj.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo

    bar1 = BazObj(foo=BazObj.Foo(data=["hello"]), bar=BazObj.Bar(data="world"))
    assert isinstance(bar1.data_stream, BazObj.Bar)
    proto_repr_bar = bar1.to_proto()
    assert proto_repr_bar.bar.data == "world"

    # json tests
    foo1 = BazObj(foo=BazObj.Foo(data=["hello"]))
    json_repr_foo = foo1.to_json()
    assert json.loads(json_repr_foo) == {
        "foo": {"data": ["hello"]},
        "bar": None,
    }


def test_dataobject_round_trip_json():
    """Make sure that a dataobject class can serialize to/from json"""

    @dataobject
    class BazObj(DataObjectBase):
        foo: str
        bar: int

    baz1 = BazObj(foo="foo", bar=1)
    js_repr = baz1.to_json()
    assert json.loads(js_repr) == {"foo": "foo", "bar": 1}
    baz2 = BazObj.from_json(js_repr)
    assert baz2.to_json() == js_repr


def test_dataobject_round_trip_proto():
    """Make sure that a dataobject class can serialize to/from proto"""

    @dataobject
    class BazObj(DataObjectBase):
        foo: str
        bar: int

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

    @dataobject
    class BazObj(DataObjectBase):
        foo: str

    x = BazObj("foobar")
    dir(x)


def test_dataobject_invocation_flavors():
    """Make sure invoking dataobject works in all of the different correct
    invocation styles and errors with all invalid styles

    VALID:
    1. No function call
    2. Function call with no args
    3. Function call with single positional argument
    4. Function call with keyword args

    INVALID:
    1. Unexpected kwargs
    2. Package as position and keyword arg
    3. No inheritance from DataObjectBase
    """
    ## Valid ##

    # 1. No function call
    @dataobject
    class Foo1(DataObjectBase):
        foo: int

    assert "foo" in Foo1._proto_class.DESCRIPTOR.fields_by_name

    # 2. Function call with no args
    @dataobject()
    class Foo2(DataObjectBase):
        foo: int

    assert "foo" in Foo2._proto_class.DESCRIPTOR.fields_by_name

    # 3. Function call with single positional argument
    @dataobject("foo.bar")
    class Foo3(DataObjectBase):
        foo: int

    assert "foo" in Foo3._proto_class.DESCRIPTOR.fields_by_name
    assert Foo3._proto_class.DESCRIPTOR.file.package == "foo.bar"

    # 4. Function call with keyword args
    @dataobject(package="foo.bar")
    class Foo4(DataObjectBase):
        foo: int

    assert "foo" in Foo4._proto_class.DESCRIPTOR.fields_by_name
    assert Foo4._proto_class.DESCRIPTOR.file.package == "foo.bar"

    ## INVALID ##

    # 1. Unexpected kwargs
    with pytest.raises(TypeError):

        @dataobject(buz="baz", package="foo.bar")
        class FooBad(DataObjectBase):
            foo: int

    # 2. Package as position and keyword arg
    with pytest.raises(TypeError):

        @dataobject("baz.bat", package="foo.bar")
        class Foo4(DataObjectBase):
            foo: int

    # 3. No inheritance from DataObjectBase
    with pytest.raises(ValueError):

        @dataobject
        class Foo4:
            foo: int


def test_dataobject_pre_existing_dataclass():
    """Make sure that wrapping a class that's already a dataclass works as
    expected by adding additional None defaults and re-making the dataclass
    """

    @dataobject
    @dataclass
    class Foo(DataObjectBase):
        foo: int

    assert is_dataclass(Foo)

    # Make sure defaults are added correctly per dataobject semantics
    inst = Foo()
    assert inst.foo is None


def test_dataobject_dataclass_non_default_init():
    """Make sure that a dataclass with a non-default __init__ does not get
    overwritten
    """

    @dataobject
    class Foo(DataObjectBase):
        foo: int

        def __init__(self, foo_base):
            self.foo = foo_base + 1

    assert is_dataclass(Foo)

    # Make sure defaults are added correctly per dataobject semantics
    inst = Foo(1)
    assert inst.foo == 2


def test_dataobject_dataclass_default_factory():
    """Make sure that a dataclass's datafactory field is preserved"""

    @dataobject
    class Foo(DataObjectBase):
        foo: List[int] = field(default_factory=list)

    assert is_dataclass(Foo)

    # Make sure default construction users the default factory
    inst = Foo()
    assert inst.foo is not None
    assert inst.foo == []


def test_enum_value_dereference():
    """Make sure that enum value objects can be used to instantiate data model
    objects and that they are correctly dereferenced in to_proto, to_dict, and
    to_json
    """

    @dataobject
    class FooEnum(Enum):
        FOO = 1
        BAR = 2

    @dataobject
    class Foo(DataObjectBase):
        foo: FooEnum

    # Create an instance with an enum value object
    inst = Foo(foo=FooEnum.FOO)
    assert inst.to_proto().foo == FooEnum.FOO.value
    assert inst.to_dict()["foo"] == FooEnum.FOO.name
    assert json.loads(inst.to_json())["foo"] == FooEnum.FOO.name

    # Create an instance with an integer value
    inst = Foo(foo=FooEnum.FOO.value)
    assert inst.to_proto().foo == FooEnum.FOO.value
    assert inst.to_dict()["foo"] == FooEnum.FOO.name
    assert json.loads(inst.to_json())["foo"] == FooEnum.FOO.name
