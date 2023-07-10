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
from typing import Dict, List, Optional, Union
import copy
import json
import os
import tempfile

# Third Party
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pb2, descriptor_pool, message, struct_pb2
import numpy as np
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
from caikit.core.data_model.data_backends.dict_backend import DictBackend
from caikit.core.data_model.dataobject import (
    _AUTO_GEN_PROTO_CLASSES,
    CAIKIT_DATA_MODEL,
    make_dataobject,
    render_dataobject_protos,
)
from caikit.core.data_model.enums import isprotobufenum
from caikit.core.data_model.json_dict import JsonDict

## Helpers #####################################################################


@pytest.fixture(autouse=True)
def temp_dpool():
    """Fixture to isolate the descriptor pool used in each test"""
    dpool = descriptor_pool.DescriptorPool()
    global_dpool = descriptor_pool._DEFAULT
    descriptor_pool._DEFAULT = dpool
    fd = descriptor_pb2.FileDescriptorProto()
    struct_pb2.DESCRIPTOR.CopyToProto(fd)
    dpool.Add(fd)

    ##
    # HACK! Doing this _appears_ to solve the mysterious segfault cause by
    # using Struct inside a temporary descriptor pool. The inspiration for this
    # was:
    #
    # https://github.com/protocolbuffers/protobuf/issues/12047
    #
    # NOTE: This only works for protobuf 4.X (and as far as we know, it's not
    #     needed for 3.X)
    ##
    try:
        # Third Party
        from google.protobuf.message_factory import GetMessageClassesForFiles

        msgs = GetMessageClassesForFiles([fd.name], dpool)
        _ = msgs["google.protobuf.Struct"]
        _ = msgs["google.protobuf.Value"]
        _ = msgs["google.protobuf.ListValue"]

    # Nothing to do for protobuf 3.X
    except ImportError:
        pass
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


def test_dataobject_simple_dict():
    """Make sure simple dicts work as expected"""

    @dataobject
    class Foo(DataObjectBase):
        bar: Dict[str, int]

    assert check_field_type(Foo.get_proto_class(), "bar", "TYPE_MESSAGE")
    assert (
        Foo.get_proto_class()
        .DESCRIPTOR.fields_by_name["bar"]
        .message_type.GetOptions()
        .map_entry
    )

    dict_input = {"foo": 1, "bar": 2}
    foo = Foo(bar=dict_input)
    assert foo.bar == dict_input

    assert Foo.from_proto(foo.to_proto()).bar == dict_input


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


def test_dataobject_with_oneof():
    """Make sure that using a Union to create a oneof works as expected"""

    @dataobject
    class BazObj(DataObjectBase):
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

    # Construct with oneof field name
    foo1 = BazObj(foo=BazObj.Foo(data=["hello"]))
    assert isinstance(foo1.data_stream, BazObj.Foo)
    assert foo1.which_oneof("data_stream") == "foo"
    assert foo1.foo is foo1.data_stream
    assert foo1.bar is None

    # Test other oneof field name
    bar1 = BazObj(bar=BazObj.Bar(data="world"))
    assert isinstance(bar1.data_stream, BazObj.Bar)
    assert bar1.which_oneof("data_stream") == "bar"
    assert bar1.bar is bar1.data_stream
    assert bar1.foo is None

    # Test to_dict
    dict_repr_foo = foo1.to_dict()
    assert dict_repr_foo == {
        "foo": {"data": ["hello"]},
    }
    assert dict_repr_foo["foo"]["data"] == ["hello"]

    # Test proto round trip
    proto_repr_foo = foo1.to_proto()
    assert proto_repr_foo.foo.data == ["hello"]
    assert BazObj.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo
    proto_repr_bar = bar1.to_proto()
    assert proto_repr_bar.bar.data == "world"

    # Test json round trip
    json_repr_foo = foo1.to_json()
    assert json.loads(json_repr_foo) == {
        "foo": {"data": ["hello"]},
    }
    assert BazObj.from_json(json_repr_foo) == foo1

    # Test setattr
    foo1.bar = BazObj.Bar(data="it's a bar")
    assert foo1.which_oneof("data_stream") == "bar"
    assert foo1.data_stream is foo1.bar
    assert foo1.foo is None

    # Construct with oneof name
    foo2 = BazObj(data_stream=BazObj.Foo(data=["some", "foo"]))
    assert foo2.data_stream.data == ["some", "foo"]
    assert foo2.bar is None
    assert foo2.foo is foo2.data_stream
    assert foo2.which_oneof("data_stream") == "foo"

    # Assign with oneof name
    foo2.data_stream = BazObj.Bar(data="asdf")
    assert foo2.foo is None
    assert foo2.bar is foo2.data_stream
    assert foo2.which_oneof("data_stream") == "bar"

    # Construct with positional oneof name
    foo2 = BazObj(BazObj.Foo(data=["some", "foo"]))
    assert foo2.data_stream.data == ["some", "foo"]
    assert foo2.bar is None
    assert foo2.foo is foo2.data_stream
    assert foo2.which_oneof("data_stream") == "foo"

    foo3 = BazObj()
    assert foo3.foo is None
    assert foo3.bar is None
    assert foo3.data_stream is None
    assert foo3.which_oneof("data_stream") == None
    # Invalid constructors
    with pytest.raises(TypeError):
        BazObj(BazObj.Foo(), foo=BazObj.Foo())
    with pytest.raises(TypeError):
        BazObj(data_stream=BazObj.Foo(), foo=BazObj.Foo())
    with pytest.raises(TypeError):
        BazObj(foo=BazObj.Foo(), bar=BazObj.Bar())


def test_dataobject_with_same_type_of_oneof():
    """Make sure that using a Union to create a oneof with the same types works as expected"""

    @dataobject
    class Foo(DataObjectBase):
        foo: Union[
            Annotated[bool, FieldNumber(10), OneofField("foo_bool1")],
            Annotated[bool, FieldNumber(20), OneofField("foo_bool2")],
        ]

    # if the fields are of the same type, then by default the first one is set
    foo1 = Foo(True)
    assert foo1.which_oneof("foo") == "foo_bool1"
    assert foo1.foo_bool1
    assert foo1.foo_bool2 == None

    # unless set explicitly
    foo2 = Foo(foo_bool2=True)
    assert foo2.which_oneof("foo") == "foo_bool2"
    assert foo2.foo_bool1 == None
    assert foo2.foo_bool2


def test_dataobject_primitive_oneof_round_trips():
    @dataobject
    class Foo(DataObjectBase):
        foo: Union[
            Annotated[int, FieldNumber(10), OneofField("foo_int")],
            Annotated[float, FieldNumber(20), OneofField("foo_float")],
        ]

    # proto round trip
    foo1 = Foo(foo_int=2)
    assert foo1.which_oneof("foo") == "foo_int"
    proto_repr_foo = foo1.to_proto()
    assert Foo.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo

    foo2 = Foo(foo=2)
    assert foo2.which_oneof("foo") == "foo_int"
    proto_repr_foo = foo2.to_proto()
    assert Foo.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo

    # dict test
    assert foo1.to_dict() == {"foo_int": 2}

    # json round trip
    json_repr_foo = foo1.to_json()
    assert json.loads(json_repr_foo) == {
        "foo_int": 2,
    }
    assert Foo.from_json(json_repr_foo) == foo1


def test_dataobject_oneof_from_backend():
    """Make sure that a oneof can be correctly accessed from a backend"""

    @dataobject
    class Foo(DataObjectBase):
        foo: Union[int, str]

    data_dict1 = {"foo_int": 1234}
    backend1 = DictBackend(data_dict1)
    msg1 = Foo.from_backend(backend1)
    assert msg1.foo == 1234
    assert msg1.foo_int == 1234
    assert msg1.which_oneof("foo") == "foo_int"

    data_dict2 = {"foo": 1234}
    backend2 = DictBackend(data_dict2)
    msg2 = Foo.from_backend(backend2)
    assert msg2.foo == 1234
    assert msg2.foo_int == 1234
    assert msg2.which_oneof("foo") == "foo_int"


def test_dataobject_oneof_numeric_type_precedence():
    """Make sure that when inferring the which field from the python type, the
    value of bool < int < float is respected
    """

    @dataobject
    class Foo(DataObjectBase):
        value: Union[
            # NOTE: The order matters here. Since float is declared first, it
            #   would naturally occur before int without correct sorting
            Annotated[float, OneofField("float_val")],
            Annotated[int, OneofField("int_val")],
            Annotated[bool, OneofField("bool_val")],
        ]

    foo_bool = Foo(True)
    assert foo_bool.which_oneof("value") == "bool_val"
    foo_int = Foo(123)
    assert foo_int.which_oneof("value") == "int_val"
    foo_float = Foo(1.23)
    assert foo_float.which_oneof("value") == "float_val"


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


def test_np_dtypes():
    """Make sure that numpy dtype types can be used in dataobjects"""

    @dataobject
    class Foo(DataObjectBase):
        int32: np.int32
        int64: np.int64
        uint32: np.uint32
        uint64: np.uint64
        float32: np.float32
        float64: np.float64

    descriptor = Foo._proto_class.DESCRIPTOR
    assert (
        descriptor.fields_by_name["int32"].type
        == _descriptor.FieldDescriptor.TYPE_INT32
    )
    assert (
        descriptor.fields_by_name["int64"].type
        == _descriptor.FieldDescriptor.TYPE_INT64
    )
    assert (
        descriptor.fields_by_name["uint32"].type
        == _descriptor.FieldDescriptor.TYPE_UINT32
    )
    assert (
        descriptor.fields_by_name["uint64"].type
        == _descriptor.FieldDescriptor.TYPE_UINT64
    )
    assert (
        descriptor.fields_by_name["float32"].type
        == _descriptor.FieldDescriptor.TYPE_FLOAT
    )
    assert (
        descriptor.fields_by_name["float64"].type
        == _descriptor.FieldDescriptor.TYPE_DOUBLE
    )


@pytest.mark.parametrize("run_num", range(100))
def test_dataobject_jsondict(temp_dpool, run_num):
    """Make sure that a JsonDict type is handled correctly in a dataobject

    NOTE: This test is repeated 100x due to a strange segfault in `upb` that it
        can trigger. The workaround above in `temp_dpool` should solve it, but
        we retain the repetition to catch anything that's missed.
    """

    @dataobject
    class Foo(DataObjectBase):
        js_dict: JsonDict

    # Make sure the field has the right type
    Struct = temp_dpool.FindMessageTypeByName("google.protobuf.Struct")
    assert Foo._proto_class.DESCRIPTOR.fields_by_name["js_dict"].message_type == Struct

    # Make sure dict is preserved on init
    js_dict = {"foo": {"bar": [1, 2, 3]}}
    foo = Foo(js_dict)
    assert foo.js_dict == js_dict

    # Make sure conversion to struct happens on to_proto
    foo_proto = foo.to_proto()
    assert set(foo_proto.js_dict.fields.keys()) == set(js_dict.keys())
    assert foo_proto.js_dict.fields["foo"].struct_value
    assert set(foo_proto.js_dict.fields["foo"].struct_value.fields.keys()) == set(
        js_dict["foo"].keys()
    )

    # Make sure conversion back to dict happens on from_proto
    foo2 = Foo.from_proto(foo_proto)
    assert foo2.js_dict == foo.js_dict


def test_dataobject_jsondict_repeated(temp_dpool):
    """Make sure that a list of JsonDict types is handled correctly in a dataobject"""

    @dataobject
    class Foo(DataObjectBase):
        js_dict: List[JsonDict]

    # Make sure the field has the right type
    Struct = temp_dpool.FindMessageTypeByName("google.protobuf.Struct")
    assert Foo._proto_class.DESCRIPTOR.fields_by_name["js_dict"].message_type == Struct

    # Make sure dict is preserved on init
    js_dict = [{"foo": {"bar": [1, 2, 3]}}]
    foo = Foo(js_dict)
    assert foo.js_dict == js_dict

    # Make sure conversion to struct happens on to_proto
    foo_proto = foo.to_proto()
    assert len(foo_proto.js_dict) == 1
    assert set(foo_proto.js_dict[0].fields.keys()) == set(js_dict[0].keys())
    assert foo_proto.js_dict[0].fields["foo"].struct_value
    assert set(foo_proto.js_dict[0].fields["foo"].struct_value.fields.keys()) == set(
        js_dict[0]["foo"].keys()
    )

    # Make sure conversion back to dict happens on from_proto
    foo2 = Foo.from_proto(foo_proto)
    assert foo2.js_dict == foo.js_dict


def test_dataobject_to_kwargs(temp_dpool):
    """to_kwargs does a non-recursive version of to_dict"""

    @dataobject
    class Bar(DataObjectBase):
        bar: int

    @dataobject
    class Foo(DataObjectBase):
        int_val: int
        type_union_val: Union[int, str]
        bar_val: Bar

    bar = Bar(bar=42)
    foo = Foo(int_val=1, type_union_val="foo", bar_val=bar)

    kwargs = foo.to_kwargs()

    # `type_union_val` is set here rather than the `type_union_val_str_val` internal oneof field name
    assert kwargs == {"int_val": 1, "type_union_val": "foo", "bar_val": bar}


def test_dataobject_inheritance(temp_dpool):
    """Make sure that dataobject classes can inherit from each other in the same
    way that dataclasses can
    """

    @dataobject
    class Base(DataObjectBase):
        foo: int
        bar: int

    @dataobject
    class Derived(Base):
        bar: str
        baz: str

    # Validate Base
    desc = Base.get_proto_class().DESCRIPTOR
    fld = desc.fields_by_name["foo"]  # To save typing
    assert "foo" in desc.fields_by_name
    assert "bar" in desc.fields_by_name
    assert "baz" not in desc.fields_by_name
    assert desc.fields_by_name["foo"].type == fld.TYPE_INT64
    assert desc.fields_by_name["bar"].type == fld.TYPE_INT64
    inst = Derived(1, 2)
    assert inst.foo == 1
    assert inst.bar == 2

    # Validate Derived
    desc = Derived.get_proto_class().DESCRIPTOR
    fld = desc.fields_by_name["foo"]  # To save typing
    assert "foo" in desc.fields_by_name
    assert "bar" in desc.fields_by_name
    assert "baz" in desc.fields_by_name
    assert desc.fields_by_name["foo"].type == fld.TYPE_INT64
    assert desc.fields_by_name["bar"].type == fld.TYPE_STRING
    assert desc.fields_by_name["baz"].type == fld.TYPE_STRING
    inst = Derived(1, "asdf", "qwer")
    assert inst.foo == 1
    assert inst.bar == "asdf"
    assert inst.baz == "qwer"


def test_dataobject_union_repeated():
    """Make sure that a oneof with lists of primitive fields works correctly"""

    @dataobject
    class Foo(DataObjectBase):
        foo: Union[List[str], List[int]]
        bar: Union[List[str], List[int]]

    # The above behaves _almost_ the same way as this
    # with some naming caveats for one-of fields being
    # foo_foointsequence instead of foo_int_sequence and
    # bar_barintsequence instead of bar_int_sequence

    # @dataobject
    # class Foo(DataObjectBase):
    #     @dataobject
    #     class FooIntSequence(DataObjectBase):
    #         values: List[int]

    #     @dataobject
    #     class FooStrSequence(DataObjectBase):
    #         values: List[str]

    #     @dataobject
    #     class BarIntSequence(DataObjectBase):
    #         values: List[int]

    #     @dataobject
    #     class BarStrSequence(DataObjectBase):
    #         values: List[str]

    #     foo: Union[FooIntSequence, FooStrSequence]
    #     bar: Union[BarIntSequence, BarStrSequence]

    # Foo
    # proto round trip
    foo_int = Foo.FooIntSequence(values=[1, 2])
    foo1 = Foo(foo=foo_int)
    assert foo1.which_oneof("foo") == "foo_int_sequence"
    proto_repr_foo = foo1.to_proto()
    assert Foo.from_proto(proto=proto_repr_foo).to_proto() == proto_repr_foo

    # dict test
    assert foo1.to_dict() == {"foo_int_sequence": {"values": [1, 2]}}

    # json round trip
    json_repr_foo = foo1.to_json()
    assert json.loads(json_repr_foo) == {"foo_int_sequence": {"values": [1, 2]}}
    foo_json_repr = Foo.from_json(json_repr_foo)
    assert foo_json_repr.to_json() == json_repr_foo

    foo_str = Foo.FooStrSequence(values=["hello", "world"])
    foo2 = Foo(foo=foo_str)
    assert foo2.which_oneof("foo") == "foo_str_sequence"
    proto_repr_foo2 = foo2.to_proto()
    assert Foo.from_proto(proto=proto_repr_foo2).to_proto() == proto_repr_foo2

    # Bar
    # proto round trip
    bar_int = Foo.BarIntSequence(values=[1, 2])
    bar1 = Foo(bar=bar_int)
    assert bar1.which_oneof("bar") == "bar_int_sequence"
    proto_repr_bar = bar1.to_proto()
    assert Foo.from_proto(proto=proto_repr_bar).to_proto() == proto_repr_bar

    # dict test
    assert bar1.to_dict() == {"bar_int_sequence": {"values": [1, 2]}}

    # json round trip
    json_repr_bar = bar1.to_json()
    assert json.loads(json_repr_bar) == {"bar_int_sequence": {"values": [1, 2]}}
    bar_json_repr = Foo.from_json(json_repr_bar)
    assert bar_json_repr.to_json() == json_repr_bar

    bar_str = Foo.BarStrSequence(values=["hello", "world"])
    bar2 = Foo(bar=bar_str)
    assert bar2.which_oneof("bar") == "bar_str_sequence"
    proto_repr_bar2 = bar2.to_proto()
    assert Foo.from_proto(proto=proto_repr_bar2).to_proto() == proto_repr_bar2


def test_dataobject_function_inheritance(temp_dpool):
    """Make sure inheritance works to override functionality without changing
    the schema of the parent
    """

    @dataobject
    class Base(DataObjectBase):
        foo: int

        def doit(self):
            return self.foo * 2

    @dataobject
    class Derived(Base):
        def doit(self):
            return self.foo * 3

    b_inst = Base(1)
    assert b_inst.doit() == 2

    d_inst = Derived(1)
    assert d_inst.doit() == 3


def test_make_dataobject_no_optionals(temp_dpool):
    """Test that dataobject classes can be created dynamically without optional
    values given
    """
    data_object = make_dataobject(
        name="FooBar",
        annotations={"foo": str, "bar": int},
    )

    assert data_object.__name__ == "FooBar"
    assert data_object.__annotations__ == {"foo": str, "bar": int}
    assert issubclass(data_object, DataObjectBase)
    assert data_object._proto_class.DESCRIPTOR.name == "FooBar"
    assert data_object._proto_class.DESCRIPTOR.file.package == CAIKIT_DATA_MODEL


def test_make_dataobject_with_optionals(temp_dpool):
    """Test that dataobject classes can be created dynamically with optional
    values given
    """

    class OtherBase:
        pass

    data_object = make_dataobject(
        name="FooBar",
        annotations={"foo": str, "bar": int},
        bases=(OtherBase,),
        attrs={"prop": "val"},
        proto_name="SomeOtherFooBar",
        package="foo.bar.baz",
    )

    assert data_object.__name__ == "FooBar"
    assert data_object.__annotations__ == {"foo": str, "bar": int}
    assert issubclass(data_object, DataObjectBase)
    assert issubclass(data_object, OtherBase)
    assert data_object.prop == "val"
    assert data_object._proto_class.DESCRIPTOR.name == "SomeOtherFooBar"
    assert data_object._proto_class.DESCRIPTOR.file.package == "foo.bar.baz"
