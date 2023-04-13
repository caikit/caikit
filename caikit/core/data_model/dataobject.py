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

"""This module defines the @schema decorator which can be used to declare data
model objects inline without manually defining the protobufs representation
"""


# Standard
from datetime import datetime
from functools import update_wrapper
from types import ModuleType
from typing import Callable, Dict, List, Optional, Set, Type, Union
import importlib
import sys
import types
import typing

# Third Party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

# First Party
from jtd_to_proto.jtd_to_proto import JTD_TO_PROTO_TYPES
from jtd_to_proto.validation import is_valid_jtd
import alog
import jtd_to_proto

# Local
from ..toolkit.errors import error_handler
from . import enums
from .base import DataBase, _DataBaseMetaClass
from .streams.data_stream import DataStream

## Globals #####################################################################

log = alog.use_channel("SCHEMA")
error = error_handler.get(log)

# Type defs for input schemas to a dataobject
_SCHEMA_VALUE_TYPE = Union[str, "_SCHEMA_VALUE_TYPE", Type[DataBase]]
_SCHEMA_DEF_TYPE = Dict[str, _SCHEMA_VALUE_TYPE]

# Registry of auto-generated protos so that they can be rendered to .proto
_AUTO_GEN_PROTO_CLASSES = []

# Reserved keywords in JTD
_JTD_KEYWORDS = [
    "elements",
    "type",
    "properties",
    "enum",
    "values",
    "optionalProperties",
    "additionalProperties",
    "discriminator",
]

# Type defs for schemas passed to jtd_to_proto
_JTD_VALUE_TYPE = Union[str, "_JTD_VALUE_TYPE", _descriptor.Descriptor]
_JTD_DEF_TYPE = Dict[str, _JTD_VALUE_TYPE]

# Python type -> jtd name
_NATIVE_TYPE_TO_JTD = {
    str: "string",
    int: "int64",
    float: "float64",
    bytes: "bytes",
    bool: "boolean",
    datetime: "timestamp",
}

## Public ######################################################################

# Common package prefix
CAIKIT_DATA_MODEL = "caikit_data_model"


def dataobject(
    schema: _SCHEMA_DEF_TYPE,
    package: str = CAIKIT_DATA_MODEL,
) -> Callable[[Type], Type[DataBase]]:
    """The @schema decorator can be used to define a Data Model object's schema
    inline with the definition of the python class rather than needing to bind
    to a pre-compiled protobufs class. For example:

    @dataobject(
        package="foo.bar",
        schema={"foo": str, "bar": int},
    )
    class MyDataObject:
        '''My Custom Data Object'''

    NOTE: The wrapped class must NOT inherit directly from DataBase. That
        inheritance will be added by this decorator, but if it is written
        directly, the metaclass that links protobufs to the class will be called
        before this decorator can auto-gen the protobufs class.

    Args:
        schema:  _SCHEMA_DEF_TYPE
            The full schema definition dict
        package:  str
            The package name to use for the generated protobufs class

    Returns:
        decorator:  Callable[[Type], Type[DataBase]]
            The decorator function that will wrap the given class
    """

    def decorator(cls: Type) -> Type[DataBase]:
        # Make sure that the wrapped class does NOT inherit from DataBase
        error.value_check(
            "<COR95184230E>",
            not issubclass(cls, DataBase),
            "{} should not directly inherit from DataBase when using @schema",
            cls.__name__,
        )

        # Create the message class from the schema
        jtd_def = _to_jtd_schema(schema)
        log.debug3("JTD Def for %s: %s", cls.__name__, jtd_def)
        proto_class = jtd_to_proto.descriptor_to_message_class(
            jtd_to_proto.jtd_to_proto(
                name=cls.__name__,
                package=package,
                jtd_def=jtd_def,
            )
        )
        # pylint: disable=unused-variable,global-variable-not-assigned
        global _AUTO_GEN_PROTO_CLASSES
        _AUTO_GEN_PROTO_CLASSES.append(proto_class)

        # Add enums to the global enums module
        for enum_class in _get_all_enums(proto_class):
            log.debug2("Importing enum [%s]", enum_class.DESCRIPTOR.name)
            enums.import_enum(enum_class)

        # Declare the merged class that binds DataBase to the wrapped class with
        # this generated proto class
        if isinstance(proto_class, type):
            wrapper_class = _make_data_model_class(proto_class, cls)
        else:
            ck_enum = enums.EnumBase(proto_class)

            # Handling enums with the @dataobject decorator is quite tricky
            # because unlike a message which is represented as a `class` in
            # python, an enum is represented as an INSTANCE of a `class`. This
            # means that the naive implementation of this decorator would apply
            # to a `class`, but return an object that is NOT a `class` (e.g.
            # isinstance(MyEnum, type) == False). This is bad for two distinct
            # reasons:
            #
            # 1. It's confusing to see code written with a decorator around a
            #   `class` and have the resulting thing NOT be a class
            # 2. It makes it harder to add additional functionality to the Enum
            #   by defining custom methods on your "enum class"
            #
            # To get around this, we need to "bind" the instance of the EnumBase
            # to a net-new `class`! This class will function as a singleton
            # wrapper around the EnumBase instance, but will allow the decorator
            # to return a true `class` and allow user-defined methods on that
            # class to persist through the decorator.
            # pylint: disable=unused-variable
            class EnumBindingMeta(type):
                def __new__(mcs, name, bases, attrs):
                    attrs.update(vars(ck_enum))
                    for method in ["toYAML", "toJSON", "toDict"]:
                        attrs[method] = getattr(ck_enum, method)
                    attrs["_proto_enum"] = proto_class
                    attrs["_singleton_inst"] = ck_enum
                    bases = tuple(list(bases) + [_EnumBaseSentinel])
                    return super().__new__(mcs, name, bases, attrs)

                def __call__(cls):
                    return ck_enum

                def __str__(cls):
                    return ck_enum.__str__()

                def __repr__(cls):
                    return ck_enum.__repr__()

            class _Dummy(cls, metaclass=EnumBindingMeta):
                pass

            update_wrapper(_Dummy, cls, updated=())
            wrapper_class = _Dummy

        # Attach the proto class to the protobufs module
        parent_mod_name = getattr(cls, "__module__", "").rpartition(".")[0]
        log.debug2("Parent mod name: %s", parent_mod_name)
        if parent_mod_name:
            proto_mod_name = ".".join([parent_mod_name, "protobufs"])
            try:
                proto_mod = importlib.import_module(proto_mod_name)
            except ImportError:
                log.debug("Creating new protobufs module: %s", proto_mod_name)
                proto_mod = ModuleType(proto_mod_name)
                sys.modules[proto_mod_name] = proto_mod
            setattr(proto_mod, cls.__name__, proto_class)

        # Return the merged data class
        return wrapper_class

    return decorator


def render_dataobject_protos(interfaces_dir: str):
    """Write out protobufs files for all proto classes generated from dataobjects
    to the target interfaces directory

    Args:
        interfaces_dir:  str
            The target directory (must already exist)
    """
    for proto_class in _AUTO_GEN_PROTO_CLASSES:
        proto_class.write_proto_file(interfaces_dir)


## Implementation Details ######################################################


class _EnumBaseSentinel:
    """This base class is used to provide a common base class for enum warpper
    classes so that they can be identified generically
    """


# pylint: disable=too-many-return-statements
def _to_jtd_schema(
    input_schema: _SCHEMA_DEF_TYPE, is_inside_properties_dict: bool = False
) -> _JTD_DEF_TYPE:
    """Recursive helper that will convert an input schema to a fully fleshed out
    JTD schema
    """
    try:
        # Unwrap optional to base type if applicable
        input_schema = _unwrap_optional_type(input_schema)

        # If it's a reference to an EnumBase, de-alias to that enum's EnumDescriptor
        # NOTE: This must come before the check for dict since EnumBase instances
        #   are themselves dicts
        if isinstance(input_schema, enums.EnumBase) or (
            isinstance(input_schema, type)
            and issubclass(input_schema, _EnumBaseSentinel)
        ):
            return {"type": input_schema._proto_enum.DESCRIPTOR}

        if isinstance(input_schema, dict):
            # If this dict is already a JTD schema, return it as is
            if is_valid_jtd(input_schema, valid_types=JTD_TO_PROTO_TYPES.keys()):
                return input_schema

            # If the dict is structured as a JTD element already, recurse on the
            # values
            if any(keyword in input_schema for keyword in _JTD_KEYWORDS):
                return {
                    k: _to_jtd_schema(v, "properties" in k.lower())
                    for k, v in input_schema.items()
                }

            # If not, assume it's a flat properties dict
            # Check to make sure we don't re-wrap *properties
            translated_dict = {k: _to_jtd_schema(v) for k, v in input_schema.items()}
            return (
                {"properties": translated_dict}
                if not is_inside_properties_dict
                else translated_dict
            )

        # If it's a reference to another data model object, de-alias to that
        # object's underlying proto descriptor
        if isinstance(input_schema, type) and issubclass(input_schema, DataBase):
            return {"type": input_schema.get_proto_class().DESCRIPTOR}

        # If it's a native type, wrap it as a "type" element
        if input_schema in _NATIVE_TYPE_TO_JTD:
            return {"type": _NATIVE_TYPE_TO_JTD[input_schema]}

        # If it's a list or data stream, wrap it with "elements":
        if typing.get_origin(input_schema) in [list, DataStream]:
            # type_ could be caikit.core.data_model.streams.data_stream.DataStream[int]
            return {"elements": _to_jtd_schema(typing.get_args(input_schema)[0])}

        # All other cases are invalid!
        raise ValueError(f"Invalid input schema: {input_schema}")

    except ValueError:
        log.error("Invalid schema: %s", input_schema)
        raise


def _unwrap_optional_type(type_: typing.Any) -> typing.Any:
    """Unwrap an Optional[T] type, or return the type as-is if it is not an optional
    NB: Optional[T] is expressed as Union[T, None]
    This function checks for Unions of [T, None] and returns T, or raises if the union
    contains more types, as those need to be handled differently (and are not yet supported)
    """
    if typing.get_origin(type_) != Union:
        return type_
    possible_types = set(typing.get_args(type_))
    possible_types.discard(type(None))
    if len(possible_types) == 1:
        return list(possible_types)[0]
    raise ValueError(f"Invalid input schema, cannot handle unions yet: {type_}")


def _get_all_enums(
    proto_class: Union[_message.Message, EnumTypeWrapper],
) -> List[EnumTypeWrapper]:
    """Given a generated proto class, recursively extract all enums"""
    all_enums = []
    if isinstance(proto_class, EnumTypeWrapper):
        all_enums.append(proto_class)
    else:
        for enum_descriptor in proto_class.DESCRIPTOR.enum_types:
            all_enums.append(getattr(proto_class, enum_descriptor.name))
        for nested_proto_descriptor in proto_class.DESCRIPTOR.nested_types:
            all_enums.extend(
                _get_all_enums(getattr(proto_class, nested_proto_descriptor.name))
            )

    return all_enums


def _make_data_model_class(proto_class, wrapped_cls):
    wrapper_cls = _DataBaseMetaClass(
        wrapped_cls.__name__,
        tuple([DataBase, wrapped_cls]),
        {"_proto_class": proto_class, **wrapped_cls.__dict__},
    )
    update_wrapper(wrapper_cls, wrapped_cls, updated=())

    # Recursively make all nested message wrappers
    for nested_message_descriptor in proto_class.DESCRIPTOR.nested_types:
        nested_message_name = nested_message_descriptor.name
        nested_proto_class = getattr(proto_class, nested_message_name)
        setattr(
            wrapper_cls,
            nested_message_name,
            _make_data_model_class(
                nested_proto_class,
                types.new_class(nested_message_name),
            ),
        )
    for nested_enum_descriptor in proto_class.DESCRIPTOR.enum_types:
        setattr(
            wrapper_cls,
            nested_enum_descriptor.name,
            getattr(enums, nested_enum_descriptor.name),
        )

    return wrapper_cls
