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
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
import dataclasses

# Third Party
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import struct_pb2
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import PY_TO_PROTO_TYPES, DataclassConverter
import alog
import py_to_proto

# Local
from ..toolkit.errors import error_handler
from . import enums
from .base import DataBase, _DataBaseMetaClass
from .json_dict import JsonDict

## Globals #####################################################################

log = alog.use_channel("SCHEMA")
error = error_handler.get(log)

# Type mapping for type hints in @dataobject classes
DATAOBJECT_PY_TO_PROTO_TYPES = {
    JsonDict: struct_pb2.Struct,
    np.int32: _descriptor.FieldDescriptor.TYPE_INT32,
    np.int64: _descriptor.FieldDescriptor.TYPE_INT64,
    np.uint32: _descriptor.FieldDescriptor.TYPE_UINT32,
    np.uint64: _descriptor.FieldDescriptor.TYPE_UINT64,
    np.float32: _descriptor.FieldDescriptor.TYPE_FLOAT,
    np.float64: _descriptor.FieldDescriptor.TYPE_DOUBLE,
    **PY_TO_PROTO_TYPES,
}

# Common package prefix
CAIKIT_DATA_MODEL = "caikit_data_model"

# Registry of auto-generated protos so that they can be rendered to .proto
_AUTO_GEN_PROTO_CLASSES = []

# Special attribute used to indicate which defaults are user provided
_USER_DEFINED_DEFAULTS = "__user_defined_defaults__"

## Public ######################################################################


class _DataObjectBaseMetaClass(_DataBaseMetaClass):
    """This metaclass is used for the DataObject base class so that all data
    objects can delay the creation of their proto class until after the
    metaclass has been instantiated.
    """

    def __new__(mcs, name, bases, attrs):
        """When instantiating a new DataObject class, the proto class will not
        yet have been generated, but the set of fields will be known since the
        class will be the raw input representation of a @dataclass
        """

        # Get the annotations that will go into the dataclass
        if name != "DataObjectBase":
            field_names = attrs.get("__annotations__")
            parent_dataobjects = [
                base for base in bases if isinstance(base, _DataBaseMetaClass)
            ]
            field_name_sets = [base.fields for base in parent_dataobjects]
            if field_names is not None:
                field_name_sets = [field_names] + field_name_sets

            # We need at least one field name set!
            if not field_name_sets:
                raise TypeError(
                    "All DataObjectBase classes must follow dataclass syntax"
                )

            # Flatten the field names
            field_names = {
                field_name
                for field_name_set in field_name_sets
                for field_name in field_name_set
            }
            attrs[_DataBaseMetaClass._FWD_DECL_FIELDS] = field_names

        # Delegate to the base metaclass
        return super().__new__(mcs, name, bases, attrs)


class DataObjectBase(DataBase, metaclass=_DataObjectBaseMetaClass):
    """A DataObject is a data model class that is backed by a @dataclass.

    Data model classes that use the @dataobject decorator must derive from this
    base class.
    """


_DataObjectBaseT = TypeVar("_DataObjectBaseT", bound=DataObjectBase)


def dataobject(*args, **kwargs) -> Callable[[_DataObjectBaseT], _DataObjectBaseT]:
    """The @dataobject decorator can be used to define a Data Model object's
    schema inline with the definition of the python class rather than needing to
    bind to a pre-compiled protobufs class. For example:

    @dataobject("foo.bar")
    class MyDataObject(DataObjectBase):
        '''My Custom Data Object'''
        foo: str
        bar: int

    NOTE: The wrapped class must NOT inherit directly from DataBase. That
        inheritance will be added by this decorator, but if it is written
        directly, the metaclass that links protobufs to the class will be called
        before this decorator can auto-gen the protobufs class.

    The `dataobject` decorator will not provide tools with enough information
    to perform type completion for constructions in an IDE, or static
    typechecking.  In order to have that, the `dataclass` decorator
    may optionally be added, with the slight overhead of wasted effort in
    creating the "standard" __init__ function which then gets re-done by
    @dataobject.  The `dataclass` must follow the `dataobject` decorator.  For example:

    @dataobject("foo.bar")
    @dataclass
    class MyDataObject(DataObjectBase):
        '''My Custom Data Object'''
        foo: str
        bar: int

    Kwargs:
        package:  str
            The package name to use for the generated protobufs class

    Returns:
        decorator:  Callable[[Type], Type[DataBase]]
            The decorator function that will wrap the given class
    """

    def decorator(cls: _DataObjectBaseT) -> _DataObjectBaseT:
        # Make sure that the wrapped class does NOT inherit from DataBase
        error.value_check(
            "<COR95184230E>",
            issubclass(cls, (DataObjectBase, Enum)),
            "{} must inherit from DataObjectBase/Enum when using @dataobject",
            cls.__name__,
        )

        # Add the package to the kwargs
        kwargs.setdefault("package", package)

        # If it's not an enum, fill in any missing field defaults as None
        # and make sure it's a dataclass
        if not issubclass(cls, Enum):
            log.debug2("Wrapping data class %s", cls)
            user_defined_defaults = {}
            for annotation in getattr(cls, "__annotations__", {}):
                user_defined_default = getattr(cls, annotation, dataclasses.MISSING)
                if user_defined_default == dataclasses.MISSING:
                    log.debug3("Filling in None default for %s.%s", cls, annotation)
                    setattr(cls, annotation, None)
                else:
                    user_defined_defaults[annotation] = user_defined_default
            # If the current __init__ is auto-generated by dataclass, remove
            # it so that a new one is created with the new defaults
            if _has_dataclass_init(cls):
                log.debug3("Resetting default dataclass init")
                delattr(cls, "__init__")
            cls = dataclasses.dataclass(cls)
            setattr(cls, _USER_DEFINED_DEFAULTS, user_defined_defaults)

        descriptor = _dataobject_to_proto(dataclass_=cls, **kwargs)

        # Create the message class from the dataclass
        proto_class = py_to_proto.descriptor_to_message_class(descriptor)
        _AUTO_GEN_PROTO_CLASSES.append(proto_class)

        # Add enums to the global enums module
        for enum_class in _get_all_enums(proto_class):
            log.debug2("Importing enum [%s]", enum_class.DESCRIPTOR.name)
            enums.import_enum(enum_class)

        # Declare the merged class that binds DataBase to the wrapped class with
        # this generated proto class
        if not isinstance(proto_class, EnumTypeWrapper):
            setattr(cls, "_proto_class", proto_class)
            cls = _make_data_model_class(proto_class, cls)

            # If this was a default-generated dataclass __init__ and there are
            # any oneofs, we need to augment the __init__ to support kwargs for
            # the individual fields
            if _has_dataclass_init(cls) and cls._fields_oneofs_map:
                setattr(cls, "__init__", _make_oneof_init(cls))

        else:
            enums.import_enum(proto_class, cls)
            setattr(cls, "_proto_enum", proto_class)

        # Return the decorated class
        return cls

    # If called without the function invocation, fill in the default argument
    if args and callable(args[0]):
        assert not kwargs, "This shouldn't happen!"
        package = CAIKIT_DATA_MODEL
        return decorator(args[0])

    # Pull the package as an arg or a keyword arg
    if args:
        package = args[0]
        if "package" in kwargs:
            raise TypeError("Got multiple values for argument 'package'")
    else:
        package = kwargs.get("package", CAIKIT_DATA_MODEL)
    return decorator


def render_dataobject_protos(interfaces_dir: str):
    """Write out protobufs files for all proto classes generated from dataobjects
    to the target interfaces directory

    Args:
        interfaces_dir (str): The target directory (must already exist)
    """
    for proto_class in _AUTO_GEN_PROTO_CLASSES:
        proto_class.write_proto_file(interfaces_dir)


def make_dataobject(
    *,
    name: str,
    annotations: Dict[str, type],
    bases: Optional[Iterable[type]] = None,
    attrs: Optional[Dict[str, Any]] = None,
    proto_name: Optional[str] = None,
    **kwargs,
) -> _DataObjectBaseMetaClass:
    """Factory function for creating net-new dataobject classes

    WARNING: This is a power-user feature that should be used with caution since
        dynamically generated dataobject classes have portability issues due to
        the use of global registries.

    Kwargs:
        name (str): The name of the class to create
        annotations (Dict[str, type]): The type annotations for the class
        bases (Optional[Iterable[type]]): Additional base classes beyond
            DataObjectBase
        attrs (Optional[Dict[str, Any]]): Additional class attributes beyond
            __annotations__
        proto_name (Optional[str]): Alternate name to use for the name of
            protobuf message

    Returns:
        dataobject_class (_DataObjectBaseMetaClass): Programmatically created
            class derived from DataObjectBase with the given name and
            annotations
    """
    bases = (DataObjectBase,) + tuple(bases or ())
    attrs = {
        "__annotations__": annotations,
        **(attrs or {}),
    }
    if proto_name is not None:
        kwargs["name"] = proto_name
    return dataobject(**kwargs)(
        _DataObjectBaseMetaClass.__new__(
            _DataObjectBaseMetaClass,
            name=name,
            bases=bases,
            attrs=attrs,
        )
    )


## Implementation Details ######################################################


def _dataobject_to_proto(*args, **kwargs):
    kwargs.setdefault("type_mapping", DATAOBJECT_PY_TO_PROTO_TYPES)
    return _DataobjectConverter(*args, **kwargs).descriptor


class _DataobjectConverter(DataclassConverter):
    """Augment the dataclass converter to be able to pull descriptors from
    existing data objects
    """

    def get_concrete_type(self, entry: Any) -> Any:
        """Also include data model classes and enums as concrete types"""
        unwrapped = self._resolve_wrapped_type(entry)
        if (
            isinstance(unwrapped, type)
            and issubclass(unwrapped, DataBase)
            and entry.get_proto_class() is not None
        ) or hasattr(unwrapped, "_proto_enum"):
            return entry
        return super().get_concrete_type(entry)

    def get_descriptor(self, entry: Any) -> Any:
        """Unpack data model classes and enums to their descriptors"""
        entry = self._resolve_wrapped_type(entry)
        if isinstance(entry, type) and issubclass(entry, DataBase):
            return entry.get_proto_class().DESCRIPTOR
        proto_enum = getattr(entry, "_proto_enum", None)
        if proto_enum is not None:
            return proto_enum.DESCRIPTOR
        return super().get_descriptor(entry)

    def get_optional_field_names(self, entry: Any) -> List[str]:
        """Get the names of any fields which are optional. This will be any
        field that has a user-defined default or is marked as Optional[]
        """
        optional_fields = list(getattr(entry, _USER_DEFINED_DEFAULTS, {}))
        for field_name, field in entry.__dataclass_fields__.items():
            if (
                field_name not in optional_fields
                and self._is_python_optional(field.type) is not None
            ):
                optional_fields.append(field_name)
        return optional_fields

    @staticmethod
    def _is_python_optional(entry: Any) -> Any:
        """Detect if this type is a python optional"""
        if get_origin(entry) is Union:
            args = get_args(entry)
            return type(None) in args


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


def _make_data_model_class(proto_class: Type[_message.Message], cls):
    if issubclass(cls, DataObjectBase):
        _DataBaseMetaClass.parse_proto_descriptor(cls)

    # Recursively make all nested message wrappers
    for nested_message_descriptor in proto_class.DESCRIPTOR.nested_types:
        nested_message_name = nested_message_descriptor.name
        nested_proto_class = getattr(proto_class, nested_message_name)
        setattr(
            cls,
            nested_message_name,
            _make_data_model_class(
                nested_proto_class,
                _DataBaseMetaClass.__new__(
                    _DataBaseMetaClass,
                    name=nested_message_name,
                    bases=(DataBase,),
                    attrs={"_proto_class": getattr(proto_class, nested_message_name)},
                ),
            ),
        )
    for nested_enum_descriptor in proto_class.DESCRIPTOR.enum_types:
        setattr(
            cls,
            nested_enum_descriptor.name,
            getattr(enums, nested_enum_descriptor.name),
        )

    return cls


def _make_oneof_init(cls):
    """Helper to augment a defaulted dataclass __init__ to support kwargs for
    oneof fields
    """
    original_init = cls.__init__
    fields_to_oneofs = cls._fields_to_oneof
    oneofs_to_fields = cls._fields_oneofs_map

    def __init__(self, *args, **kwargs):
        new_kwargs = {}
        to_remove = []
        which_oneof = {}
        for field_name, val in kwargs.items():
            if oneof_name := fields_to_oneofs.get(field_name):
                oneof_pos_idx = list(cls.__dataclass_fields__.keys()).index(oneof_name)
                has_pos_val = len(args) > oneof_pos_idx
                if has_pos_val:
                    error(
                        "<COR09282193E>",
                        TypeError(
                            "Received conflicting oneof args/kwargs for {}/{}".format(
                                oneof_name,
                                field_name,
                            )
                        ),
                    )

                other_oneof_fields = (
                    field
                    for field in [oneof_name] + oneofs_to_fields[oneof_name]
                    if field != field_name
                )
                if any(field in kwargs for field in other_oneof_fields):
                    error(
                        "<COR59933157E>",
                        TypeError(
                            "Received multiple keyword arguments for oneof {}".format(
                                oneof_name,
                            )
                        ),
                    )
                new_kwargs[oneof_name] = val
                to_remove.append(field_name)
                which_oneof[oneof_name] = field_name

        for kwarg in to_remove:
            del kwargs[kwarg]
        kwargs.update(new_kwargs)
        original_init(self, *args, **kwargs)
        # noinspection PyProtectedMember
        setattr(self, _DataBaseMetaClass._WHICH_ONEOF_ATTR, which_oneof)

    return __init__


def _has_dataclass_init(cls) -> bool:
    """When the dataclass decorator adds an __init__ to a class, it adds
    __annotations__ to the init function itself. This function uses that fact to
    detect if the class's __init__ function was generated by @dataclass
    """
    return bool(getattr(cls.__init__, "__annotations__", None)) and not any(
        cls.__init__ is base.__init__ for base in cls.__bases__
    )
