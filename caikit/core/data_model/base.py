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


"""Base classes and functionality for all data structures.
"""

# metaclass-generated field members cannot be detected by pylint
# pylint: disable=no-member

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import base64
import json

# Third Party
from google.protobuf import json_format
from google.protobuf.descriptor import Descriptor, FieldDescriptor, OneofDescriptor
from google.protobuf.internal import type_checkers as proto_type_checkers
from google.protobuf.message import Message as ProtoMessageType

# First Party
import alog

# Local
from ..toolkit.errors import error_handler
from . import enums, json_dict

log = alog.use_channel("DATAM")
error = error_handler.get(log)


class _DataBaseMetaClass(type):
    """Meta class for all structures in the data model."""

    # store a registry of all classes that use this metaclass, i.e.,
    # all classes that extend DataBase. This is used for constructing new
    # instances by name without having to introspect all modules in data_model.
    class_registry = {}

    # This sentinel value is used to determine whether a given attribute is
    # present on a class without doing `getattr` twice in the case where the
    # attribute does exist.
    _MISSING_ATTRIBUTE = "missing attribute"

    # Special attribute used to communicate that the proto fields are forward
    # declared and will be populated after the metaclass has completed
    # construction.
    _FWD_DECL_FIELDS = "__fwd_decl_fields__"

    # Special instance attributes that an instance of a class derived from
    # DataBase may have. These are added to __slots__.
    _BACKEND_ATTR = "_backend"
    _WHICH_ONEOF_ATTR = "_which_oneof"

    # When inferring which field in a oneof a given value should be used for
    # based on the python type, we need to check types in order with bool first,
    # ints next, then floats values that fit a "more flexible" type don't
    # accidentally get assigned to the wrong field. These are the lists of int
    # and bool type values in protobuf.
    _PROTO_TYPE_ORDER = [FieldDescriptor.TYPE_BOOL] + [
        val
        for name, val in vars(FieldDescriptor).items()
        if name.startswith("TYPE_") and "INT" in name
    ]

    def __new__(mcs, name, bases, attrs):
        """When constructing a new data model class, we set the 'fields' class variable from the
        protobufs descriptor and then set the '__slots__' magic class attribute to fields.  This
        provides two benefits: (a) performance is improved since the classes only need to know
        about these attributes (b) it helps to enforce that all member variables in these classes
        are described in the protobufs.

        Note:  If you want to add a variable for internal use that is not described in the
        protobufs, it can be named in the tuple class variable _private_slots and will
        automatically be added to __slots__.
        """

        # Protobufs fields can be divided into these categories, which are used
        # to automatically determine appropriate behavior in a number of methods
        attrs["full_name"] = name
        attrs["fields_enum_map"] = {}
        attrs["fields_enum_rev"] = {}
        attrs["_fields_oneofs_map"] = {}
        attrs["_fields_to_oneof"] = {}
        attrs["_fields_map"] = ()
        attrs["_fields_message"] = ()
        attrs["_fields_message_repeated"] = ()
        attrs["_fields_enum"] = ()
        attrs["_fields_enum_repeated"] = ()
        attrs["_fields_primitive"] = ()
        attrs["_fields_primitive_repeated"] = ()

        # Look for the set of fields either from a predefined protobuf class or
        # from a forward declaration from @dataobject
        fields = ()
        proto_class = None
        if name not in ["DataBase", "DataObjectBase"]:
            # Look for a precompiled proto class and if found, parse its
            # descriptor
            proto_class = attrs.get("_proto_class")
            if proto_class is not None:
                all_oneof_fields = [
                    field.name
                    for oneof in proto_class.DESCRIPTOR.oneofs
                    for field in oneof.fields
                ]
                fields = tuple(
                    (
                        field
                        for field in proto_class.DESCRIPTOR.fields_by_name
                        if field not in all_oneof_fields
                    )
                ) + tuple(proto_class.DESCRIPTOR.oneofs_by_name)

            # Otherwise, we need to get the fields from a "special" attribute
            else:
                fields = attrs.pop(mcs._FWD_DECL_FIELDS, None)
                log.debug4(
                    "Using dataclass forward declaration fields %s for %s", fields, name
                )
                error.value_check(
                    "<COR49310991E>",
                    fields is not None,
                    "No proto class found for {}",
                    name,
                )
        attrs["fields"] = fields
        attrs["_proto_class"] = proto_class

        # Look if any private slots are declared as class variables
        private_slots = attrs.setdefault("_private_slots", ())

        # Class slots are fields + private slots, this prevents other
        # member attributes from being set and also improves performance
        attrs["__slots__"] = tuple(
            [f"_{field}" for field in fields]
            + list(private_slots)
            + [mcs._BACKEND_ATTR, mcs._WHICH_ONEOF_ATTR]
        )

        # Create the instance of the type
        instance = super().__new__(mcs, name, bases, attrs)

        # If there's a valid proto class, perform proto descriptor parsing
        if proto_class is not None:
            mcs.parse_proto_descriptor(instance)

        # Return the constructed class instance
        return instance

    @classmethod
    def parse_proto_descriptor(mcs, cls):
        """Encapsulate the logic for parsing the protobuf descriptor here. This
        allows the parsing to be done as a post-process after metaclass
        initialization
        """

        # use the fully qualified protobuf name to avoid conflicts with
        # nested messages that have matching names
        cls.full_name = cls._proto_class.DESCRIPTOR.full_name

        # preserve old fields for _make_property_getter later
        old_fields = cls.fields

        # overwrite to only have proto-specific fields present
        cls.fields = tuple(cls._proto_class.DESCRIPTOR.fields_by_name)

        # map from all enum fields to their enum classes
        # note: enums are also primitives, these overlap
        cls.fields_enum_map = {
            field.name: getattr(enums, field.enum_type.name)
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.enum_type is not None
        }

        cls.fields_enum_rev = {
            field.name: getattr(enums, field.enum_type.name + "Rev")
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.enum_type is not None
        }

        # mapping of all oneofs and the fields that are part of them
        # NOTE: protobuf makes an interesting use of oneof to wrap types that
        #   should be explicitly optional. We don't want to consider these
        #   oneofs in the general oneof handling.

        # Sort the names of the fields in this map to ensure that ordering is
        # correct such that bool < int < float

        cls._fields_oneofs_map = {
            oneof_name: mcs._sorted_oneof_field_names(oneof)
            for oneof_name, oneof in cls._proto_class.DESCRIPTOR.oneofs_by_name.items()
            if len(oneof.fields) != 1 or oneof.name != f"_{oneof.fields[0].name}"
        }
        cls._fields_to_oneof = {
            field_name: oneof_name
            for (oneof_name, oneof_fields) in cls._fields_oneofs_map.items()
            for field_name in oneof_fields
        }

        # all repeated fields
        fields_repeated = tuple(
            field.name
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.label == field.LABEL_REPEATED
        )

        # all messages, repeated or not
        _fields_message_all = tuple(
            field.name
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.type == field.TYPE_MESSAGE
        )

        # all enums, repeated or not
        _fields_enum_all = tuple(
            field.name
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.enum_type is not None
        )

        # all fields of type map
        cls._fields_map = tuple(
            field.name
            for field in cls._proto_class.DESCRIPTOR.fields
            if field.message_type and field.message_type.GetOptions().map_entry
        )

        # all primitives, repeated or not
        _fields_primitive_all = (
            frozenset(cls.fields)
            .difference(cls._fields_map)
            .difference(_fields_message_all)
            .difference(_fields_enum_all)
        )

        # messages that are not repeated
        cls._fields_message = frozenset(_fields_message_all).difference(fields_repeated)

        # messages that are repeated
        cls._fields_message_repeated = frozenset(fields_repeated).intersection(
            _fields_message_all
        )

        # enums that are not repeated
        cls._fields_enum = frozenset(_fields_enum_all).difference(fields_repeated)

        # enums that are repeated
        cls._fields_enum_repeated = frozenset(_fields_enum_all).intersection(
            fields_repeated
        )

        # primitives that are not repeated
        cls._fields_primitive = frozenset(_fields_primitive_all).difference(
            fields_repeated
        )

        # primitives that are repeated
        cls._fields_primitive_repeated = frozenset(fields_repeated).intersection(
            _fields_primitive_all
        )

        # Update the global class and proto registries
        # NOTE: Explicitly not respecting metaclass inheritance so single
        #   registry shared for all
        _DataBaseMetaClass.class_registry[cls.full_name] = cls

        # Add properties that use the underlying backend. Also add fields that
        # existed in old_fields for supporting oneofs
        # see https://github.com/caikit/caikit/pull/107 for details
        for field in set(cls.fields + tuple(old_fields)):
            # If the field is the name of a field within a oneof and it was not
            # in the old fields, the data is held under the oneof's name if this
            # is the set value for the oneof
            if oneof_name := cls._fields_to_oneof.get(field):
                setattr(cls, field, mcs._make_property_getter(field, oneof_name))

            # If the field is a plain field or the name of a oneof, it will be
            # accessed directly
            else:
                setattr(cls, field, mcs._make_property_getter(field))

        # If there is not already an __init__ function defined, make one
        current_init = cls.__init__
        if current_init is None or current_init is DataBase.__init__:
            setattr(cls, "__init__", mcs._make_init(cls.fields))

    @classmethod
    def _make_property_getter(mcs, field, oneof_name=None):
        """This helper creates an @property attribute getter for the given field

        NOTE: This needs to live as a standalone function in order for the given
            field name to be properly bound to the closure for the attrs
        """
        private_name = f"_{field}" if oneof_name is None else oneof_name

        def _property_getter(self):
            # Check to see if the private name is defined and just return it if
            # it is
            current = getattr(self, private_name, mcs._MISSING_ATTRIBUTE)
            if current is not mcs._MISSING_ATTRIBUTE:
                return current

            # If not currently set, delegate to the backend
            backend = self.backend
            if backend is None:
                error(
                    "<COR66616239E>",
                    AttributeError(
                        f"{type(self)} missing attribute {field} and no backend set"
                    ),
                )
            attr_val = backend.get_attribute(self.__class__, field)
            if isinstance(attr_val, self.__class__.OneofFieldVal):
                log.debug2("Got a OneofFieldVal from the backend")
                assert field in self.__class__._fields_oneofs_map
                self._get_which_oneof_dict()[field] = attr_val.which_oneof
                attr_val = attr_val.val

            # If the backend says that this attribute should be cached, set it
            # as an attribute on the class
            if backend.cache_attribute(field, attr_val):
                setattr(self, field, attr_val)

            # Return the value found by the backend
            return attr_val

        # If this is a oneof, add an extra layer of wrapping to check
        # which_oneof before returning a valid result
        if oneof_name:

            def _oneof_property_getter(self):
                if self.which_oneof(oneof_name) == field:
                    return _property_getter(self)

            return property(_oneof_property_getter)

        return property(_property_getter)

    @staticmethod
    def _make_init(fields):
        """This helper creates an __init__ function for a class which has the
        arguments for all the fields and just sets them as instance attributes.
        """

        # Format and preserve docstring
        docstring = """Construct with arguments for each field on the object
            Args:
                {}
        """.format(
            "\n    ".join(fields)
        )

        def __init__(self, *args, **kwargs):
            num_args = len(args)
            num_kwargs = len(kwargs)
            num_fields = len(fields)
            used_fields = []

            # If the proto has oneofs, set up which_oneof
            which_oneof = {}
            cls = self.__class__
            if cls._fields_oneofs_map:
                setattr(self, _DataBaseMetaClass._WHICH_ONEOF_ATTR, which_oneof)

            if num_args + num_kwargs > num_fields:
                error(
                    "<COR71444420E>",
                    TypeError(f"Too many arguments given. Args are: {fields}"),
                )

            if num_args > 0:  # Do a quick check for performance reason
                for i, field_val in enumerate(args):
                    field_name = fields[i]
                    setattr(self, field_name, field_val)
                    used_fields.append(field_name)

            if num_kwargs > 0:  # Do a quick check for performance reason
                for field_name, field_val in kwargs.items():
                    # If this is a oneof field, alias to the oneof name
                    if oneof_name := cls._fields_to_oneof.get(field_name):
                        which_oneof[oneof_name] = field_name
                        field_name = oneof_name

                    if (
                        field_name not in fields
                        and field_name not in cls._fields_oneofs_map
                    ):
                        error(
                            "<COR71444421E>", TypeError(f"Unknown field {field_name}")
                        )
                    elif field_name in used_fields:
                        error(
                            "<COR71444422E>",
                            TypeError(f"Got multiple values for field {field_name}"),
                        )
                    setattr(self, field_name, field_val)
                    used_fields.append(field_name)

            # Default all unspecified fields to None
            if num_fields > 0:  # Do a quick check for performance reason
                for field_name in fields:
                    if (
                        field_name not in used_fields
                        and field_name not in cls._fields_to_oneof
                    ):
                        setattr(self, field_name, None)

        # Set docstring to the method explicitly
        setattr(__init__, "__doc__", docstring)
        return __init__

    @classmethod
    def _sorted_oneof_field_names(mcs, oneof: OneofDescriptor) -> List[str]:
        """Helper to get the list of oneof fields while ensuring field names are
        sorted such that bool < int < float. This ensures that when iterating
        fields for which_oneof inference, lower-precedence types take
        precedence.
        """
        return [
            field.name
            for field in sorted(
                oneof.fields,
                key=lambda fld: mcs._PROTO_TYPE_ORDER.index(fld.type)
                if fld.type in mcs._PROTO_TYPE_ORDER
                else len(mcs._PROTO_TYPE_ORDER),
            )
        ]


class DataBase(metaclass=_DataBaseMetaClass):
    """Base class for all structures in the data model.

    Notes:
        All leaves in the hierarchy of derived classes should have a corresponding protobufs class
        defined in the interface definitions.  If not, an exception will be thrown at runtime.
    """

    @dataclass
    class OneofFieldVal:
        """Helper struct that backends can use to return information about
        values in oneofs along with which of the oneofs is currently valid
        """

        val: Any
        which_oneof: str

    def __setattr__(self, name, val):
        """Handle attribute setting for oneofs and named fields with delegation
        to backends as needed
        """
        # If setting a oneof directly, remove any oneof information
        cls = self.__class__
        if name in cls._fields_oneofs_map:
            self._get_which_oneof_dict().pop(name, None)

        # If this is the name of a oneof field, set the oneof itself
        if oneof_name := cls._fields_to_oneof.get(name):
            self._get_which_oneof_dict()[oneof_name] = name
            name = oneof_name

        # If attempting to set one of the named fields or a oneof, instead set
        # the private version of the attribute.
        if name in cls.fields or name in cls._fields_oneofs_map:
            super().__setattr__(f"_{name}", val)
        else:
            super().__setattr__(name, val)

    @classmethod
    def get_proto_class(cls) -> Type[ProtoMessageType]:
        return cls._proto_class

    @classmethod
    def get_field_message_type(cls, field_name: str) -> Optional[Type["DataBase"]]:
        """Get the data model class for the given field if the field is a
        message or a repeated message

        Args:
            field_name (str): Field name to check (AttributeError raised if name
                is invalid)

        Returns:
            data_model_type:  Type[DataBase]
                The data model class type for the given field
        """
        if field_name not in cls.fields:
            raise AttributeError(f"Invalid field {field_name}")
        if (
            field_name in cls._fields_message
            or field_name in cls._fields_message_repeated
        ):
            return cls.get_class_for_proto(
                cls.get_proto_class().DESCRIPTOR.fields_by_name[field_name].message_type
            )
        return None

    @classmethod
    def from_backend(cls, backend):
        instance = cls.__new__(cls)
        setattr(instance, _DataBaseMetaClass._BACKEND_ATTR, backend)
        return instance

    @property
    def backend(self) -> Optional["DataModelBackendBase"]:
        return getattr(self, _DataBaseMetaClass._BACKEND_ATTR, None)

    def which_oneof(self, oneof_name: str) -> Optional[str]:
        """Get the name of the oneof field set for the given oneof or None if no
        field is set
        """
        # If the internal dict is already set, use that information
        which_oneof = self._get_which_oneof_dict()
        if current_val := which_oneof.get(oneof_name):
            return current_val

        # Get the current value for the oneof and introspect which field its
        # type matches
        oneof_val = getattr(self, oneof_name)

        # Re-check in case the getattr pulled a OneofFieldVal that populated the
        # which_oneof dict with knowledge from the backend
        if current_val := which_oneof.get(oneof_name):
            return current_val

        # Try to figure out the field based on the type
        which_field = self._infer_which_oneof(oneof_name, oneof_val)
        if which_field is not None:
            which_oneof[oneof_name] = which_field
        return which_field

    @classmethod
    def _infer_which_oneof(cls, oneof_name: str, oneof_val: Any) -> Optional[str]:
        """Check each candidate field within the oneof to see if it's a type
        match

        NOTE: In the case where fields within a oneof have the same type, the
          first field whose type matches will be used!
        """
        # NOTE: The list of field names are guaranteed to be sorted so that
        #   bool < int < float
        for field_name in cls._fields_oneofs_map.get(oneof_name, []):
            if cls._is_valid_type_for_field(field_name, oneof_val):
                return field_name

    def _get_which_oneof_dict(self) -> Dict[str, str]:
        which_oneof = getattr(self, _DataBaseMetaClass._WHICH_ONEOF_ATTR, None)
        if which_oneof is None:
            super().__setattr__(_DataBaseMetaClass._WHICH_ONEOF_ATTR, {})
            which_oneof = getattr(self, _DataBaseMetaClass._WHICH_ONEOF_ATTR)
        return which_oneof

    @classmethod
    def _is_valid_type_for_field(cls, field_name: str, val: Any) -> bool:
        """Check whether the given value is valid for the given field"""
        field_descriptor = cls._proto_class.DESCRIPTOR.fields_by_name[field_name]

        if val is None:
            return False

        # If it's a data object or an enum and the descriptors match, it's a
        # good type
        if (
            isinstance(val, DataBase)
            and field_descriptor.message_type == val.get_proto_class().DESCRIPTOR
        ) or (
            isinstance(val, Enum)
            and field_descriptor.enum_type == val.get_proto_class().DESCRIPTOR
        ):
            return True

        # If it's a data object or an enum and the descriptors don't match, it's
        # a bad type
        if field_descriptor.type in [
            field_descriptor.TYPE_MESSAGE,
            field_descriptor.TYPE_ENUM,
        ]:
            return False

        # If the field is a bool field, only accept python bools. Proto is ok to
        # accept ints, but we are stricter than that.
        if field_descriptor.type == field_descriptor.TYPE_BOOL:
            return isinstance(val, bool)

        # If it's a primitive, use protobuf type checkers
        checker = proto_type_checkers.GetTypeChecker(field_descriptor)
        try:
            checker.CheckValue(val)
            return True
        except TypeError:
            pass
        return False

    @classmethod
    def from_binary_buffer(cls, buf):
        """Builds the data model object out of the binary string

        Args:
            buf: The binary buffer containing a serialized protobufs message
        Returns:
            A data model object instantiated from the protobufs message deserialized out of `buf`
        """
        proto_message = cls.get_proto_class()()
        proto_message.ParseFromString(buf)

        return cls.from_proto(proto_message)

    @classmethod
    def from_proto(cls, proto):
        """Build a DataBase from protobufs.

        Args:
            proto: A protocol buffer to serialize from.
        Returns:
            protobufs: A DataBase object.
        """
        error.type_check("<COR45207671E>", ProtoMessageType, proto=proto)
        if cls._proto_class.DESCRIPTOR.name != proto.DESCRIPTOR.name:
            error(
                "<COR71783894E>",
                ValueError(
                    "class name `{}` does not match protobufs name `{}`".format(
                        cls._proto_class.DESCRIPTOR.name, proto.DESCRIPTOR.name
                    )
                ),
            )

        kwargs = {}
        for field in cls.fields:
            try:
                proto_attr = getattr(proto, field)
            except AttributeError:
                error(
                    "<COR71783905E>",
                    AttributeError(
                        "protobufs `{}` does not have field `{}`".format(
                            proto.DESCRIPTOR.name, field
                        )
                    ),
                )

            if field in cls._fields_primitive or field in cls._fields_enum:
                # special case for oneofs
                if field not in cls._fields_to_oneof or proto.HasField(field):
                    kwargs[field] = proto_attr
            elif (
                field in cls._fields_primitive_repeated
                or field in cls._fields_enum_repeated
            ):
                kwargs[field] = list(proto_attr)

            elif field in cls._fields_map:
                kwargs[field] = {}
                for key, value in proto_attr.items():
                    # Similar to filling; if our value is a non-primitive, i.e., a message,
                    # we need to look up the data model class attached to it.
                    if hasattr(value, "DESCRIPTOR"):
                        contained_class = cls.get_class_for_proto(value)
                        kwargs[field][key] = contained_class.from_proto(value)
                    # If it's not a message, the value can be left alone, i.e., it's a primitive
                    else:
                        kwargs[field][key] = value

            elif field in cls._fields_message:
                if proto.HasField(field):
                    if proto_attr.DESCRIPTOR.full_name == "google.protobuf.Struct":
                        kwargs[field] = json_dict.struct_to_dict(proto_attr)
                    else:
                        contained_class = cls.get_class_for_proto(proto_attr)
                        contained_obj = contained_class.from_proto(proto_attr)
                        kwargs[field] = contained_obj

            elif field in cls._fields_message_repeated:
                elements = []
                contained_class = None
                for item in proto_attr:
                    if item.DESCRIPTOR.full_name == "google.protobuf.Struct":
                        elements.append(json_dict.struct_to_dict(item))
                    else:
                        if contained_class is None:
                            contained_class = cls.get_class_for_proto(item)
                        elements.append(contained_class.from_proto(item))
                kwargs[field] = elements

            else:
                error(
                    "<COR71783815E>",
                    AttributeError(
                        "field `{}` is not a protobufs primitive, message, map or "
                        "repeated".format(field)
                    ),
                )

        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_str):
        """Build a DataBase from a given JSON string. Use google's protobufs.json_format for
        deserialization

        Args:
            json_str (str or dict): A stringified JSON specification/dict of the
                data_model

        Returns:
            caikit.core.data_model.DataBase: A DataBase object.
        """
        # Get protobufs class required for parsing
        error.type_check("<COR91037250E>", str, dict, json_str=json_str)
        if isinstance(json_str, dict):
            # Convert dict object to a JSON string
            json_str = json.dumps(json_str)

        try:
            # Parse given JSON into google.protobufs.pyext.cpp_message.GeneratedProtocolMessageType
            parsed_proto = json_format.Parse(
                json_str, cls.get_proto_class()(), ignore_unknown_fields=False
            )

            # Use from_proto to return the DataBase object from the parsed proto
            return cls.from_proto(parsed_proto)

        except json_format.ParseError as ex:
            error("<COR90619980E>", ValueError(ex))

    def to_proto(self):
        """Return a new protobufs populated with the information in this data structure."""
        # get the name of the protobufs class
        proto_class = self.__class__.get_proto_class()
        if proto_class is None:
            error(
                "<COR71783827E>",
                AttributeError(
                    "protobufs not found for class `{}`".format(self.__class__)
                ),
            )

        # create the protobufs and call fill_proto to populate it
        return self.fill_proto(proto_class())

    def to_binary_buffer(self):
        """Returns a binary buffer with a serialized protobufs message of this data model"""
        return self.to_proto().SerializeToString()

    def fill_proto(self, proto):
        """Populate a protobufs with the values from this data model object.

        Args:
            proto: A protocol buffer to be populated.
        Returns:
            protobufs: The filled protobufs.

        Notes:
            The protobufs is filled in place, so the argument and the return
            value are the same at the end of this call.
        """
        for field in self.fields:
            try:
                attr = getattr(self, field)

            except AttributeError:
                error(
                    "<COR71783840E>",
                    AttributeError(
                        "class `{}` has no attribute `{}` but it is in the protobufs".format(
                            self.__class__.__name__, field
                        )
                    ),
                )

            if attr is None:
                continue

            if field in self._fields_primitive:
                setattr(proto, field, attr)
            elif field in self._fields_enum:
                if isinstance(attr, Enum):
                    setattr(proto, field, attr.value)
                else:
                    setattr(proto, field, attr)
            elif field in self._fields_map:
                subproto = getattr(proto, field)
                for key, value in attr.items():
                    # If our values aren't primitives, the subproto will have a DESCRIPTOR;
                    # in this case we need to fill down recursively, i.e., this is a
                    # protobufs message map container
                    if hasattr(subproto[key], "DESCRIPTOR"):
                        value.fill_proto(subproto[key])
                    # Otherwise we have a protobufs scalar map container, and we can set the
                    # primitive value like a normal dictionary.
                    else:
                        subproto[key] = value
            elif (
                field in self._fields_primitive_repeated
                or field in self._fields_enum_repeated
            ):
                subproto = getattr(proto, field)
                subproto.extend(attr)

            elif field in self._fields_message:
                subproto = getattr(proto, field)
                if subproto.DESCRIPTOR.full_name == "google.protobuf.Struct":
                    subproto.CopyFrom(
                        json_dict.dict_to_struct(attr, subproto.__class__)
                    )
                else:
                    attr.fill_proto(subproto)

            elif field in self._fields_message_repeated:
                subproto = getattr(proto, field)
                for item in attr:
                    elem_type = subproto.add()
                    if isinstance(item, dict):
                        elem_type.CopyFrom(
                            json_dict.dict_to_struct(item, elem_type.__class__)
                        )
                    else:
                        item.fill_proto(elem_type)
            else:
                error(
                    "<COR71783852E>",
                    AttributeError(
                        "field `{}` is not a protobufs primitive, message or repeated".format(
                            field
                        )
                    ),
                )

        return proto

    def to_dict(self) -> dict:
        """Convert to a dictionary representation."""
        # maintain a list of fields to convert to dict, special handling for oneofs
        fields_to_dict = []
        for field in self.fields:
            if (
                not field in self._fields_to_oneof
                or self.which_oneof(self._fields_to_oneof[field]) == field
            ):
                fields_to_dict.append(field)
        return {field: self._field_to_dict_element(field) for field in fields_to_dict}

    def to_kwargs(self) -> dict:
        """Convert to flat dictionary representation. (Like .to_dict, but not recursive)
        This keeps the attribute names of any fields backed by oneofs, instead of using the
        internal oneof field name
        """
        fields_to_dict = []
        for field in self.fields:
            if field not in self._fields_to_oneof:
                fields_to_dict.append(field)
            else:
                fields_to_dict.append(self._fields_to_oneof[field])
        return {field: getattr(self, field) for field in fields_to_dict}

    def to_json(self, **kwargs) -> str:
        """Convert to a json representation."""

        def _default_serialization_overrides(obj):
            """Default handler for nonserializable objects; currently this only handles bytes."""
            if isinstance(obj, bytes):
                return base64.encodebytes(obj).decode("utf-8")
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        if "default" not in kwargs:
            kwargs["default"] = _default_serialization_overrides
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self):
        """Human-friendly representation."""
        return self.to_json(indent=2, ensure_ascii=False)

    def _field_to_dict_element(self, field):
        """Convert field into a representation that can be placed into a dictionary.  Recursively
        calls to_dict on other data model objects.
        """
        try:
            attr = getattr(self, field)

        except AttributeError:
            error(
                "<COR71783864E>",
                AttributeError(
                    "class `{}` has no attribute `{}` but it is in the protobufs".format(
                        self.__class__.__name__, field
                    )
                ),
            )

        # if field is None, assume it's unset and just return None
        if attr is None:
            return None

        if field in self._fields_enum:
            # if field is an enum, do the reverse lookup from int -> str
            enum_rev = self.fields_enum_rev.get(field)
            if enum_rev is not None:
                return (
                    enum_rev[attr.value] if isinstance(attr, Enum) else enum_rev[attr]
                )

        if field in self._fields_enum_repeated:
            # if field is an enum, do the reverse lookup from int -> str
            enum_rev = self.fields_enum_rev.get(field)
            if enum_rev is not None:
                return [enum_rev[item] for item in attr]

        # if field is a primitive, just return it to be placed in dict
        if field in self._fields_primitive or field in self._fields_primitive_repeated:
            return attr

        def _recursive_to_dict(_attr):
            if isinstance(_attr, dict):
                return {key: _recursive_to_dict(value) for key, value in _attr.items()}
            if isinstance(_attr, list):
                return [_recursive_to_dict(listitem) for listitem in _attr]
            if isinstance(_attr, DataBase):
                return _attr.to_dict()

            return _attr

        # If field is an object in out data model/map/list call to_dict recursively on each element
        if (
            field in self._fields_map
            or field in self._fields_message
            or field in self._fields_message_repeated
        ):
            return _recursive_to_dict(attr)

        # fallback to the string representation
        return str(attr)

    @staticmethod
    def get_class_for_proto(
        proto: Union[Descriptor, ProtoMessageType]
    ) -> Type["DataBase"]:
        """Look up the data model class corresponding to the given protobuf

        If no data model is found, this raises an AttributeError

        Args:
            proto (Union[Descriptor, ProtoMessageType])
                The proto name or descriptor to look up against

        Returns:
            dm_class (Type[DataBase]): The data model class corresponding to the
                given protobuf
        """
        error.type_check(
            "<COR46446770E>",
            Descriptor,
            ProtoMessageType,
            proto=proto,
        )
        proto_full_name = (
            proto.full_name
            if isinstance(proto, Descriptor)
            else proto.DESCRIPTOR.full_name
        )
        cls = _DataBaseMetaClass.class_registry.get(proto_full_name)
        if cls is None:
            error(
                "<COR71783879E>",
                AttributeError(
                    "no data model class found in registry for protobufs named `{}`".format(
                        proto_full_name
                    )
                ),
            )

        return cls

    @staticmethod
    def get_class_for_name(class_name: str) -> Type["DataBase"]:
        """Look up the data model class corresponding to the given name

        This lookup attempts to encode various naming conventions that might be
        used, but it can fail in multiple ways:

        1. No class with the given name is known
        2. Multiple classes with the same name, but different qualified parents
           are found

        A ValueError will be raised if either of the above happens

        Args:
            class_name (str)
                The name of the class either as a fully-qualified protobuf name
                or as the unqualified class name

        Returns:
            dm_class (Type[DataBase]): The data model class corresponding to the
                given protobuf
        """
        dm_class = _DataBaseMetaClass.class_registry.get(class_name)
        if dm_class is not None:
            return dm_class
        matching_classes = [
            (full_name, dm_class)
            for full_name, dm_class in _DataBaseMetaClass.class_registry.items()
            if full_name.rpartition(".")[-1] == class_name
        ]
        if len(matching_classes) == 1:
            return matching_classes[0][1]
        if len(matching_classes) > 1:
            error(
                "<COR02514290E>",
                ValueError(
                    "Conflicting data model classes for [{}]: {}".format(
                        class_name, [match[0] for match in matching_classes]
                    )
                ),
            )
        error(
            "<COR99562895E>",
            ValueError(f"No data model class match for {class_name}"),
        )
