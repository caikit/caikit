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
"""Data structures for Rule Based Response.
"""

# Standard
from typing import Dict, List, Union
import re

# Third Party
import google.protobuf.message
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber, OneofField
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject, protobufs
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import text_primitives

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class PropertyListValueBool(DataObjectBase):
    val: Annotated[List[bool], FieldNumber(1)]

    def __init__(self, val):
        """
        Args:
            val:  list(bool)
                The list of the given type.
        """
        error.type_check_all("<NLP86314799E>", bool, val=val)
        self.val = val


@dataobject(package="caikit_data_model.nlp")
class PropertyListValueStr(DataObjectBase):
    val: Annotated[List[str], FieldNumber(1)]

    def __init__(self, val):
        """
        Args:
            val:  list(str)
                The list of the given type.
        """
        error.type_check_all("<NLP08840091E>", str, val=val)
        self.val = val


@dataobject(package="caikit_data_model.nlp")
class PropertyListValueFloat(DataObjectBase):
    val: Annotated[List[float], FieldNumber(1)]

    def __init__(self, val):
        """
        Args:
            val:  list(float)
                The list of the given type.
        """
        error.type_check_all("<NLP11024978E>", float, val=val)
        self.val = val


@dataobject(package="caikit_data_model.nlp")
class PropertyListValueInt(DataObjectBase):
    val: Annotated[List[np.int32], FieldNumber(1)]

    def __init__(self, val):
        """
        Args:
            val:  list(int)
                The list of the given type.
        """
        error.type_check_all("<NLP21931919E>", int, val=val)
        self.val = val


@dataobject(package="caikit_data_model.nlp")
class PropertyListValueSpan(DataObjectBase):
    val: Annotated[List[text_primitives.Span], FieldNumber(1)]

    def __init__(self, val):
        """
        Args:
            val:  list(Span)
                The list of the given type.
        """
        error.type_check_all("<NLP84585941E>", text_primitives.Span, val=val)
        self.val = val


@dataobject(package="caikit_data_model.nlp")
class ViewPropertyValue(DataObjectBase):
    """Value type encoding the acceptable types for a view property value."""

    _private_slots = ("value", "_active_val")

    value: Union[
        Annotated[str, OneofField("str_val"), FieldNumber(1)],
        Annotated[float, OneofField("float_val"), FieldNumber(2)],
        Annotated[np.int32, OneofField("int_val"), FieldNumber(3)],
        Annotated[bool, OneofField("bool_val"), FieldNumber(4)],
        Annotated[text_primitives.Span, OneofField("span_val"), FieldNumber(5)],
        Annotated[PropertyListValueStr, OneofField("list_str_val"), FieldNumber(6)],
        Annotated[PropertyListValueFloat, OneofField("list_float_val"), FieldNumber(7)],
        Annotated[PropertyListValueInt, OneofField("list_int_val"), FieldNumber(8)],
        Annotated[PropertyListValueBool, OneofField("list_bool_val"), FieldNumber(9)],
        Annotated[PropertyListValueSpan, OneofField("list_span_val"), FieldNumber(10)],
    ]

    @alog.logged_function(log.debug3)
    def __init__(self, val):
        """
        Args:
            val:  str|float|int|bool|dm.Span|list(...)
                The raw value to use for this entry.
        """
        if not isinstance(
            val, protobufs.ViewPropertyValue
        ) and not ViewPropertyValue.is_valid_aql_primitive(val):
            error(
                "<NLP98988709E>",
                TypeError(
                    "Bad type for ViewPropertyValue: {}".format(type(val).__name__)
                ),
            )

        if isinstance(val, protobufs.ViewPropertyValue):
            log.debug2("Handling input protobufs.ViewPropertyValue")
            val_field = val.WhichOneof("value")
            if val_field is None:
                val = None
            else:
                val = getattr(val, val_field)

            # If the field value is a nested message (i.e. a list wrapper), we
            # need to convert that too.
            if isinstance(val, google.protobuf.message.Message):
                submsg_class_name, submsg_class = self._get_class_for_proto(val)
                log.debug2(
                    "Sub value is a message. Converting to %s" % submsg_class_name
                )
                if (
                    isinstance(submsg_class, type)
                    and issubclass(DataObjectBase)
                    and submsg_class_name == "Span"
                ):
                    val = submsg_class.from_proto(val)
                else:
                    val = submsg_class.from_proto(val).val

        # Store the value directly
        self.value = val

        # Keep a reference to the active type val's key name
        self._active_val = None

        # Default all field options to None
        self.str_val = None
        self.float_val = None
        self.int_val = None
        self.bool_val = None
        self.span_val = None
        self.list_str_val = None
        self.list_float_val = None
        self.list_int_val = None
        self.list_bool_val = None
        self.list_span_val = None

        # Basic types
        # NOTE: isinstance(True, int) == True, so the bool check must come first
        if isinstance(val, bool):
            self.bool_val = val
            self._active_val = "bool_val"
        elif isinstance(val, str):
            self.str_val = val
            self._active_val = "str_val"
        elif isinstance(val, float):
            self.float_val = val
            self._active_val = "float_val"
        elif isinstance(val, int):
            self.int_val = val
            self._active_val = "int_val"
        elif isinstance(val, text_primitives.Span):
            self.span_val = val
            self._active_val = "span_val"

        # Repeated types
        elif isinstance(val, list) and val:
            first_element = val[0]
            if isinstance(first_element, bool):
                self.list_bool_val = PropertyListValueBool(val)
                self._active_val = "list_bool_val"
            elif isinstance(first_element, str):
                self.list_str_val = PropertyListValueStr(val)
                self._active_val = "list_str_val"
            elif isinstance(first_element, float):
                self.list_float_val = PropertyListValueFloat(val)
                self._active_val = "list_float_val"
            elif isinstance(first_element, int):
                self.list_int_val = PropertyListValueInt(val)
                self._active_val = "list_int_val"
            elif isinstance(first_element, text_primitives.Span):
                self.list_span_val = PropertyListValueSpan(val)
                self._active_val = "list_span_val"

    @alog.logged_function(log.debug3)
    def to_dict(self):
        """Override the default to_dict to return the raw python value. This allows both to_dict
        and to_json to appear as expected, rather than with nasty nested fields.
        """
        # Handle Span.to_dict()
        if hasattr(self.value, "to_dict"):
            return self.value.to_dict()

        # Handle nested lists of
        elif isinstance(self.value, list):
            return [v.to_dict() if hasattr(v, "to_dict") else v for v in self.value]

        # Handle raw primitives. Note that this does not return a dict and is
        # therefore only for the recursion.
        else:
            return self.value

    @classmethod
    @alog.logged_function(log.debug3)
    def from_proto(cls, proto):
        """Override the default from_proto behavior to handle all of the differnt types that could
        be present in the proto message.
        """
        error.type_check("<NLP67382567E>", google.protobuf.message.Message, proto=proto)

        val_field = proto.WhichOneof("value")
        if val_field is None:
            proto_field = None
        else:
            proto_field = getattr(proto, val_field)

        if isinstance(proto_field, protobufs.Span):
            return cls(text_primitives.Span(proto_field.begin, proto_field.end))

        elif isinstance(
            proto_field,
            (
                protobufs.PropertyListValueInt,
                protobufs.PropertyListValueFloat,
                protobufs.PropertyListValueStr,
                protobufs.PropertyListValueBool,
            ),
        ):
            return cls(list(proto_field.val))

        elif isinstance(proto_field, protobufs.PropertyListValueSpan):
            return cls([text_primitives.Span.from_proto(v) for v in proto_field.val])

        else:
            return cls(proto_field)

    @staticmethod
    def is_valid_aql_primitive(val):
        return (
            val is None
            or isinstance(val, (int, float, str, bool, text_primitives.Span))
            or (
                isinstance(val, list)
                and all(ViewPropertyValue.is_valid_aql_primitive(v) for v in val)
            )
        )

    @classmethod
    def from_json(cls, json_str):
        error(
            "<COR88809173E>",
            NotImplementedError("This is not available in this module."),
        )


@dataobject(package="caikit_data_model.nlp")
class ViewProperty(DataObjectBase):
    aql_property: Annotated[Dict[str, ViewPropertyValue], FieldNumber(1)]

    """Individual property of a given view.  Represented as a dict of key-value pairs."""

    @alog.logged_function(log.debug3)
    def __init__(self, aql_property):
        """
        Args:
            aql_property:  dict
                Dictionary of key/val pairs for the view property.
        """
        error.type_check("<NLP94850566E>", dict, aql_property=aql_property)
        if not all(isinstance(k, str) for k in aql_property.keys()):
            error("<NLP26807632E>", TypeError("`aql_property` key is not a string"))
        if not all(
            (
                isinstance(v, ViewPropertyValue)
                or ViewPropertyValue.is_valid_aql_primitive(v)
            )
            for v in aql_property.values()
        ):
            error(
                "<NLP72611789E>",
                TypeError("`aql_property` value is not a valid AQL type"),
            )

        # Convert values if necessary
        def convert_value(val):
            if not isinstance(val, ViewPropertyValue):
                return ViewPropertyValue(val)
            return val

        self.aql_property = {k: convert_value(v) for k, v in aql_property.items()}

    def schema_matches(self, other):
        """Test whether the schema for another ViewProperty matches this one's schema.

        Args:
            other:  ViewProperty
                The other view property to match against.

        Returns:
            True if the keys in other match this property's keys AND the types of each key's values
            match, False otherwise.
        """
        # Get sorted lists of items of each so that the comparisons are accurate
        sorted_self = sorted(self.aql_property.items())
        sorted_other = sorted(other.aql_property.items())

        # Make sure keys match
        zipped_keys = list(
            zip((e[0] for e in sorted_self), (e[0] for e in sorted_other))
        )
        keys_match = all(a == b for a, b in zipped_keys)
        log.debug4("Zipped Keys Match: %s -> %s", keys_match, str(zipped_keys))
        if not keys_match:
            return False

        # Make sure value types match, excluding null values
        zipped_val_types = list(
            zip(
                (e[1]._active_val for e in sorted_self),
                (e[1]._active_val for e in sorted_other),
            )
        )
        val_types_match = all(a == b for a, b in zipped_val_types if a and b)
        log.debug4(
            "Zipped Types Match: %s -> %s", val_types_match, str(zipped_val_types)
        )
        return val_types_match


@dataobject(package="caikit_data_model.nlp")
class View(DataObjectBase):
    name: Annotated[str, FieldNumber(1)]
    properties: Annotated[List[ViewProperty], FieldNumber(2)]

    """AQL View, with dynamic list of properties (represented as a list of map)"""

    def __init__(self, name, properties):
        """
        Args:
            name: str
                Name of view.
            properties: list(dict) or list(watson_nlp.data_model.ViewProperty)
                List of properties (key-value pairs which are dependent on the model being used).
        """
        error.type_check_all(
            "<NLP00246803E>", ViewProperty, dict, properties=properties
        )
        error.type_check("<NLP88201150E>", str, name=name)

        # Convert any raw-dict properties to ViewProperty
        def to_view_property(aql_property):
            if isinstance(aql_property, dict):
                return ViewProperty(aql_property)
            else:
                return aql_property

        properties = [to_view_property(v) for v in properties]

        # Make sure all properties have the same set of keys and the same value types
        if len(properties) > 1:
            error.value_check(
                "<NLP15126117E>",
                all(properties[0].schema_matches(p) for p in properties[1:]),
                "Rule property schemas do not all match",
            )

        self.name = name
        self.properties = properties

    def property_names(self):
        """Return list of property values matching the property name."""
        if self.properties:
            return list(self.properties[0].aql_property.keys())
        else:
            return []

    def property(self, property_name):
        """Return list of property values matching the property name."""
        error.value_check(
            "<NLP45321985E>",
            property_name in self.property_names(),
            "`property_name` `{}` not found",
            property_name,
        )

        return [
            view_property.aql_property[property_name].value
            for view_property in self.properties
        ]


@dataobject(package="caikit_data_model.nlp")
class RulesPrediction(DataObjectBase):
    views: Annotated[List[View], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    def __init__(self, views, producer_id=None):
        """
        Args:
            views: list(View)
                List of Views for the prediction model.
            producer_id: (optional) ProducerID
                producer Id of the block producing the output.
        """
        error.type_check_all("<NLP19809667E>", View, views=views)
        error.type_check(
            "<NLP52702886E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        self.views = views
        self.producer_id = producer_id

    def view_names(self):
        """List of available view names."""
        return [view.name for view in self.views]

    def filter_view(self, view_regex):
        """Filter views based on the regex.  Returns list of matching views."""
        compiled_regex = re.compile(view_regex)
        return [view for view in self.views if compiled_regex.search(view.name)]

    def view(self, view_name):
        """Returns view matching the view name."""
        if view_name not in self.view_names():
            error(
                "<NLP63208082E>",
                ValueError("`view_name` `{}` is invalid".format(view_name)),
            )

        return next((view for view in self.views if view.name == view_name), None)
