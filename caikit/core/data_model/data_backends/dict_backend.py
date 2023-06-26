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

"""The dict-based backend implementation
"""

# Standard
from typing import Any, Iterable, Type

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from ..base import DataBase
from .base import DataModelBackendBase

log = alog.use_channel("DATAB")
error = error_handler.get(log)


# DictBackend ##################################################################


class DictBackend(DataModelBackendBase):
    """Data model backend for a raw dict"""

    def __init__(self, data_dict: dict):
        """Construct with the dict"""
        error.type_check("<COR85037210E>", dict, data_dict=data_dict)
        self._data_dict = data_dict

    def get_attribute(self, data_model_class: Type[DataBase], name: str) -> Any:
        """Fetch the attribute out of the internal dict and validate it against
        the target data model class. If the target attribute is a nested data
        model type, wrap the corresponding nested dict in an instance of this
        same backend.

        Args:
            data_model_class (Type[DataBase]): The frontend data model class
                that is accessing this attribute
            name (str): The name of the attribute to access

        Returns:
            value:  Any
                The extracted attribute value
        """

        # NOTE: We do not type-check the args here for efficiency. This method
        #   should only be called by the DataBase class, so we can assume it's
        #   being used correctly.

        # Make sure the name is a valid field on the given class
        if (
            name not in data_model_class.fields
            and name not in data_model_class._fields_oneofs_map
        ):
            error(
                "<COR85037211E>",
                AttributeError(
                    f"No such attribute [{name}] on [{data_model_class.__name__}]"
                ),
            )

        # Get the value from the internal dict
        raw_value = self._data_dict.get(name)

        # If the target attribute is itself a message, make sure the value is a
        # dict, then wrap it in the corresponding data model object with the a
        # new backend instance.
        if name in data_model_class._fields_message and raw_value is not None:
            error.type_check("<COR85037212E>", dict, **{name: raw_value})
            proto_class = data_model_class.get_proto_class()
            field_dm_class = DataBase.get_class_for_proto(
                proto_class.DESCRIPTOR.fields_by_name[name].message_type
            )
            return field_dm_class.from_backend(self.__class__(raw_value))

        # If the target attribute is a repeated message, convert it to a list of
        # nested messages with dict backends
        #
        # TODO: It may be better to do this lazily, but that would come at the
        #   expense of being able to re-iterate or randomly-index into the
        #   object. We could consider writing a lazily constructed list to avoid
        #   costruction of the messages that aren't used.
        if name in data_model_class._fields_message_repeated and raw_value is not None:
            error.type_check("<COR85037213E>", Iterable, **{name: raw_value})
            field_dm_class = data_model_class.get_field_message_type(name)
            return [
                field_dm_class.from_backend(self.__class__(entry))
                for entry in raw_value
            ]

        # If the target attribure is a oneof name, look for the oneof name in the data dict first.
        # If it exists, infer the right field name based on the type of the value and return the
        # field name and the value. If oneof name isn't in the data dict, check for field name
        # instead and return the right value
        if name in data_model_class._fields_oneofs_map:
            if name in self._data_dict:
                val = self._data_dict[name]
                which_oneof = data_model_class._infer_which_oneof(name, val)
                return data_model_class.OneofFieldVal(val=val, which_oneof=which_oneof)

            oneof_fields = data_model_class._fields_oneofs_map[name]
            for field in oneof_fields:
                if field in self._data_dict:
                    return data_model_class.OneofFieldVal(
                        val=self._data_dict[field], which_oneof=field
                    )

        return raw_value
