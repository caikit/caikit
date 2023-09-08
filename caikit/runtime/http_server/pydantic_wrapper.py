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
This module holds the Pydantic wrapping required by the REST server,
capable of converting to and from Pydantic models to our DataObjects.
"""
# Standard
from typing import Dict, List, Type, Union, get_args, get_type_hints
import base64
import enum

# Third Party
from pydantic.functional_validators import BeforeValidator
import numpy as np
import pydantic

# First Party
from py_to_proto.dataclass_to_proto import (  # Imported here for 3.8 compat
    Annotated,
    get_origin,
)

# Local
from caikit.core.data_model.base import DataBase
from caikit.interfaces.common.data_model.primitive_sequences import (
    BoolSequence,
    FloatSequence,
    IntSequence,
    StrSequence,
)

# PYDANTIC_TO_DM_MAPPING is essentially a 2-way map of DMs <-> Pydantic models, you give it a
# pydantic model, it gives you back a DM class, you give it a
# DM class, you get back a pydantic model.
PYDANTIC_TO_DM_MAPPING = {
    # Map primitive sequences to lists
    StrSequence: List[str],
    IntSequence: List[int],
    FloatSequence: List[float],
    BoolSequence: List[bool],
}


# Base class for pydantic models
# We want to set the config to forbid extra attributes
# while instantiating any pydantic models
# This is done to make sure any oneofs can be
# correctly infered by pydantic
class ParentPydanticBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


def pydantic_to_dataobject(pydantic_model: pydantic.BaseModel) -> DataBase:
    """Convert pydantic objects to our DM objects"""
    dm_class_to_build = PYDANTIC_TO_DM_MAPPING.get(type(pydantic_model))
    dm_kwargs = {}

    for field_name, field_value in pydantic_model:
        # field could be a DM:
        # pylint: disable=unidiomatic-typecheck
        if type(field_value) in PYDANTIC_TO_DM_MAPPING:
            dm_kwargs[field_name] = pydantic_to_dataobject(field_value)
        elif isinstance(field_value, list):
            if all(type(val) in PYDANTIC_TO_DM_MAPPING for val in field_value):
                dm_kwargs[field_name] = [
                    pydantic_to_dataobject(val) for val in field_value
                ]
            else:
                dm_kwargs[field_name] = field_value
        else:
            dm_kwargs[field_name] = field_value

    return dm_class_to_build(**dm_kwargs)


def dataobject_to_pydantic(dm_class: Type[DataBase]) -> Type[pydantic.BaseModel]:
    """Make a pydantic model based on the given proto message by using the data
    model class annotations to mirror as a pydantic model
    """
    # define a local namespace for type hints to get type information from.
    # This is needed for pydantic to have a handle on JsonDict and JsonDictValue while
    # creating its base model
    localns = {"JsonDict": dict, "JsonDictValue": dict}

    if dm_class in PYDANTIC_TO_DM_MAPPING:
        return PYDANTIC_TO_DM_MAPPING[dm_class]

    annotations = {
        field_name: _get_pydantic_type(field_type)
        for field_name, field_type in get_type_hints(dm_class, localns=localns).items()
    }
    pydantic_model = type(ParentPydanticBaseModel)(
        dm_class.get_proto_class().DESCRIPTOR.full_name,
        (ParentPydanticBaseModel,),
        {
            "__annotations__": annotations,
            **{
                name: None
                for name, _ in get_type_hints(
                    dm_class,
                    localns=localns,
                ).items()
            },
        },
    )
    PYDANTIC_TO_DM_MAPPING[dm_class] = pydantic_model
    # also store the reverse mapping for easy retrieval
    # should be fine since we only check for dm_class in this dict
    PYDANTIC_TO_DM_MAPPING[pydantic_model] = dm_class
    return pydantic_model


# pylint: disable=too-many-return-statements
def _get_pydantic_type(field_type: type) -> type:
    """Recursive helper to get a valid pydantic type for every field type"""
    # pylint: disable=too-many-return-statements

    # Leaves: we should have primitive types and enums
    if np.issubclass_(field_type, np.integer):
        return int
    if np.issubclass_(field_type, np.floating):
        return float
    if field_type == bytes:
        return Annotated[bytes, BeforeValidator(_from_base64)]
    if field_type in (int, float, bool, str, dict, type(None)):
        return field_type
    if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return field_type

    # These can be nested within other data models
    if (
        isinstance(field_type, type)
        and issubclass(field_type, DataBase)
        and not issubclass(field_type, pydantic.BaseModel)
    ):
        # NB: for data models we're calling the data model conversion fn
        return dataobject_to_pydantic(field_type)

    # And then all of these types can be nested in other type annotations
    if get_origin(field_type) is Annotated:
        return _get_pydantic_type(get_args(field_type)[0])
    if get_origin(field_type) is Union:
        return Union[  # type: ignore
            tuple((_get_pydantic_type(arg_type) for arg_type in get_args(field_type)))
        ]
    if get_origin(field_type) is list:
        return List[_get_pydantic_type(get_args(field_type)[0])]

    if get_origin(field_type) is dict:
        return Dict[
            _get_pydantic_type(get_args(field_type)[0]),
            _get_pydantic_type(get_args(field_type)[1]),
        ]

    raise TypeError(f"Cannot get pydantic type for type [{field_type}]")


def _from_base64(data: Union[bytes, str]) -> bytes:
    if isinstance(data, str):
        return base64.b64decode(data.encode("utf-8"))
    return data
