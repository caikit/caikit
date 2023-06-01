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
This file contains our logic about what constitutes a "primitive" for RPC generation purposes
"""

# Standard
from typing import Dict, List, Type, Union, get_args, get_origin
import inspect
import sys
import typing

# First Party
import alog

# Local
from .type_helpers import PROTO_TYPE_MAP
from caikit.core.data_model.base import DataBase

log = alog.use_channel("MODULE_PRIMS")


def to_primitive_signature(
    signature: Dict[str, Type]
) -> Dict[str, Type]:
    """Returns dictionary of primitive types only
    If there is a Union, pick the primitive type

    Args:
        signature: Dict[str, Type]
            module signature of parameters and types
    """
    primitives = {}
    log.debug("Building primitive signature for %s", signature)
    for arg, arg_type in signature.items():
        primitive_arg_type = handle_primitives_in_union(
            arg_type
        )
        if primitive_arg_type:
            primitives[arg] = primitive_arg_type

    return primitives


def handle_primitives_in_union(arg_type: Type) -> Type:
    """Handles various primitive arg types from a Union:
    Union[supported_type, unsupported_type] -> returns only the supported_type
    Union[supported_type1, supported_type2] -> returns the union which creates the oneof
    Union[supported_type1, supported_type2, unsupported_type] ->
        returns the first primitive object in the Union
    """
    if _is_primitive_type(arg_type):
        if typing.get_origin(arg_type) == Union:
            union_primitives = [
                union_val
                for union_val in typing.get_args(arg_type)
                if _is_primitive_type(union_val)
            ]
            # if all are primitives, return the union (which will create a oneof)
            if len(union_primitives) == len(typing.get_args(arg_type)):
                return arg_type
            # if there's only 1 primitive found, return that
            if len(union_primitives) == 1:
                return union_primitives[0]
            # otherwise, try to get the primitive dm objects in the Union
            dm_types = [
                arg
                for arg in union_primitives
                if inspect.isclass(arg) and issubclass(arg, DataBase)
            ]
            # if there are multiple, pick the first one
            if len(dm_types) > 0:
                log.debug2(
                    "Picking first data model type %s in union primitives %s",
                    dm_types,
                    union_primitives,
                )
                return dm_types[0]
            log.debug(
                "Just picking first primitive type %s in union",
                union_primitives[0],
            )
            return union_primitives[0]
        return arg_type
    log.debug("Skipping non-primitive argument type [%s]", arg_type)


def extract_data_model_type_from_union(arg_type: Type) -> Type:
    """Helper function that determines the right data model type to use from a Union"""

    # Decompose this type using typing to determine if it's a useful typing hint
    typing_origin = get_origin(arg_type)
    typing_args = get_args(arg_type)

    # If this is a data model type, no need to do anything
    if isinstance(arg_type, type) and issubclass(arg_type, DataBase):
        return arg_type

    # Handle Unions by looking for a data model object in the union
    if typing_origin is Union:
        dm_types = [
            arg
            for arg in typing_args
            if inspect.isclass(arg) and issubclass(arg, DataBase)
        ]
        if dm_types:
            log.debug2(
                "Found data model types in Union: [%s], taking first one", dm_types
            )
            return extract_data_model_type_from_union(dm_types[0])

    # if it's anything else we just return as is
    # we don't actually want to throw errors from service generation
    log.warning("Return type [%s] not a DM type, returning as is", arg_type)
    return arg_type


def _is_primitive_type(arg_type: Type) -> bool:
    """
    Returns True is arg_type is in PROTO_TYPE_MAP(float, int, bool, str, bytes)
    Or if it's an imported Caikit data model class.
    Or if it's a Union of at least one of those.
    Or if it's a List of one of those.
    False otherwise"""
    primitive_set = list(PROTO_TYPE_MAP.keys())

    if arg_type in primitive_set:
        return True
    if isinstance(arg_type, type) and issubclass(arg_type, DataBase):
        return True

    if typing.get_origin(arg_type) == list:
        log.debug2("Arg is List")
        # check that list is not nested
        if len(typing.get_args(arg_type)) == 1:
            return typing.get_args(arg_type)[0] in primitive_set
        # TODO: that check is always true
        log.debug2("Arg is a list more than one type")

    if typing.get_origin(arg_type) == Union:
        log.debug2("Arg is Union")
        # pylint: disable=use-a-generator
        return any(
            [
                _is_primitive_type(arg)
                for arg in typing.get_args(arg_type)
            ]
        )

    log.debug2("Arg is not primitive, arg_type: %s", arg_type)
    return False


def _get_library_dm_primitives(primitive_data_model_types) -> List[Type[DataBase]]:
    """For a given caikit.* library name, determine the set of data model "primitives"."""

    lib_dm_primitives = []
    for primitive_name in primitive_data_model_types:
        parts = primitive_name.split(".")
        current = sys.modules.get(parts[0])
        for part in parts[1:]:
            current = getattr(current, part, None)
        if current is not None:
            lib_dm_primitives.append(current)
    log.debug3("DM Primitives: %s", lib_dm_primitives)
    return lib_dm_primitives


def _is_optional_type(arg_type: Type):
    if typing.get_origin(arg_type) == Union and type(None) in typing.get_args(arg_type):
        return True
    return False
