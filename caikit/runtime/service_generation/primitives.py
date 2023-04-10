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
from typing import Dict, List, Type, Union
import inspect
import sys
import typing

# First Party
import alog

# Local
from .signature_parsing.module_signature import CaikitCoreModuleMethodSignature
from .type_helpers import PROTO_TYPE_MAP
from caikit.core.data_model.base import DataBase

log = alog.use_channel("MODULE_PRIMS")


def to_primitive_signature(
    signature: Dict[str, Type], primitive_data_model_types: List[str]
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
        if _is_primitive_type(arg_type, primitive_data_model_types):
            if typing.get_origin(arg_type) == Union:
                union_primitives = [
                    union_val
                    for union_val in typing.get_args(arg_type)
                    if _is_primitive_type(union_val, primitive_data_model_types)
                ]
                if len(union_primitives) == 1:
                    primitives[arg] = union_primitives[0]
                else:
                    dm_types = [
                        arg
                        for arg in union_primitives
                        if inspect.isclass(arg) and issubclass(arg, DataBase)
                    ]
                    if len(dm_types) > 0:
                        log.debug2(
                            "Picking first data model type %s in union primitives %s",
                            dm_types,
                            union_primitives,
                        )
                        primitives[arg] = dm_types[0]
                    else:
                        log.debug(
                            "Just picking first primitive type %s in union",
                            union_primitives[0],
                        )
                        primitives[arg] = union_primitives[0]

            else:
                primitives[arg] = arg_type
        else:
            log.debug("Skipping non-primitive argument [%s], type [%s]", arg, arg_type)

    return primitives


def is_primitive_method(
    method: CaikitCoreModuleMethodSignature, primitive_data_model_types: List[str]
) -> bool:
    """Determine if the arguments to the module's run function meet the criteria
    for being a "primitive" interface this means that all **non-optional** arguments
    types must be either (a) a primitive data model type for the given library or
    (b) a language-primitive type.

    Args:
        method (CaikitCoreModuleMethodSignature): The method signature of the "primitive"
            data model types for each library
    """

    # pylint: disable=use-a-generator
    return all(
        [
            (
                _is_primitive_type(arg_type, primitive_data_model_types)
                or _is_optional_type(arg_type)
            )
            for arg_type in method.parameters.values()
        ]
    )


def _is_primitive_type(arg_type: Type, primitive_data_model_types: List[str]) -> bool:
    """
    Returns True is arg_type is in PROTO_TYPE_MAP(float, int, bool, str, bytes)
    Or if it's an imported Caikit library primitive.
    Or if it's a Union of at least one primitive.
    Or if it's a List of primitives.
    False otherwise"""
    lib_dm_primitives = _get_library_dm_primitives(primitive_data_model_types)
    primitive_set = list(PROTO_TYPE_MAP.keys()) + lib_dm_primitives
    if arg_type in primitive_set:
        return True
    if typing.get_origin(arg_type) == list:
        log.debug2("Arg is List")
        # check that list is not nested
        if len(typing.get_args(arg_type)) == 1:
            return typing.get_args(arg_type)[0] in primitive_set
        log.debug2("Arg is a list more than one type")

    if typing.get_origin(arg_type) == Union:
        log.debug2("Arg is Union")
        # pylint: disable=use-a-generator
        return any(
            [
                _is_primitive_type(arg, primitive_data_model_types)
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
