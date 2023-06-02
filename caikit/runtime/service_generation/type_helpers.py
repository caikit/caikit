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
Some type conversion helpers for going between python and protocol buffer interfaces
"""
# Standard
from inspect import isclass
from typing import Optional, Type, Union, get_args, get_origin

# Local
from caikit.core import ModuleBase
from caikit.core.data_model import DataBase, DataStream


def has_data_stream(arg_type: Type) -> bool:
    """Recursive check for a DataStream container in a type annotation"""
    if _is_data_stream(arg_type):
        return True

    typing_args = get_args(arg_type)
    if len(typing_args) > 0:
        for typ in typing_args:
            if has_data_stream(typ):
                return True

    return False


def get_data_stream_type(arg_type: Type) -> Optional[Type]:
    """Extracts the `DataStream` annotation from inside a nested type annotation, if it exists"""
    if _is_data_stream(arg_type):
        return arg_type
    typing_args = get_args(arg_type)
    for typ in typing_args:
        if has_data_stream(typ):
            return get_data_stream_type(typ)

    return None


def is_model_type(arg_type: Type) -> bool:
    if isclass(arg_type) and issubclass(arg_type, ModuleBase):
        return True
    typing_origin = get_origin(arg_type)
    typing_args = get_args(arg_type)

    if typing_origin is Union:
        for typ in typing_args:
            if isclass(typ) and issubclass(typ, ModuleBase):
                return True
    return False


def is_data_model_type(arg_type: Type) -> bool:
    return isclass(arg_type) and issubclass(arg_type, DataBase)


def _is_data_stream(arg_type: Type) -> bool:
    return arg_type == DataStream or get_origin(arg_type) == DataStream
