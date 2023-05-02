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

# Standard
from typing import Callable, List, Set, Type, Union

# First Party
import alog

# Local
from caikit.core.data_model import DataStream
from caikit.core.data_model.base import DataBase
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("TASK_BASE")
error = error_handler.get(log)

ProtoableInputTypes = Type[Union[int, float, str, bytes, bool, DataBase]]


class DomainBase:
    @classmethod
    def validate_task_inputs(cls) -> bool:
        pass

    @classmethod
    def get_input_type_set(cls) -> Set[ProtoableInputTypes]:
        raise NotImplementedError("This is implemented by the @domain decorator!")


def domain(
    input_types: Set[ProtoableInputTypes],
) -> Callable[[Type[DomainBase]], Type[DomainBase]]:
    """The decorator for AI Domains"""

    def type_check(x: type) -> bool:
        return (
            x == int
            or x == float
            or x == str
            or x == bytes
            or x == bool
            or (isinstance(x, type) and issubclass(x, DataBase))
        )

    for input_type in input_types:
        error.value_check(
            "<COR98288712E>",
            type_check(input_type),
            input_type,
            msg="Domain inputs must be python primitive types or data model types. Got {}",
        )

    def get_input_type_set(_) -> Set[ProtoableInputTypes]:
        return input_types

    def decorator(cls: Type[DomainBase]) -> Type[DomainBase]:
        error.value_check(
            "<COR98211745E>",
            isinstance(cls, type) and issubclass(cls, DomainBase),
            cls,
            msg="@domain class must extend DomainBase",
        )
        setattr(cls, "get_input_type_set", classmethod(get_input_type_set))
        return cls

    return decorator
