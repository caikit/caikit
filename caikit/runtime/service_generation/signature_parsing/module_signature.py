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
This package exposes useful metadata about a method on a caikit.core module, including method
name, return type and input parameters.

This encapsulates a lot of custom logic used to parse input/output parameters, especially when it
comes to the caikit core common data model.
"""

# Standard
from typing import Dict, Optional, Type
import inspect

# First Party
import alog

# Local
from . import parsers
from caikit.core import ModuleBase

log = alog.use_channel("MODULE-SIGNR")


# Maybe TODO: support for `INPUT_TYPES` and `OUTPUT_TYPE` from module annotations was removed
class CaikitCoreModuleMethodSignature:
    """Metadata about a method on a caikit core module

    Determines the argument types and return type for a function (run, train, etc.)
    of the given caikit core module in any way possible!

    This is the most "heuristic" part of this process. Currently, there is no
    well-defined mechanism for defining the types of a module's run signature in
    the caikit.core API, so it is enforced by convention only. As such, this
    package needs to reverse engineer those conventions! It does so by walking
    through a list of candidate ways to determine the signature in order least-
    hacky to most-hacky:

    1. Look for a known type mapping based on the name of the argument
    2. Look for python type annotations
    3. Look for a default value and check its type
    4. Parse the docstring

    Return types can also be inferred as "{module_type}Prediction" if the type cannot be
    determined any other way.
    """

    def __init__(self, caikit_core_module: Type[ModuleBase], method_name: str):
        self._module = caikit_core_module
        self._method_name = method_name

        try:
            self._method_pointer = getattr(self._module, self._method_name)
            method_signature = inspect.signature(self._method_pointer)
            self._return_type = parsers.get_output_type_name(
                self._module, method_signature, self._method_pointer
            )

            self._parameters = {
                name: parsers.get_argument_type(
                    param, self._module, self._method_pointer
                )
                for name, param in method_signature.parameters.items()
                if name not in ["self", "args", "kwargs", "_"]
            }
        except AttributeError:
            log.warning(
                "Could not find method [%s] in this module",
                self.method_name,
            )
            self._return_type = None
            self._parameters = None

    @property
    def module(self) -> Type[ModuleBase]:
        """The concrete caikit.core.ModuleBase type"""
        return self._module

    @property
    def method_name(self) -> str:
        """The name of the method on this module, e.g. 'run' or 'train'"""
        return self._method_name

    @property
    def return_type(self) -> Optional[Type]:
        """The return type annotation of the method, or None if the method does not exist"""
        return self._return_type

    @property
    def parameters(self) -> Optional[Dict[str, Type]]:
        """A dictionary of the parameter names to their types, or None if the method does not
        exist"""
        return self._parameters


class CustomSignature(CaikitCoreModuleMethodSignature):
    """(TBD on new class)? Need something to hold an intentionally mutated representation of a
    method signature. This represents the extra indirection that lives in the runtime, between the
    service API and the actual method. For example: .train functions return a fully constructed
    module, but the runtime will invoke .train asynchronously and instead return some handle that
    can be used to check training status."""

    def __init__(
        self,
        original_signature: CaikitCoreModuleMethodSignature,
        parameters: Dict[str, Type],
        return_type: Optional[Type],
    ):
        super().__init__(original_signature.module, original_signature.method_name)
        self._module = original_signature.module
        self._method_name = original_signature.method_name
        self._method_pointer = original_signature._method_pointer

        self._return_type = return_type
        self._parameters = parameters
