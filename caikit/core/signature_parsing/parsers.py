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
Contains functions that attempt to parse the I/O types of member methods on `caikit.core.module`s
"""
# Standard
from typing import Any, Callable, Dict, List, Optional, Type
import inspect

# First Party
import alog

# Local
from ..data_model.base import DataBase
from ..modules import ModuleBase
from . import docstrings

log = alog.use_channel("SIG-PARSING")

# Constants ##################################
KNOWN_ARG_TYPES = {
    "producer_id": "ProducerId",
}


@alog.logged_function(log.debug2)
def get_output_type_name(
    module_class: ModuleBase.__class__,
    fn_signature: inspect.Signature,
    fn: Callable,
) -> Type:
    """Get the type for a return type based on the name of the module class and
    the Caikit library naming convention.
    """
    log.debug(fn_signature)
    # Check type annotation first
    if fn_signature.return_annotation != fn_signature.empty:
        if isinstance(fn_signature.return_annotation, str):
            log.debug("Return type annotation is a string!")
            if fn_signature.return_annotation == module_class.__name__:
                log.debug(
                    "Assuming return annotation is for the module class itself: %s matches %s",
                    module_class,
                    fn_signature.return_annotation,
                )
                return module_class
        else:
            return fn_signature.return_annotation

    # Check the docstring
    type_from_docstring = docstrings.get_return_type(fn)

    if type_from_docstring:
        return type_from_docstring

    # If we get here, it means no annotation or docstring for type was provided

    # Warn unless this was a base function (e.g., don't warn if there is no train() override)
    if fn.__module__ != ModuleBase.__module__:
        log.warning(
            "Could not deduct output type from function %s for module class %s.",
            fn.__name__,
            module_class.__name__,
        )
    else:
        log.debug(
            "Could not deduct output type from function %s for module class %s using %s.",
            fn.__name__,
            module_class.__name__,
            fn.__qualname__,
        )


def get_argument_types(module_method: Callable) -> Dict[str, Type]:
    """Get the python types for each parameter to this method, returned in a dict.
    This does more than simply looking at inspect.Signature, see _get_argument_type

    Args:
        module_method (Callable): A pointer to a method

    Returns:
        Dict[str, Type]: A dictionary of parameter name to parameter type
    """
    method_signature = inspect.signature(module_method)
    return {
        name: _get_argument_type(param, module_method)
        for name, param in method_signature.parameters.items()
        if name not in ["self", "args", "kwargs", "_", "__"]
    }


def get_args_with_defaults(module_method: Callable) -> Dict[str, Any]:
    """Get the the mapping of all argument names that have defaults to their
    default values.

    Args:
        module_method (Callable): A pointer to a method

    Returns:
        Dict[str: Any]: A set of all parameter names which have a default value.
            Empty if none have defaults or no parameters exist.
    """
    method_signature = inspect.signature(module_method)
    return {
        param.name: param.default
        for param in method_signature.parameters.values()
        if param.default != inspect.Parameter.empty
    }


# pylint: disable=too-many-return-statements
@alog.logged_function(log.debug2)
def _get_argument_type(
    arg: inspect.Parameter,
    module_method: Callable,
) -> Type:
    """Get the python type for a named argument to a Module's given method. This
    is where the heuristics for determining types are implemented:

    * Look for a known type mapping based on the name of the argument
    * Look for python type annotations
    * Look for a default value and check its type
    * Parse the docstring
    * Look for a data model object whose name matches the argument name
    """
    # TODO: KNOWN_ARG_TYPES should be configurable

    # Use known arg types first
    # This avoids cases where docstrings are very obviously flubbed, such as
    #   `raw_document` being annotated as a `str` only in caikit.interfaces.nlp
    dm_type_from_known_arg_types = _get_dm_type_from_name(KNOWN_ARG_TYPES.get(arg.name))
    if dm_type_from_known_arg_types:
        # Not checking if this is optional: These known types should never be optional (maybe...?)
        # This could totally be incorrect!
        log.info(
            "Using well known type %s for parameter name %s",
            dm_type_from_known_arg_types,
            arg.name,
        )
        return dm_type_from_known_arg_types

    # Check docstring for optional arg
    optional_arg = docstrings.is_optional(module_method, arg.name)

    # Look for a type annotation
    if arg.annotation != inspect.Parameter.empty:
        log.debug("Found annotation for %s", arg.name)
        if optional_arg:
            return Optional[arg.annotation]
        return arg.annotation

    log.debug("No annotation found for %s", arg.name)

    # Check for a default argument and return its type
    default_type = _get_default_type(arg)
    if default_type:
        return default_type

    # Parse docstring
    type_from_docstring = docstrings.get_arg_type(module_method, arg.name)
    if type_from_docstring:
        if optional_arg:
            return Optional[type_from_docstring]
        return type_from_docstring

    # Look for a data model object whose name matches the argument name and fall
    # back to the KNOWN_ARG_TYPES dict
    candidate_name = _snake_to_camel(arg.name)
    type_from_candidate = _get_dm_type_from_name(candidate_name)
    if optional_arg:
        return Optional[type_from_candidate]
    return type_from_candidate


def _snake_to_camel(string: str) -> str:
    """Simple snake -> camel conversion"""
    return "".join([part[0].upper() + part[1:] for part in string.split("_")])


def _get_dm_type_from_name(data_model_class_name: Optional[str]) -> Type:
    """Given a data model class name, look up the data model class itself"""
    if data_model_class_name is None:
        return None
    try:
        return DataBase.get_class_for_name(data_model_class_name)
    except ValueError:
        return None


def _get_default_type(arg: inspect.Parameter) -> Optional[Type]:
    """
    Tries to infer a type from the default value of the argument
    Args:
        arg: (inspect.Parameter) The inspected argument
    Returns: (Optional[Type]) The type of the argument,: or None if no default value is
        present
    """
    if arg.default != inspect.Parameter.empty and arg.default is not None:
        log.debug3("Found default with type %s", type(arg.default))

        # If the default is a list or tuple, we'll create the corresponding
        # typing type using List. Note that even when the default is a tuple,
        # we use List because we assume that tuple is used to avoid the problem
        # of mutable defaults.
        if isinstance(arg.default, (tuple, list)):
            value_types = {type(val) for val in arg.default}
            if value_types:
                if len(value_types) > 1:
                    log.warning(
                        "Found argument [%s] with iterable default [%s] and multiple types",
                        arg.name,
                        arg.default,
                    )
                return List[list(value_types)[0]]
        else:
            return type(arg.default)
    # Return None if no default argument was given
    return None
