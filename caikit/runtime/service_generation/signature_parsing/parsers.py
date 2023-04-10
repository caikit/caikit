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
from types import FunctionType
from typing import List, Optional, Type
import inspect

# First Party
import alog

# Local
from . import docstrings
from caikit.core.data_model.base import DataBase
from caikit.core.module import ModuleBase

log = alog.use_channel("SIG-PARSING")

# Constants ##################################
KNOWN_ARG_TYPES = {
    "syntax_doc": "SyntaxPrediction",
    "producer_id": "ProducerId",
    "raw_document": "RawDocument",
}

KNOWN_OUTPUT_TYPES = {}


@alog.logged_function(log.debug2)
def get_output_type_name(
    module_class: ModuleBase.__class__,
    fn_signature: inspect.Signature,
    fn: FunctionType,
) -> Type:
    """Get the type for a return type based on the name of the module class and
    the Caikit library naming convention.
    """
    extra_candidate_names = []
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

            log.debug(
                "Adding %s to list of candidate type names to search for concrete types:",
                fn_signature.return_annotation,
            )
            extra_candidate_names.append(fn_signature.return_annotation)
        else:
            return fn_signature.return_annotation

    # Check the docstring
    # TODO: NOOOOOOOO need the real function
    type_from_docstring = docstrings.get_return_type(module_class, fn)
    if type_from_docstring:
        return type_from_docstring

    # Check based on naming conventions and then known output types
    module_parts = module_class.__module__.split(".")
    log.debug3("Parent module parts for %s: %s", module_class.__name__, module_parts)

    # TODO: Why the F doesn't this work with docstrings?
    # please test lol

    class_name = _snake_to_camel(module_parts[2]) + "Prediction"
    return _get_dm_type_from_name(class_name) or _get_dm_type_from_name(
        KNOWN_OUTPUT_TYPES.get(class_name)
    )


# pylint: disable=too-many-return-statements
@alog.logged_function(log.debug2)
def get_argument_type(
    arg: inspect.Parameter,
    module_class: ModuleBase.__class__,
    module_method: FunctionType,
) -> Type:
    """Get the python type for a named argument to a Module's given method. This
    is where the heuristics for determining types are implemented:

    * Look for a known type mapping based on the name of the argument
    * Look for python type annotations
    * Look for a default value and check its type
    * Parse the docstring
    * Look for a data model object whose name matches the argument name
    """
    # Check docstring for optional arg
    optional_arg = docstrings.is_optional(module_method, arg.name)

    # Use known arg types first
    # This avoids cases where docstrings are very obviously flubbed, such as
    #   `raw_document` being annotated as a `str` only in caikit.interfaces.nlp
    dm_type_from_known_arg_types = _get_dm_type_from_name(KNOWN_ARG_TYPES.get(arg.name))
    if dm_type_from_known_arg_types:
        # Not checking if this is optional: These known types should never be optional (maybe...?)
        # This could totally be incorrect!
        return dm_type_from_known_arg_types

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

    type_from_docstring = docstrings.get_arg_type(module_class, module_method, arg.name)
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

    Returns: (Optional[Type]) The type of the argument,
        or None if no default value is present
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
