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
"""This package handles all the gorp of finding types from docstrings given our custom
conventions"""

# Standard
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import builtins
import sys

# Third Party
from docstring_parser import ParseError
import docstring_parser

# First Party
import alog

# Local
from caikit.core.data_model.base import DataBase
import caikit.core

log = alog.use_channel("DOCSTRINGS")


def get_return_type(fn: Callable) -> Optional[Type]:
    """
    Grabs the return type off the docstring, if possible
    Args:
        fn: The function to get the return value of
            e.g. my_caikit_library.modules.classification.Transformer.run

    Returns:
        The return type of `fn`, if it can be parsed from the docstring. Otherwise, None
    """
    try:
        docstring = docstring_parser.parse(fn.__doc__)
    except docstring_parser.ParseError as e:
        log.warning("Could not parse docstring: %s fn.__doc__ ", fn.__doc__, exc_info=e)
        return None

    type_names, desc_names = _get_candidate_type_names_from_docstring(docstring.returns)

    return_type = _get_docstring_type(type_names)
    if return_type:
        return return_type

    return _get_docstring_type(desc_names)


def is_optional(fn: Callable, arg_name: str) -> bool:
    """
    Checks if the `arg_name` param from `fn`s docstring is optional
    by checking if param description starts with "an optional"
    or "optional".

    Args:
        fn: The function to get the type of a parameter from
            e.g.  my_caikit_library.modules.classification.Transformer.run
        arg_name: The name of the parameter that we should try to get the type of
            e.g. "raw_document"
    """
    try:
        docstring = docstring_parser.parse(fn.__doc__)
    except ParseError as parse_error:
        log.warning(
            "Failed to parse docstring for %s when looking for optional flag on parameter %s",
            fn,
            arg_name,
            exc_info=parse_error,
        )
        return False

    ds_param = [param for param in docstring.params if param.arg_name == arg_name]
    if ds_param:
        if len(ds_param) > 1:
            log.warning("Docstring has multiple args with the same name! %s", arg_name)
        ds_param = ds_param[0]

        if ds_param.description is not None:
            for description_line in ds_param.description.split("\n"):
                if description_line.lower().startswith(
                    "optional"
                ) or description_line.lower().startswith("an optional"):
                    log.debug2("Optional parameter found: %s", ds_param)
                    return True

        return False
    return False


def get_arg_type(fn: Callable, arg_name: str) -> Optional[Type]:
    """
    Grabs the type of the `arg_name` param from `fn`s docstring, if possible
    Args:
        fn: The function to get the type of a parameter from
            e.g. my_caikit_library.modules.classification.Transformer.run
        arg_name: The name of the parameter that we should try to get the type of
            e.g. "raw_document"
    Returns:
        The return type of `fn`, if it can be parsed from the docstring. Otherwise, None
    """
    try:
        docstring = docstring_parser.parse(fn.__doc__)
    except ParseError as parse_error:
        log.warning(
            "Failed to parse docstring for %s when looking for type on parameter %s",
            fn,
            arg_name,
            exc_info=parse_error,
        )
        return None

    ds_param = [param for param in docstring.params if param.arg_name == arg_name]
    if ds_param:
        if len(ds_param) > 1:
            log.warning("Docstring has multiple args with the same name! %s", arg_name)
        ds_param = ds_param[0]
        type_names, desc_names = _get_candidate_type_names_from_docstring(ds_param)
        docstring_type = _get_docstring_type(type_names)
        if not docstring_type:
            docstring_type = _get_docstring_type(desc_names)
        if docstring_type is not None:
            log.debug2("Found type from docstring for %s: %s", arg_name, docstring_type)
            return docstring_type
    else:
        log.warning("Found no parameter named %s:%s", arg_name, fn.__name__)
    return None


def _get_candidate_type_names_from_docstring(
    param: Optional[docstring_parser.common.DocstringParam],
) -> Tuple[List[str], List[str]]:
    if param is None:
        return [], []

    # Check the official 'type_name'
    candidate_type_names = []
    candidate_types_from_description = []
    if param.type_name is not None:
        candidate_type_names.append(param.type_name)

    # If not in type_name, try parsing our convention from the
    # description
    if param.description is not None:
        candidate_types_from_description.extend(
            [
                val
                for val in param.description.split("\n")[0].split()
                if val not in ["or", "|"]
            ]
        )
    log.debug3(
        "Candidate type names: %s, %s",
        candidate_type_names,
        candidate_types_from_description,
    )

    return candidate_type_names, candidate_types_from_description


def _get_docstring_type(
    candidate_type_names: List[str],
) -> Optional[Type]:
    """Given a parsed docstring parameter, look in all of the possible places
    for the actual type
    """

    log.debug2(
        "Candidate type names for docstring parsing are: %s", candidate_type_names
    )

    # If we can't find the name in either place, we're done
    if not candidate_type_names:
        log.debug2("Could not find type name from docstring")
        return None

    # Check all candidate type names
    valid_candidates = []
    for type_name in candidate_type_names:
        # Check for builtin types
        builtin_type = getattr(builtins, type_name, None)
        if builtin_type is not None:
            valid_candidates.append(builtin_type)
            log.debug2(f"Found valid candidate type: {builtin_type}")
            continue

        # Try to find things like "list(str)"
        # List[str]???
        nested_type = _extract_nested_type(type_name)
        if nested_type is not None:
            valid_candidates.append(nested_type)
            log.debug2(f"Found valid nested type: {nested_type}")
            continue

        # Try to spelunk down `sys.modules` for the type. This should work if it is fully qualified
        candidate_type = _extract_type_from_pymodule(sys.modules, type_name)
        if candidate_type is not None:
            valid_candidates.append(candidate_type)
            log.debug2(f"Found valid candidate type on sys.modules: {candidate_type}")
            continue

        # If the type was not fully qualified (like a `ProducerId`), look in a couple well known
        # places - the caikit core data model itself
        candidate_type = _extract_type_from_pymodule(caikit.core.data_model, type_name)
        if candidate_type is not None:
            valid_candidates.append(candidate_type)
            log.debug2(
                # pylint: disable=line-too-long
                f"Found valid candidate type on caikit.core.data_model: {candidate_type}"
            )
            continue
        # ...And the data model within the interfaces, including those defined in library
        try:
            candidate_type = DataBase.get_class_for_name(type_name)
        except ValueError:
            log.debug2(f"Data model match failed on {candidate_type}, continuing")
            continue
        if candidate_type is not None:
            valid_candidates.append(candidate_type)
            log.debug2(f"Found valid data model candidate type: {candidate_type}")
            continue

    log.debug3("valid candidates %s", valid_candidates)

    # If valid candidates were found, return either the single or a Union
    if valid_candidates:
        if len(valid_candidates) == 1:
            return valid_candidates[0]

        # pylint: disable=unnecessary-dunder-call
        return Union.__getitem__(tuple(valid_candidates))
    log.debug2(
        "Unable to pull type name [%s]",
        candidate_type_names,
    )


def _extract_nested_type(type_name: str) -> Optional[Type]:
    type_name = type_name.replace("[", "(").replace("]", ")")
    is_a_list = type_name.lower().startswith("list")
    if is_a_list:
        start_child_type_name = type_name.find("(") + 1
        end_child_type_name = type_name.rfind(")")
        child_type_name = type_name[start_child_type_name:end_child_type_name]

        child_type = _get_docstring_type(candidate_type_names=[child_type_name])
        if child_type:
            return List[child_type]


def _extract_type_from_pymodule(
    py_module: Union[Type, Dict], type_name: str
) -> Optional[Type]:
    """This walks down a type hierarchy to try to find the concrete type given an input string name

    Args:
        py_module (Type | Dict): A python module, or dictionary of modules, to
            start walking to find "type_name"
        type_name (str): The name of the type that we're trying to find. e.g.
            "caikit.core.data_model.ProducerId"

    Returns:
        Optional[Type]: The type of "type_name", or None if it cannot be found
    """
    output_type = py_module

    for part in type_name.split("."):
        if output_type != sys.modules:
            log.debug2(f"Looking for part {part} in {output_type}")
        else:
            log.debug2(f"Looking for part {part} in sys.modules")

        if isinstance(output_type, dict):
            output_type = output_type.get(part, None)
        else:
            output_type = getattr(output_type, part, None)
        if output_type is None:
            if not isinstance(py_module, dict):
                log.debug2(
                    "Couldn't find type name [%s] as an attribute on [%s]",
                    type_name,
                    py_module,
                )
            return None
    if output_type not in [None, py_module]:
        return output_type
