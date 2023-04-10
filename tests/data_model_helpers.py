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
Helpers for testing out data model functionality
"""

# Standard
from contextlib import contextmanager
from types import ModuleType
from typing import Dict, List, Union
import importlib
import os
import random
import string
import sys
import tempfile

# Third Party
from google.protobuf import descriptor_pool

# Local
import caikit.core


@contextmanager
def temp_dpool():
    """Fixture to isolate the descriptor pool used in each test"""
    dpool = descriptor_pool.DescriptorPool()
    global_dpool = descriptor_pool._DEFAULT
    descriptor_pool._DEFAULT = dpool
    yield dpool
    descriptor_pool._DEFAULT = global_dpool


def justify_script_string(script_str):
    """Helper to allow us to write 'scripts' as strings with space padding so
    that they don't look ugly!
    """
    min_indent = None
    lines = script_str.split("\n")
    for line in lines:
        if not line.strip():
            continue
        indent_len = len(line) - len(line.lstrip(" "))
        if min_indent is None or indent_len < min_indent:
            min_indent = indent_len
    if min_indent is not None:
        lines = [line[min_indent:] for line in lines]
    return "\n".join(lines)


def _make_data_model_module(
    module_dir: str, proto_defs: Union[str, List[str]]
) -> ModuleType:
    """Make a sample data model module!"""

    # Make the directory structure
    dm_dir = os.path.join(module_dir, "data_model")
    os.mkdir(dm_dir)

    # Set up the module hierarchy
    module_init = os.path.join(module_dir, "__init__.py")
    with open(module_init, "w", encoding="utf-8") as handle:
        handle.write(
            justify_script_string(
                """
                from . import data_model
                """
            )
        )
    dm_init = os.path.join(dm_dir, "__init__.py")
    if isinstance(proto_defs, str):
        proto_defs = [proto_defs]
    with open(dm_init, "w", encoding="utf-8") as handle:
        for proto_def in proto_defs:
            handle.write(proto_def)


@contextmanager
def temp_module():
    """Helper that will create an ephemeral module that can be imported and then
    clean it up afterwards
    """
    with tempfile.TemporaryDirectory() as workdir:
        # Make the randomized module name and directory
        mod_suffix = "".join(random.choice(string.ascii_lowercase) for i in range(10))
        mod_name = f"temp_mod_{mod_suffix}"
        mod_dir = os.path.join(workdir, mod_name)
        os.mkdir(mod_dir)

        # Add this the parent dir to the sys path so that it can be merged
        sys.path.append(workdir)

        # Yield the module directory and name so that the test can muck with it
        # before importing it
        yield mod_name, mod_dir

        # Remove the workdir from the sys path and delete the module from
        # sys.modules if it's there
        sys.path.pop()
        sys.modules.pop(mod_name, None)


@contextmanager
def temp_data_model(proto_defs: Union[str, List[str]]) -> ModuleType:
    """This contextmanager creates a temporary data model hierarchy with the
    given set of protobufs defs. It can be used to evaluate different
    combinations of message types without needing to manually write and compile
    protos for each test.
    """
    with temp_module() as (mod_name, module_dir):
        with temp_dpool():
            _make_data_model_module(module_dir, proto_defs)
            temp_mod = importlib.import_module(mod_name)
            yield temp_mod.data_model


def _random_package_suffix() -> str:
    some_list = list(
        "thisisabigoldstringofcharactersthatwillgetshuffledintoarandompackagesuffixthisisnotsuperb"
        "ombproofbutwhatever"
    )
    random.shuffle(some_list)
    return "".join(some_list)


def _get_proto_val_name(field_val) -> str:
    if isinstance(field_val, str):
        return field_val
    if isinstance(field_val, type):
        return field_val.__name__
    if isinstance(field_val, list):
        assert len(field_val) == 1
        return f"""{{"elements": {_get_proto_val_name(field_val[0])}}}"""
    raise RuntimeError(f"Invalid field type specifier: {field_val}")


def make_proto_def(message_specs: Dict[str, dict], pkg_suffix: str = None) -> str:
    """Helper for writing a syntatically correct protobufs file"""
    if pkg_suffix is None:
        pkg_suffix = _random_package_suffix()
    package_name = f"{caikit.core.data_model.CAIKIT_DATA_MODEL}.{pkg_suffix}"
    # pylint: disable=f-string-without-interpolation
    out = justify_script_string(
        f"""
        import caikit.core

        """
    )
    for message_name, message_spec in message_specs.items():
        # pylint: disable=f-string-without-interpolation
        msg_str = f"""\n@caikit.core.dataobject(schema={{"""
        schema_fields = [
            f'"{field_name}": {_get_proto_val_name(field_val)}'
            for field_name, field_val in message_spec.items()
        ]
        msg_str += ", ".join(schema_fields)
        msg_str += f'}}, package="{package_name}")\nclass {message_name}: pass\n\n'
        out += msg_str
    return out
