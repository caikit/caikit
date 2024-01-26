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
from typing import Dict, List, Optional, Union
import copy
import importlib
import os
import random
import re
import string
import sys
import tempfile

# Third Party
from google.protobuf import (
    descriptor,
    descriptor_pb2,
    descriptor_pool,
    struct_pb2,
    timestamp_pb2,
)

# First Party
import alog

# Local
from caikit.core.data_model.dataobject import _AUTO_GEN_PROTO_CLASSES
import caikit.core

log = alog.use_channel("TEST")


def add_fd_to_pool(
    fd: descriptor.FileDescriptor,
    dpool: descriptor_pool.DescriptorPool,
) -> List[descriptor_pb2.FileDescriptorProto]:
    """Helper to add a FileDescriptor to a new pool. This assumes the name is a
    unique identifier
    """
    added_fds = []
    try:
        dpool.FindFileByName(fd.name)
    except KeyError:
        log.debug2("Adding file %s to dpool %s", fd.name, dpool)
        for dep_fd in fd.dependencies:
            added_fds.extend(add_fd_to_pool(dep_fd, dpool))
        fd_proto = descriptor_pb2.FileDescriptorProto()
        fd.CopyToProto(fd_proto)
        dpool.Add(fd_proto)
        added_fds.append(fd_proto)
    return added_fds


@contextmanager
def temp_dpool(inherit_global: bool = False, skip_inherit: Optional[List[str]] = None):
    """Context to isolate the descriptor pool used in each test"""
    dpool = descriptor_pool.DescriptorPool()
    global_dpool = descriptor_pool._DEFAULT
    descriptor_pool._DEFAULT = dpool
    all_fd_protos = []
    all_fd_protos.extend(add_fd_to_pool(struct_pb2.DESCRIPTOR, dpool))
    all_fd_protos.extend(add_fd_to_pool(timestamp_pb2.DESCRIPTOR, dpool))

    # If inheriting from the current global, copy everything over
    if inherit_global:
        skip_inherit = skip_inherit or []
        for (
            dm_class
        ) in caikit.core.data_model.base._DataBaseMetaClass.class_registry.values():
            proto_class = dm_class.get_proto_class()
            fd_name = proto_class.DESCRIPTOR.file.name
            if any(re.match(skip_expr, fd_name) for skip_expr in skip_inherit):
                continue
            try:
                global_dpool.FindFileByName(fd_name)
                all_fd_protos.extend(add_fd_to_pool(proto_class.DESCRIPTOR.file, dpool))
            except KeyError:
                pass

    ##
    # HACK! Doing this _appears_ to solve the mysterious segfault cause by
    # using Struct inside a temporary descriptor pool. The inspiration for this
    # was:
    #
    # https://github.com/protocolbuffers/protobuf/issues/12047
    #
    # NOTE: This only works for protobuf 4.X (and as far as we know, it's not
    #     needed for 3.X)
    ##
    try:
        # Third Party
        from google.protobuf.message_factory import GetMessageClassesForFiles

        # Extra HACK! It seems that the below loop which _should_ do exactly
        # this somehow does not and by switching to it, the segfault reappears.
        msgs = GetMessageClassesForFiles(["google/protobuf/struct.proto"], dpool)
        _ = msgs["google.protobuf.Struct"]
        _ = msgs["google.protobuf.Value"]
        _ = msgs["google.protobuf.ListValue"]

        # for fd_proto in all_fd_protos:
        #     msgs = GetMessageClassesForFiles([fd_proto.name], dpool)
        #     for key in msgs:
        #         _ = msgs[key]

    # Nothing to do for protobuf 3.X
    except ImportError:
        pass
    try:
        yield dpool
    finally:
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
        return f"List[{_get_proto_val_name(field_val[0])}]"
    raise RuntimeError(f"Invalid field type specifier: {field_val}")


def make_proto_def(
    message_specs: Dict[str, dict],
    pkg_suffix: str = None,
    mock_compiled: bool = False,
) -> str:
    """Helper for writing a syntatically correct protobufs file"""
    if pkg_suffix is None:
        pkg_suffix = _random_package_suffix()
    package_name = f"{caikit.core.data_model.CAIKIT_DATA_MODEL}.{pkg_suffix}"
    out = justify_script_string(
        """
        from typing import Dict, List, Union
        """
    )
    if mock_compiled:
        out += justify_script_string(
            """
            from caikit.core.data_model import DataBase
            from caikit.core.data_model.dataobject import _make_data_model_class
            from py_to_proto import dataclass_to_proto, descriptor_to_message_class
            from dataclasses import dataclass
            """
        )
    else:
        out += "\nfrom caikit.core.data_model import DataObjectBase, dataobject\n"
    for message_name, message_spec in message_specs.items():
        type_annotations = ""
        for field_name, field_type in message_spec.items():
            type_annotations += f"    {field_name}: {_get_proto_val_name(field_type)}\n"

        # Manually create the proto class and use the precompiled proto style
        if mock_compiled:
            dataclass_name = f"_{message_name}"
            proto_name = f"{dataclass_name}_proto"
            msg_str = f"\n@dataclass\nclass {dataclass_name}:\n"
            msg_str += type_annotations
            msg_str += justify_script_string(
                f"""

                {proto_name} = descriptor_to_message_class(
                    dataclass_to_proto(
                        dataclass_={dataclass_name},
                        package="{package_name}",
                        name="{message_name}",
                    )
                )

                class {message_name}(DataBase):
                    _proto_class = {proto_name}
                    
                {message_name} = _make_data_model_class({proto_name},{message_name})
                """
            )

        # Add as a dataobject
        else:
            msg_str = f'\n@dataobject("{package_name}")\nclass {message_name}(DataObjectBase):\n'
            msg_str += type_annotations

        out += msg_str
    return out


@contextmanager
def reset_global_protobuf_registry():
    """Reset the global registry of generated protos"""
    prev_auto_gen_proto_classes = copy.copy(_AUTO_GEN_PROTO_CLASSES)
    prev_class_registry = copy.copy(
        caikit.core.data_model.base._DataBaseMetaClass.class_registry
    )
    _AUTO_GEN_PROTO_CLASSES.clear()
    yield
    _AUTO_GEN_PROTO_CLASSES.extend(prev_auto_gen_proto_classes)
    caikit.core.data_model.base._DataBaseMetaClass.class_registry.clear()
    caikit.core.data_model.base._DataBaseMetaClass.class_registry.update(
        prev_class_registry
    )
