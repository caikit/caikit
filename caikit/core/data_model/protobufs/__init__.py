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


"""This module contains the python protobufs messages defined in the protobufs interfaces.
These are used as a backend and for serialization of caikit.core objects but should generally
be used only internally or by those intending to serialze data structures. Otherwise,
you are probably interested in the classes found directly in data_model.
"""

# Standard
import importlib
import os


def import_protobufs(proto_dir, package_base_name, current_globals):
    """Add protobufs definitions to the data model so that we can create custom models.
    To do this, we need to provide the path to the proto directory, the base package
    name, and the globals dict for the package being initialized. Usually this will be called
    in your __init__.py within your protobufs package, and look a lot like the following.

    import os

    # Get the import helper from the core
    from caikit.core.data_model.protobufs import import_protobufs
    proto_dir = os.path.dirname(os.path.realpath(__file__))

    # Import all probobufs as extensions to the core
    import_protobufs(proto_dir, __name__, globals())

    While we could do something like this with introspection, things (unfortunately) don't play
    nice with inspecting a wheel whose contents have been compiled to bytecode. :(

    Args:
        proto_dir (str): Path to the proto directory, i.e., the directory that
            you __init__ protobufs file is in.
        package_base_name (str): full name of your package, e.g., __name__ from
            the __init__ protobufs file.
        current_globals (dict): global dictionary from your protobufs package
            __init__ file.
    """

    # look for *_pb2.py files in proto_dir, we will consider these to be our protobufs files
    module_names = [
        filename.rstrip(".py")
        for filename in os.listdir(proto_dir)
        if filename.endswith("_pb2.py")
    ]

    # if there are no modules discovered, fallback to looking for .pyc files
    # this is necessary for binary-only releases
    if not module_names:
        module_names = [
            filename.rstrip(".pyc")
            for filename in os.listdir(proto_dir)
            if filename.endswith("_pb2.pyc")
        ]

    # dynamically load all protobufs files as relative modules
    all_modules = [
        importlib.import_module("." + module_name, package=package_base_name)
        for module_name in module_names
    ]

    # name of protobuf package to use, we ignore anything not in caikit_data_model for now
    _package_name = "caikit_data_model"

    # add all protobufs messages to current module and to the core's data_model
    all_enum_names = []
    for module in all_modules:
        if module.DESCRIPTOR.package.startswith(_package_name):
            for message_name in module.DESCRIPTOR.message_types_by_name.keys():
                message_val = getattr(module, message_name)
                current_globals[message_name] = message_val
                globals()[message_name] = message_val
            for enum_name in module.DESCRIPTOR.enum_types_by_name.keys():
                enum_val = getattr(module, enum_name)
                current_globals[enum_name] = enum_val
                globals()[enum_name] = enum_val
                all_enum_names.append(enum_name)
    current_globals["all_enum_names"] = all_enum_names
