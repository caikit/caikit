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


"""Enumeration data structures map from strings to integers and back.
"""

# Standard
from enum import Enum
from typing import Dict, Optional, Tuple, Type

# Third Party
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
import google
import munch

# First Party
import alog

# Local
from ..toolkit.errors import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@classmethod
def to_dict(cls) -> Dict[str, int]:
    """Return a dict representation of the keys and values"""
    if not hasattr(cls, "__dict_repr__"):
        setattr(
            cls,
            "__dict_repr__",
            {
                entry.name: entry.value
                for entry in cls  # pylint: disable=not-an-iterable
            },
        )
    return cls.__dict_repr__


@classmethod
def to_munch(cls) -> munch.Munch:
    """Return a munchified version of the enum"""
    if not hasattr(cls, "__munch_repr__"):
        setattr(cls, "__munch_repr__", munch.Munch(cls.to_dict()))
    return cls.__munch_repr__


__all__ = ["import_enums", "import_enum"]


def import_enum(
    proto_enum: EnumTypeWrapper, enum_class: Optional[Type[Enum]] = None
) -> Tuple[str, str]:
    """Import a single enum into the global enum module by name

    Args:
        proto_enum (EnumTypeWrapper): The enum to import
        enum_class (Optional[Type[Enum]]): A pre-existing enum class that this
            proto enum binds to

    Returns:
        name:  str
            The name of the enum global
        rev_name:  str
            The name of the reversed enum global
    """
    if not isprotobufenum(proto_enum):
        error(
            "<COR71783964E>",
            AttributeError(f"`{proto_enum}` is not a valid protobuf enumeration"),
        )

    name = proto_enum.DESCRIPTOR.name
    log.debug2("Importing enum named %s", name)
    if enum_class is None:
        log.debug2("Creating Enum class for %s", name)
        enum_class = Enum._create_(name, proto_enum.items())

    # Add extra utility functions
    setattr(enum_class, "to_dict", to_dict)
    setattr(enum_class, "to_munch", to_munch)

    globals()[name] = enum_class
    rev_name = name + "Rev"
    globals()[rev_name] = munch.Munch({v: k for k, v in proto_enum.items()})
    __all__.append(name)
    __all__.append(rev_name)
    return name, rev_name


def import_enums(current_globals):
    """Add all enums and their reverse enum mappings a module's global symbol table. Note that
    we also update __all__. In general, __all__ controls the stuff that comes with a wild (*)
    import.

    Examples tend to make stuff like this easier to understand. Let's say the first name we hit
    is the Entity Mention Type. Then, after the first cycle through the loop below, you'll see
    something like:

        '__all__': ['import_enums', 'EntityMentionType', 'EntityMentionTypeRev']
        'EntityMentionType': { "MENTT_UNSET": 0, "MENTT_NAM": 1, ... , "MENTT_NONE": 4}
        'EntityMentionTypeRev': { "0": "MENTT_UNSET", "1": "MENTT_NAM", ... , "4": "MENTT_NONE"}

    since this is called explicitly below, you can thank this function for automagically syncing
    your enums (as importable from this file) with the data model.

    Args:
        current_globals (dict): global dictionary from your data model package
            __init__ file.
    """
    # Like the proto imports, we'd one day like to do this with introspection using something
    # like below, but can't because our wheel is compiled. If you can think of a cleaner way
    # to do this, open a PR!
    # caller = inspect.stack()[1]
    # caller_module = inspect.getmodule(caller[0])
    # current_globals = caller_module.__dict__

    # Add the str->int (EnumBase) and int->str (EnumRevBase) mapping for each enum
    # to the calling module's symbol table, then update __all__ to include the names
    # for the added objects.
    protobufs = current_globals.get("protobufs")
    all_enum_names = getattr(protobufs, "all_enum_names", [])
    for name in all_enum_names:
        proto_enum = getattr(protobufs, name)
        name, rev_name = import_enum(proto_enum)
        current_globals[name] = globals()[name]
        current_globals[rev_name] = globals()[rev_name]


def isprotobufenum(obj):
    """Returns True if obj is a protobufs enum."""
    return isinstance(obj, google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper)
