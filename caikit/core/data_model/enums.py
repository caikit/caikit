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
from typing import Tuple, Type
import json

# Third Party
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
import munch

# First Party
# First party
import alog

# Local
from ..toolkit.errors import error_handler
from ..toolkit.isa import isprotobufenum
from . import protobufs

log = alog.use_channel("DATAM")
error = error_handler.get(log)


class EnumBase(munch.Munch):
    """Enumerations maps from string to integer."""

    def __init__(self, proto_enum: EnumTypeWrapper):
        name = proto_enum.DESCRIPTOR.name
        if proto_enum is None:
            error(
                "<COR71783952E>",
                AttributeError(f"could not locate protobuf enum `{name}`"),
            )

        if not isprotobufenum(proto_enum):
            error(
                "<COR71783964E>",
                AttributeError(f"`{name}` is not a valid protobuf enumeration"),
            )

        self._proto_enum = proto_enum

        super().__init__(proto_enum.items())

    def __repr__(self):
        return json.dumps(self, indent=2)

    def items(self):
        """We need to overwrite items() from dict so that hidden attributes are
        not shown
        """
        return self.__dict__.items()

    def toDict(self):
        """We need to overwrite toDict from the default Munch implementation so
        that hidden attributes are not shown.
        """
        return {k: v for k, v in super().toDict().items() if k != "_proto_enum"}


class _EnumRevInject(munch.Munch, dict):
    """Use multiple inheritance dependency injection to reverse the order
    of the enum map before passing to dict parent.  In order to understand
    this consider the method resolution order (MRO) for __init__ in this
    class hierarchy:

    EnumRevBase -> EnumBase -> Munch -> _EnumRevInject -> Munch -> dict -> object
    """

    def __init__(self, forward_map):
        # reverse keys and values and call munch constructor
        super().__init__({value: key for key, value in forward_map})


class EnumRevBase(EnumBase, _EnumRevInject):
    """Reverse enumeration maps from integer to string."""


__all__ = ["import_enums", "import_enum"]


def import_enum(proto_enum: EnumTypeWrapper) -> Tuple[Type[EnumBase], str]:
    """Import a single enum into the global enum module by name

    Args:
        proto_enum:  EnumTypeWrapper
            The enum to import

    Returns:
        name:  str
            The name of the enum global
        rev_name:  str
            The name of the reversed enum global
    """
    name = proto_enum.DESCRIPTOR.name
    enum_class = EnumBase(proto_enum)
    globals()[name] = enum_class
    rev_name = name + "Rev"
    globals()[rev_name] = EnumRevBase(proto_enum)
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
        current_globals: dict
            global dictionary from your data model package __init__ file.
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
    all_enum_names = getattr(current_globals.get("protobufs"), "all_enum_names", [])
    for name in all_enum_names:
        proto_enum = getattr(protobufs, name)
        name, rev_name = import_enum(proto_enum)
        current_globals[name] = globals()[name]
        current_globals[rev_name] = globals()[rev_name]
