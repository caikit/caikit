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


"""Utility functions for testing if an object is an instance of a given type.
"""

# Standard
import collections

# Third Party
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
import google
import semver


def isprotobuf(obj):
    """Returns True if obj is a protobufs message, otherwise False."""
    return isinstance(obj, google.protobuf.message.Message)


def isprotobufclass(obj):
    """Returns True if obj is a protobufs message class.
    If you want to test for a protobufs message instance, not a class, use isprotobuf instead.
    """
    return isinstance(
        obj, google.protobuf.pyext.cpp_message.GeneratedProtocolMessageType
    )


def isprotobufenum(obj):
    """Returns True if obj is a protobufs enum."""
    return isinstance(obj, google.protobuf.internal.enum_type_wrapper.EnumTypeWrapper)


def isprotobufrepeated(obj):
    """Returns true if obj is a repeated Message or repeated primitive"""
    return isinstance(
        obj, (RepeatedCompositeFieldContainer, RepeatedScalarFieldContainer)
    )


def isprimitive(obj):
    """Returns True if obj is a python primitive (bool, int, float, str), otherwise False."""
    return (obj is None) or isinstance(obj, (bool, int, float, str))


def isiterable(obj):
    """Returns True if obj can be iterated over, otherwise False."""
    return isinstance(obj, collections.abc.Iterable)


def isvalidversion(obj):
    """Returns True if obj is a valid parseable semantic version (https://semver.org/)"""
    try:
        semver.VersionInfo.parse(obj)  # Parse doesn't fail on valid object
        return True
    except (TypeError, ValueError):
        return False
