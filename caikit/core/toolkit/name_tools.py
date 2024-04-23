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

"""Common string functions that are generally helpful for generating runtime RPC names
and other Protobuf names
"""

# Standard
import re


def snake_to_upper_camel(string: str) -> str:
    """Simple snake -> upper camel conversion for descriptors"""
    return "".join([part[0].upper() + part[1:] for part in string.split("_") if part])


def camel_to_snake_case(string: str, kebab_case: bool = False) -> str:
    """Convert from CamelCase (or camelCase) to snake_case or kebab-case"""
    return re.sub(
        r"(?<!^)(?=[A-Z])",
        "-" if kebab_case else "_",
        string,
    ).lower()
