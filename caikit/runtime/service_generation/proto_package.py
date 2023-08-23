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
This module holds the common helpers for managing the protobuf package name for
the runtime
"""

# Local
from ...config import get_config


def snake_to_upper_camel(string: str) -> str:
    """Simple snake -> upper camel conversion"""
    return "".join([part[0].upper() + part[1:] for part in string.split("_")])


def get_ai_domain() -> str:
    """Get the string name for the AI domain"""
    caikit_config = get_config()
    lib = caikit_config.runtime.library
    default_ai_domain_name = snake_to_upper_camel(lib.replace("caikit_", ""))
    ai_domain_name = (
        caikit_config.runtime.service_generation.domain or default_ai_domain_name
    )
    return ai_domain_name


def get_runtime_service_package() -> str:
    """This helper will get the common runtime package"""
    caikit_config = get_config()
    ai_domain_name = get_ai_domain()
    default_package_name = f"caikit.runtime.{ai_domain_name}"
    package_name = (
        caikit_config.runtime.service_generation.package or default_package_name
    )
    return package_name
