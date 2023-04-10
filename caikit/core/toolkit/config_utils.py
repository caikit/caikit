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

# Standard
from typing import Optional


def merge_configs(base: Optional[dict], overrides: Optional[dict]) -> dict:
    """Helper to perform a deep merge of the overrides into the base. The merge
    is done in place, but the resulting dict is also returned for convenience.
    The merge logic is quite simple: If both the base and overrides have a key
    and the type of the key for both is a dict, recursively merge, otherwise
    set the base value to the override value.
    Args:
        base:  Optional[dict]
            The base config that will be updated with the overrides
        overrides:  Optional[dict]
            The override config
    Returns:
        merged:  dict
            The merged results of overrides merged onto base
    """
    # Handle none args
    if base is None:
        return overrides or {}
    if overrides is None:
        return base or {}

    # Do the deep merge
    for key, value in overrides.items():
        if (
            key not in base
            or not isinstance(base[key], dict)
            or not isinstance(value, dict)
        ):
            base[key] = value
        else:
            base[key] = merge_configs(base[key], value)

    return base
