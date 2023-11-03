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


"""This file exists for backwards compatibility.
It imports from the caikit.core.exceptions package where things moved"""

# Standard
import warnings as _warnings

# Local
from caikit.core.exceptions import DataValidationError, error_handler

# 🌶️🌶️🌶️ Warn if this package is ever imported
# Allow DeprecationWarnings through if anything tries to import from `toolkit.errors`
_warnings.filterwarnings("default", category=DeprecationWarning)
# And actually warn them
_warnings.warn(  # noqa: B028 # no explicit stacklevel keyword argument
    "The caikit.toolkit.errors package has moved to caikit.core.exceptions",
    DeprecationWarning,
)
