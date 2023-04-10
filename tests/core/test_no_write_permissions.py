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
"""RQA has a build that does not get write permissions, so a whole bunch of things can fail

We want to make sure that at least `import caikit.core` does not barf, which it will if we aren't
careful
"""
# Standard
from unittest.mock import patch

with patch("os.makedirs") as makedirs:
    makedirs.side_effect = PermissionError("No files for you")

    # Make sure importing does not fail
    # pylint: disable=unused-import
    # Local
    import caikit.core
