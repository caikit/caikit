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

"""This library defines the taxonomy of Data Model objects and Tasks for the
entire CAIKit project. Data objects and tasks are grouped domain, making for a
three-level hierarchy for models:

problem domain -> task -> implementation

This library intentionally does NOT define any implementations, as those are
left to the domain implementation libraries.
"""

# NOTE: We do not proactively import sub-modules in the interfaces here as they
#   may contain optional dependencies
