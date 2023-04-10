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


"""The `workflows` within the `caikit.core` library are essentially "super blocks" -- blocks
that call other blocks and compose an execution graph that describes how the output of one
block feeds into another block. Each `workflow` adheres to a contract that extends the contract
of a block, offering `.__init__()`, `.load()`, `.run()`, `.save()`, and `.train()` methods.
"""

# Local
from .base import WorkflowLoader, WorkflowSaver, workflow
