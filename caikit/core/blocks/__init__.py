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


"""The `blocks` within the `caikit.core` library are essentially the conduits of algorithms.
Each block follows sets of principles about how they work including `.__init__()`, `.load()`,
`.run()`, `.save()`, and `.train()`. Blocks often require each other as inputs and support many
models.
"""

#################
## Core Blocks ##
#################

# Local
from .base import BlockSaver, block
