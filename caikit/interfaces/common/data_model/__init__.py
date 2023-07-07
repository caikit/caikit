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
Domain agnostic data model objects
"""

################################################################################
# We really wanted to avoid having concrete data model objects live in
# caikit.core, but because ModuleBase used ProducerId, we had to keep that
# there. That said, it is forwarded to caikit.interfaces.common, so from a
# user's perspective, we should consider it to be part of
# caikit.interfaces.common not caikit.core.data_model.
################################################################################

# Local
# Import individual packages
from . import primitive_sequences, producer
from .primitive_sequences import BoolSequence, FloatSequence, IntSequence, StrSequence
from .producer import ProducerId
