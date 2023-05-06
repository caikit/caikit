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


"""This contains the `base` class from which ALL blocks inherit. This class is not for direct use
and most methods are, in fact, abstract.
"""

# Standard
import threading

# First Party
import alog

# Local
from .. import module as mod
from ..module_type import module_type
from ..toolkit.errors import error_handler

log = alog.use_channel("BLBASE")
error = error_handler.get(log)


@module_type("block")
# pylint: disable=abstract-method
class BlockBase(mod.ModuleBase):
    """Abstract base class for creating Blocks. Inherits from ModuleBase."""

    # This mutex is shared among TensorFlow / Keras models to use around model loading.
    # In TensorFlow 1.x, model loading was *not* thread safe and this was required.
    # We need to verify whether or not model loading operations are thread safe in TensorFlow 2.x
    tensorflow_graph_mutex = threading.Lock()


# Hoist the @block decorator
block = BlockBase.block


class BlockSaver(mod.ModuleSaver):
    """DEPRECATED. Use ModuleSaver directly"""
