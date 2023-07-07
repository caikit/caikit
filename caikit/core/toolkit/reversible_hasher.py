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
"""The ReversibleHasher provides a simple obfuscation tool for munging strings
in a reversible way. It is useful when needing a repeatable way to obfuscate a
string that may come from a configuration detail that should not be
transparently visible to a user, but needs to be trivially reversed, possibly
with entirely new in-memory state.

WARNING: This utility is intentionally not a cryptographically secure hash! It
    is explicitly designed to be reversible.
"""

# Standard
import random
import string

# Create a statically seeded instance of Random to perform shuffling in a
# consistent way
_RAND_INST = random.Random()
_RAND_INST.seed(42)
_CHARSET = string.ascii_letters + string.digits + "-_"
_SHUFFLED = list(_CHARSET)
_RAND_INST.shuffle(_SHUFFLED)

# Create the static forward/backward maps for the character set
FORWARD_MAP = dict(zip(_CHARSET, _SHUFFLED))
BACKWARD_MAP = {v: k for k, v in FORWARD_MAP.items()}


class ReversibleHasher:
    __doc__ = __doc__

    @classmethod
    def hash(cls, val: str):
        return "".join([FORWARD_MAP.get(ch, ch) for ch in val])

    @classmethod
    def reverse_hash(cls, hash_val: str):
        return "".join([BACKWARD_MAP.get(ch, ch) for ch in hash_val])
