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


class _ReversibleHasherMeta(type):
    """The _ReversibleHasherMeta is used here to ensure that the random state is
    only temporarily seeded at import time.

    WARNING: If this class is dynamically imported in a multi-threaded
        application, other threads that share the random state will be effected
        by the temporary seed!
    """

    def __new__(mcs, name, bases, attrs):
        # Temporarily seed random
        random_state = random.getstate()
        random.seed(42)

        # Generate the shuffling of all acceptable characters
        charset = string.ascii_letters + string.digits + "-_"
        shuffled = list(charset)
        random.shuffle(shuffled)

        # Add the mapping attrs to the class
        fwd_map = dict(zip(charset, shuffled))
        attrs["FORWARD_MAP"] = fwd_map
        attrs["BACKWARD_MAP"] = {v: k for k, v in fwd_map.items()}

        # Reset the random state
        random.setstate(random_state)
        return super().__new__(mcs, name, bases, attrs)


class ReversibleHasher(metaclass=_ReversibleHasherMeta):
    __doc__ = __doc__

    @classmethod
    def hash(cls, val: str):
        return "".join([cls.FORWARD_MAP.get(ch, ch) for ch in val])

    @classmethod
    def reverse_hash(cls, hash_val: str):
        return "".join([cls.BACKWARD_MAP.get(ch, ch) for ch in hash_val])
