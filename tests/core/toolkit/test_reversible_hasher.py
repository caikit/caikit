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
Tests for the reversible hasher
"""

# Standard
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random

# Local
from caikit.core.toolkit.reversible_hasher import ReversibleHasher

## Helpers #####################################################################


def do_hash(x):
    hashed = ReversibleHasher.hash(x)
    reversed_ = ReversibleHasher.reverse_hash(hashed)
    return (hashed, reversed_)


## Tests #######################################################################


def test_reversible_hasher_single_process():
    """Make sure that hashing the same string multiple times in sequence is
    repeatable and reversible
    """
    sample_string = "thisisatest1234-of_stuff"
    hashed1 = ReversibleHasher.hash(sample_string)
    hashed2 = ReversibleHasher.hash(sample_string)
    reversed1 = ReversibleHasher.reverse_hash(hashed1)
    reversed2 = ReversibleHasher.reverse_hash(hashed2)
    assert hashed1 == hashed2
    assert reversed1 == reversed2 == sample_string


def test_reversible_hasher_multi_thread():
    """Make sure that hashing the same string multiple times in different
    threads is repeatable and reversible
    """
    sample_string = "ANoTHe-r_t327"
    executor = ThreadPoolExecutor()
    future1 = executor.submit(do_hash, x=sample_string)
    future2 = executor.submit(do_hash, x=sample_string)
    (hashed1, reversed1) = future1.result()
    (hashed2, reversed2) = future2.result()
    assert hashed1 == hashed2
    assert reversed1 == reversed2 == sample_string


def test_reversible_hasher_multi_process():
    """Make sure that hashing the same string multiple times in different
    processes is repeatable and reversible
    """
    sample_string = "y3t_ANoTHe-r_tES T"
    executor = ProcessPoolExecutor()
    future1 = executor.submit(do_hash, x=sample_string)
    future2 = executor.submit(do_hash, x=sample_string)
    (hashed1, reversed1) = future1.result()
    (hashed2, reversed2) = future2.result()
    assert hashed1 == hashed2
    assert reversed1 == reversed2 == sample_string


def test_reversible_hasher_random_still_random():
    """Make sure that the random state is not changed by hashing"""
    rand_state = random.getstate()
    do_hash("yetanother")
    assert random.getstate() == rand_state
