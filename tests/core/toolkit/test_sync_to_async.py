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
"""Tests for sync -> async wrappers"""
# Standard
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading
import time

# Third Party
import pytest

# Local
from caikit.core.toolkit.sync_to_async import async_wrap_iter

## Helpers #####################################################################


class IterableHelper:
    def __init__(self, iterable, delay=0, raise_element=None):
        self.iterable = iterable
        self.delay = delay
        self.raise_element = raise_element
        self.idx = 0
        self.iter = iter(self.iterable)

    def __iter__(self):
        return self

    def __next__(self):
        if self.delay:
            time.sleep(self.delay)
        next_element = next(self.iter)
        if self.idx == self.raise_element:
            raise RuntimeError("Raising")
        self.idx += 1
        return next_element


async def azip(*aiterables):
    aiterators = [aiterable.__aiter__() for aiterable in aiterables]
    while aiterators:
        results = await asyncio.gather(
            *[ait.__anext__() for ait in aiterators],
            return_exceptions=True,
        )
        if any(isinstance(result, StopAsyncIteration) for result in results):
            return
        yield tuple(results)


## Tests #######################################################################


@pytest.mark.asyncio
async def test_async_wrap_iter_happy():
    """Make sure a simple iterable can be wrapped as async and round trip
    cleanly
    """
    lst = [1, 2, 3, 4]
    async_round_trip = [x async for x in async_wrap_iter(lst)]
    assert lst == async_round_trip


@pytest.mark.asyncio
async def test_async_wrap_iter_concurrent():
    """Make sure that multiple wrapped synchronous iterators can interleve in
    the event loop
    """
    lst1 = [1, 2, 3]
    lst2 = [4, 5, 6]
    iter1 = IterableHelper(lst1)
    iter2 = IterableHelper(lst2)
    aiter1 = async_wrap_iter(iter1)
    aiter2 = async_wrap_iter(iter2)
    zipped = [elt async for elt in azip(aiter1, aiter2)]
    assert zipped == list(zip(lst1, lst2))


@pytest.mark.asyncio
async def test_async_wrap_iter_exception_propagation():
    """Make sure exceptions thrown during iteration are raised in the parent
    async context
    """
    with pytest.raises(RuntimeError):
        async for _ in async_wrap_iter(IterableHelper([1, 2, 3], raise_element=1)):
            pass


@pytest.mark.asyncio
async def test_async_wrap_iter_shared_executor():
    """Make sure that a shared ThreadPoolExecutor can be used to manage the work
    without exhausting thread counts
    """
    lst = [1, 2, 3, 4]
    pool = ThreadPoolExecutor(max_workers=1)

    start_event = threading.Event()
    finish_event = threading.Event()

    def slow_task():
        start_event.set()
        finish_event.wait()

    # Start the slow task and make sure it's started
    slow_future = pool.submit(slow_task)
    start_event.wait()

    # Use the same pool to start the async iteration
    agen = async_wrap_iter(lst, pool)

    # Make sure the async iteration is blocked
    assert pool._work_queue.qsize() == 1

    # Wait for the slow task to finish
    finish_event.set()
    slow_future.result()

    # Make sure the async iteration is unblocked
    assert pool._work_queue.qsize() == 0
    async_round_trip = [x async for x in agen]
    assert lst == async_round_trip
