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
"""This module holds utilities for wrapping synchronous functionality into async
wrapper functions
"""
# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Iterable, Optional
import asyncio


def async_wrap_iter(
    it: Iterable,
    pool: Optional[ThreadPoolExecutor] = None,
) -> AsyncGenerator:
    """Wrap blocking iterable into an asynchronous one

    CITE: https://stackoverflow.com/a/62297994

    This function manages a single thread for the iteration that shuttles the
    synchronous outputs back to the async generator using a queue
    """
    loop = asyncio.get_event_loop()
    q = asyncio.Queue(1)
    exception = None
    _END = object()

    async def yield_queue_items():
        while True:
            next_item = await q.get()
            if next_item is _END:
                break
            yield next_item
        if exception is not None:
            # the iterator has raised, propagate the exception
            raise exception

    def iter_to_queue():
        nonlocal exception
        try:
            for item in it:
                # This runs outside the event loop thread, so we
                # must use thread-safe API to talk to the queue.
                asyncio.run_coroutine_threadsafe(q.put(item), loop).result()
        except Exception as e:  # pylint: disable=broad-exception-caught
            exception = e
        finally:
            asyncio.run_coroutine_threadsafe(q.put(_END), loop).result()

    loop.run_in_executor(pool, iter_to_queue)
    return yield_queue_items()
