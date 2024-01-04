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
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import datetime
import random
import time

# Third Party
import grpc
import pytest

# Local
from caikit.runtime.types.aborted_exception import AbortedException
from caikit.runtime.work_management.abortable_context import (
    AbortableContext,
    ThreadInterrupter,
)
from caikit.runtime.work_management.rpc_aborter import RpcAborter
from tests.fixtures import Fixtures

## Helpers #####################################################################


@pytest.fixture(scope="session")
def thread_interrupter():
    interrupter = ThreadInterrupter()
    interrupter.start()

    yield interrupter

    interrupter.stop()


@pytest.fixture()
def grpc_context() -> grpc.ServicerContext:
    return Fixtures.build_context("abortable-context-test")


@pytest.fixture()
def rpc_aborter(grpc_context):
    return RpcAborter(grpc_context)


def wait_for_interrupter_to_run(interrupter, timeout=1):
    """Helper to wait until the interrupter's queue is empty.
    This should only deadlock if the interrupter's polling thread exits.
    """
    start = datetime.datetime.now()
    while (datetime.datetime.now() - start).total_seconds() < timeout:
        if interrupter._queue.empty():
            return
        time.sleep(0.001)


## Tests #######################################################################


def test_context_runs_stuff(thread_interrupter, rpc_aborter):
    """Just an ordinary context manager here"""
    one_plus_one = 0
    with AbortableContext(rpc_aborter, thread_interrupter):
        one_plus_one += 2

    assert one_plus_one == 2


def test_context_can_be_canceled(thread_interrupter, rpc_aborter, grpc_context):
    """An AbortedException is raised as soon as the rpc context is canceled"""
    result = 0
    with pytest.raises(AbortedException):
        with AbortableContext(rpc_aborter, thread_interrupter):
            result += 1
            grpc_context.cancel()
            assert not thread_interrupter._queue.empty()
            wait_for_interrupter_to_run(thread_interrupter)
            assert False

    assert result == 1


def test_context_aborts_if_rpc_already_canceled(
    thread_interrupter, rpc_aborter, grpc_context
):
    """The context will abort if the rpc context was previously canceled"""
    grpc_context.cancel()

    with pytest.raises(AbortedException):
        with AbortableContext(rpc_aborter, thread_interrupter):
            wait_for_interrupter_to_run(thread_interrupter)
            assert False


def test_exceptions_can_be_raised_in_context(thread_interrupter, rpc_aborter):
    """Exceptions work normally"""

    with pytest.raises(ValueError, match="this is a test"):
        with AbortableContext(rpc_aborter, thread_interrupter):
            raise ValueError("this is a test")


def test_many_threads_can_run_in_abortable_context_at_once(thread_interrupter):
    """This test tries to replicate a multithreaded situation where many threads can complete
    an AbortableContext and many others are aborted. We want to make sure only the contexts that
    we canceled are actually aborted- i.e. the interrupter interrupts the correct contexts."""

    @dataclasses.dataclass
    class TestTask:

        context: grpc.ServicerContext
        wait_for_cancel: bool
        future: Future = None

        def run(self):
            """Dummy task that either returns quickly or spins forever waiting to be interrupted"""
            aborter = RpcAborter(self.context)
            with AbortableContext(aborter=aborter, interrupter=thread_interrupter):
                if self.wait_for_cancel:
                    while True:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)

    # Create a bunch of tasks, half of them need to be interrupted
    tasks = []
    for i in range(25):
        tasks.append(
            TestTask(
                context=Fixtures.build_context(f"test-task-{i}"), wait_for_cancel=False
            )
        )
    for i in range(25):
        tasks.append(
            TestTask(
                context=Fixtures.build_context(f"test-cancel-task-{i}"),
                wait_for_cancel=True,
            )
        )
    random.shuffle(tasks)

    # Submit them all and cancel the context of the ones that need interrupting
    pool = ThreadPoolExecutor(max_workers=50)
    for t in tasks:
        t.future = pool.submit(t.run)
    for t in tasks:
        if t.wait_for_cancel:
            t.context.cancel()

    # Assert that the ones we canceled throw, and the rest don't
    for t in tasks:
        if t.wait_for_cancel:
            with pytest.raises(AbortedException):
                t.future.result()
        else:
            t.future.result()
