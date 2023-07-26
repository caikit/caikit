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
try:
    # Standard
    from test.support.threading_helper import catch_threading_exception
except (NameError, ModuleNotFoundError):
    from tests.base import catch_threading_exception

# Standard
import threading
import time

# Third Party
import pytest

# Local
from caikit.core.toolkit.destroyable_thread import (
    DestroyableThread,
    ThreadDestroyedException,
)


def test_threads_can_be_interrupted():
    def infinite_wait():
        while True:
            time.sleep(0.001)

    thread = DestroyableThread(infinite_wait)
    thread.start()
    thread.destroy()
    assert thread.canceled
    thread.join(60)
    assert not thread.is_alive()


def test_threads_canceled_when_interrupt_fails():
    """This test exercises the case where the DestroyableThread fails to
    interrupt the thread with the internal exception on destroy(). The easiest
    way to simulate this is with a "long" `time.sleep` but that still doesn't
    fully guarantee that it will always be free of race conditions. In
    particular, the following two are still possible:

    1. The destroy() call lands in between waiting on the start event and
        beginning the time.sleep. This would cause the thread to die cleanly.
    2. The assertion that thread.is_alive() lands after the time.sleep has
        finished and the destroying exception has done its job to kill the
        thread.
    """
    start_event = threading.Event()
    end_event = threading.Event()

    def blocking_fn():
        start_event.wait()
        time.sleep(0.01)
        end_event.wait()

    thread = DestroyableThread(blocking_fn)
    thread.start()
    start_event.set()
    thread.destroy()
    assert thread.canceled
    # NOTE: We don't assert that thread.is_alive() here since it's potentially
    #   susceptible to the above mentioned race conditions. It _should_ always
    #   be true, though, based on reasonable timing.
    thread.join(60)
    assert not thread.is_alive()


def test_threads_can_catch_the_interrupts():
    started = threading.Event()
    caught = threading.Event()

    def test_catcher(started_event: threading.Event, caught_event: threading.Event):
        try:
            started_event.set()
            while True:
                time.sleep(0.001)
        except Exception as e:
            caught_event.set()
            raise e

    thread = DestroyableThread(test_catcher, started_event=started, caught_event=caught)

    thread.start()
    started.wait()

    thread.destroy()
    thread.join(60)

    assert not thread.is_alive()
    assert caught.is_set()


def test_threads_can_return_results():
    expected = "test-any-result"
    thread = DestroyableThread(lambda: expected)

    thread.start()
    thread.join()

    assert expected == thread.get_or_throw()


def test_threads_not_canceled_after_success():
    thread = DestroyableThread(lambda: None)
    thread.start()
    thread.join()
    assert not thread.canceled
    thread.destroy()
    assert not thread.canceled


def test_threads_can_throw():
    expected = ValueError("test-any-error")

    def thrower():
        raise expected

    thread = DestroyableThread(thrower)

    thread.start()
    thread.join()

    with pytest.raises(ValueError) as ctx:
        thread.get_or_throw()

    assert expected == ctx.value


def test_threads_will_not_execute_if_destroyed_before_starting():
    thread = DestroyableThread(lambda: time.sleep(1000))

    with catch_threading_exception() as cm:
        thread.destroy()
        thread.start()
        thread.join(1)

        assert not thread.is_alive()

        # Make sure the correct exception was raised
        assert cm.exc_type == ThreadDestroyedException


def test_event_is_set_on_completion():
    event = threading.Event()
    thread = DestroyableThread(lambda: None, work_done_event=event)

    assert not event.is_set()
    thread.start()
    thread.join()
    assert event.is_set()


def test_event_is_set_on_exception():
    event = threading.Event()

    def thrower():
        raise ValueError("test-any-exception")

    thread = DestroyableThread(thrower, work_done_event=event)

    assert not event.is_set()
    thread.start()
    thread.join()
    assert event.is_set()
