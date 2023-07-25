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
import time

# Third Party
import pytest

# Local
from caikit.core.toolkit.destroyable_process import FORK_CTX, DestroyableProcess


def test_processes_can_be_interrupted():
    def infinite_wait():
        while True:
            time.sleep(0.1)

    proc = DestroyableProcess(infinite_wait)
    proc.start()
    assert not proc.destroyed
    assert not proc.canceled
    assert not proc.ran
    assert not proc.threw
    proc.destroy()
    proc.join(60)
    assert not proc.is_alive()
    assert proc.destroyed
    assert proc.canceled
    assert proc.ran
    assert not proc.threw


def test_processes_can_return_results():
    expected = "test-any-result"
    proc = DestroyableProcess(lambda: expected)
    proc.start()
    proc.join()
    assert expected == proc.get_or_throw()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_process_not_canceled_after_success():
    proc = DestroyableProcess(lambda: None)
    proc.start()
    proc.join()
    assert not proc.canceled
    proc.destroy()
    assert not proc.canceled


def test_processes_can_be_set_to_not_return_results():
    expected = "test-any-result"
    proc = DestroyableProcess(lambda: expected, return_result=False)
    proc.start()
    proc.join()
    assert proc.get_or_throw() is None
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_processes_can_throw():
    expected = ValueError("test-any-error")

    def thrower():
        raise expected

    proc = DestroyableProcess(thrower)
    proc.start()
    proc.join()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert proc.threw

    with pytest.raises(ValueError) as ctx:
        proc.get_or_throw()

    assert str(expected) == str(ctx.value)


def test_processes_will_not_execute_if_destroyed_before_starting():
    proc = DestroyableProcess(lambda: time.sleep(1000))
    proc.destroy()
    proc.start()
    assert not proc.is_alive()
    proc.join()
    with pytest.raises(RuntimeError):
        proc.get_or_throw()
    assert proc.destroyed
    assert proc.canceled
    assert not proc.ran
    assert proc.threw


def test_event_is_set_on_completion():
    event = FORK_CTX.Event()
    proc = DestroyableProcess(lambda: None, completion_event=event)
    assert not event.is_set()
    proc.start()
    proc.join()
    assert event.is_set()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert not proc.threw


def test_event_is_set_on_exception():
    event = FORK_CTX.Event()

    def thrower():
        raise ValueError("test-any-exception")

    proc = DestroyableProcess(thrower, completion_event=event)
    assert not event.is_set()
    proc.start()
    proc.join()
    assert event.is_set()
    assert not proc.destroyed
    assert not proc.canceled
    assert proc.ran
    assert proc.threw


def test_default_event_is_set_on_completion():
    proc = DestroyableProcess(lambda: None)
    assert not proc.completion_event.is_set()
    proc.start()
    proc.join()
    assert proc.completion_event.is_set()
