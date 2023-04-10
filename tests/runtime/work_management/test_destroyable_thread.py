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
import threading
import time
import unittest

# Local
from caikit.runtime.work_management.destroyable_thread import DestroyableThread


class TestDestroyableThread(unittest.TestCase):
    def test_threads_can_be_interrupted(self):
        def infinite_wait():
            while True:
                time.sleep(0.1)

        thread = DestroyableThread(threading.Event(), infinite_wait)

        thread.start()
        thread.destroy()
        thread.join(60)

        self.assertFalse(thread.is_alive())

    def test_threads_can_catch_the_interrupts(self):
        started = threading.Event()
        caught = threading.Event()

        def test_catcher(started_event: threading.Event, caught_event: threading.Event):
            try:
                started_event.set()
                while True:
                    time.sleep(0.1)
            except Exception as e:
                caught_event.set()
                raise e

        thread = DestroyableThread(threading.Event(), test_catcher, started, caught)

        thread.start()
        started.wait()

        thread.destroy()
        thread.join(60)

        self.assertFalse(thread.is_alive())
        self.assertTrue(caught.is_set())

    def test_threads_can_return_results(self):
        expected = "test-any-result"
        thread = DestroyableThread(threading.Event(), lambda: expected)

        thread.start()
        thread.join()

        self.assertEqual(expected, thread.get_or_throw())

    def test_threads_can_throw(self):
        expected = ValueError("test-any-error")

        def thrower():
            raise expected

        thread = DestroyableThread(threading.Event(), thrower)

        thread.start()
        thread.join()

        with self.assertRaises(ValueError) as ctx:
            thread.get_or_throw()

        self.assertEqual(expected, ctx.exception)

    def test_threads_will_not_execute_if_destroyed_before_starting(self):
        thread = DestroyableThread(threading.Event(), lambda: time.sleep(1000))

        thread.destroy()
        thread.start()
        thread.join(1)

        self.assertFalse(thread.is_alive())

    def test_event_is_set_on_completion(self):
        event = threading.Event()
        thread = DestroyableThread(event, lambda: None)

        self.assertFalse(event.is_set())
        thread.start()
        thread.join()
        self.assertTrue(event.is_set())

    def test_event_is_set_on_exception(self):
        event = threading.Event()

        def thrower():
            raise ValueError("test-any-exception")

        thread = DestroyableThread(event, thrower)

        self.assertFalse(event.is_set())
        thread.start()
        thread.join()
        self.assertTrue(event.is_set())


if __name__ == "__main__":
    unittest.main()
