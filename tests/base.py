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
"""Declares parent class for more specific test classes.
"""

# Standard
from typing import Any
import os
import threading
import unittest

# Local
from caikit.core.toolkit import logging


class TestCaseBase(unittest.TestCase):
    """Parent class for all specific test classes in Caikit Core."""

    # common reference to the place where we put our fixtures, because a ton of tests use this
    fixtures_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")
    # Other things that we use often
    modules_fixtures_dir = os.path.join(fixtures_dir, "modules")
    toolkit_fixtures_dir = os.path.join(fixtures_dir, "toolkit")

    def __init__(self, *args, **kwargs):
        # initialize parent class
        super().__init__(*args, **kwargs)

    def _compare_seqs(self, obj_seq1, obj_seq2, properties):
        """Given two sequences, ensure that they are the same length, and that each specified
        property is equal per object. This method explodes if the sequences are not the same
        as defined by assertEqual on the properties of interest.

        Args:
            obj_seq1: list | tuple
                List or tuple sequence.
            obj_seq2: list | tuple
                List or tuple sequence.
            properties: list(str) | tuple(str)
                attributes whose values will be compared across sequences.
        """
        self.assertEqual(len(obj_seq1), len(obj_seq2))
        for obj1, obj2 in zip(obj_seq1, obj_seq2):
            for prop in properties:
                self.assertEqual(getattr(obj1, prop), getattr(obj2, prop))

    def assert_equals_file(self, obj: Any, file_path: str):
        """Compare the string representation

        Args:
            obj: Any
                Object to compare against the contents of the file
            file_path: str
                Location of the file containing the expected string value of `obj`
        """
        with open(file_path, "r", encoding="utf-8") as file_handle:
            file_contents = file_handle.read()
        self.assertEqual(str(obj), file_contents)


def skip_in_wheel_test(cls):
    """In some cases (scripts, model evaluation), the code may not be shipped as part of the
    compiled wheel. In these situations, we only run the tests in unit tests, not in wheel tests.
    We do this by decorating the class - if env var NO_SOURCE is set to 1, as it is in the wheel
    test env, we return None so that the class is skipped by PyTest. Otherwise we return the class
    so that it can run normally.

    Args:
        cls: type
            Test class to be run if NO_SOURCE is not set to 1.
    Returns:
        type | None
            The test class if this is not running in the wheel tests, otherwise None.
    """
    if os.getenv("NO_SOURCE") is not None and int(os.getenv("NO_SOURCE")) == 1:
        return None
    return cls


class catch_threading_exception:
    """
    catch_threading_exception is only available in Python 3.10+
    Maintaining this copy to run tests with Python 3.8 and 3.9

    From https://github.com/python/cpython/blob/fbc9d0dbb22549bac2706f61f3ab631239d357b4/Lib/test/support/threading_helper.py#LL154C1-L208C24

    Context manager catching threading.Thread exception using
    threading.excepthook.

    Attributes set when an exception is caught:

    * exc_type
    * exc_value
    * exc_traceback
    * thread

    See threading.excepthook() documentation for these attributes.

    These attributes are deleted at the context manager exit.

    Usage:

        with threading_helper.catch_threading_exception() as cm:
            # code spawning a thread which raises an exception
            ...

            # check the thread exception, use cm attributes:
            # exc_type, exc_value, exc_traceback, thread
            ...

        # exc_type, exc_value, exc_traceback, thread attributes of cm no longer
        # exists at this point
        # (to avoid reference cycles)
    """

    def __init__(self):
        self.exc_type = None
        self.exc_value = None
        self.exc_traceback = None
        self.thread = None
        self._old_hook = None

    def _hook(self, args):
        self.exc_type = args.exc_type
        self.exc_value = args.exc_value
        self.exc_traceback = args.exc_traceback
        self.thread = args.thread

    def __enter__(self):
        self._old_hook = threading.excepthook
        threading.excepthook = self._hook
        return self

    def __exit__(self, *exc_info):
        threading.excepthook = self._old_hook
        del self.exc_type
        del self.exc_value
        del self.exc_traceback
        del self.thread
