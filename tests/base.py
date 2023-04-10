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
import unittest

# Local
from caikit.core.toolkit import logging


class TestCaseBase(unittest.TestCase):
    """Parent class for all specific test classes in Caikit Core."""

    # common reference to the place where we put our fixtures, because a ton of tests use this
    fixtures_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")
    # Other things that we use often
    block_fixtures_dir = os.path.join(fixtures_dir, "blocks")
    resource_fixtures_dir = os.path.join(fixtures_dir, "resources")
    workflow_fixtures_dir = os.path.join(fixtures_dir, "workflows")
    toolkit_fixtures_dir = os.path.join(fixtures_dir, "toolkit")

    def __init__(self, *args, **kwargs):
        # configure logging from env
        default_level = os.environ.get("ALOG_DEFAULT_LEVEL", "error")
        filters = os.environ.get("ALOG_FILTERS", "")
        logging.configure(default_level, filters)

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
