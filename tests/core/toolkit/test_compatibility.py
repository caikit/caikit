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
from unittest.mock import patch

# Local
from caikit.core.toolkit.compatibility import (
    is_importable_on_platform,
    unsupported_platforms,
)

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestCompatibilityUtilities(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.echo_input = "hello world!"

    @staticmethod
    def build_unsupported_echo(platform_types):
        @unsupported_platforms(platform_types)
        def echo(some_arg):
            """The function that we are testing our decorator against."""
            return some_arg

        return echo

    @patch("sys.platform", "linux")
    def test_os_compatibility_happy_platform(self):
        """Ensure if we run a supported platform with one unsupported, we can run normally"""
        # Build an echo that explodes on Darwin [OS X], test against running Linux
        deco_echo = TestCompatibilityUtilities.build_unsupported_echo("darwin")
        # Ensures that args / kwargs forward appropriately
        echo_result = deco_echo(self.echo_input)
        self.assertEqual(echo_result, self.echo_input)

    @patch("sys.platform", "linux")
    def test_os_compatibility_happy_platforms(self):
        """Ensure if we run a supported platform with multiple unsupported, we can run normally"""
        deco_echo = TestCompatibilityUtilities.build_unsupported_echo(
            ["win32", "darwin"]
        )
        echo_result = deco_echo(self.echo_input)
        self.assertEqual(echo_result, self.echo_input)

    @patch("sys.platform", "linux")
    def test_os_compatibility_sad_platform_string(self):
        """Ensure if we run the only unspported platform, we explode before calling"""
        # Build an echo that explodes on Linux, test against running Linux
        deco_echo = TestCompatibilityUtilities.build_unsupported_echo("linux")
        with self.assertRaises(NotImplementedError):
            deco_echo(self.echo_input)

    @patch("sys.platform", "linux")
    def test_os_compatibility_sad_platforms(self):
        """Ensure if we run one of many unspported platforms, we explode before calling"""
        # Build an echo that explodes on Linux & Darwin [OS X], test against running Linux
        deco_echo = TestCompatibilityUtilities.build_unsupported_echo(
            ["linux", "darwin"]
        )
        with self.assertRaises(NotImplementedError):
            deco_echo(self.echo_input)

    def test_os_compatibility_bad_decorator_arg_types(self):
        """Ensure if the decorator type is wrong, we explode when the decorator runs"""
        with self.assertRaises(TypeError):
            TestCompatibilityUtilities.build_unsupported_echo(100)

    def test_os_compatibility_bad_decorator_arg_inner_types(self):
        """Ensure if the decorator inner type is wrong, we explode when the decorator runs"""
        with self.assertRaises(TypeError):
            TestCompatibilityUtilities.build_unsupported_echo([100])

    ###############################################################################################
    #                   Tests for OS-based conditional error handling on imports                  #
    ###############################################################################################
    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_on_happy_os(self, import_module_mock):
        """Ensure that if we can import on a happy platform, we return True"""
        self.assertTrue(
            is_importable_on_platform(
                ".foo", "bar", ImportError, "linux", "Some message"
            )
        )

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_happy_on_sad_os(self, import_module_mock):
        """Ensure that we return True if we can import on a platform with special error handling"""
        self.assertTrue(
            is_importable_on_platform(
                ".foo", "bar", ImportError, "darwin", "Some message"
            )
        )

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_unhandled_import_failure_raises_with_sad_os(self, import_module_mock):
        """Ensure that we still raise if an error type we aren't expecting raises on import"""
        import_module_mock.side_effect = ValueError("Boom!")
        with self.assertRaises(ValueError):
            # NOTE: Import only runs through this function if it's a platform we watch, so we
            # need to match our patched value, which in this case is Darwin, to raise; otherwise
            # we would just return True outright and go on to import after the conditional.
            is_importable_on_platform(
                ".foo", "bar", ImportError, "darwin", "Some message"
            )

    @patch("importlib.import_module")
    @patch("sys.platform", "linux")
    def test_unhandled_import_failure_raises_with_happy_os(self, import_module_mock):
        """Ensure that import doesn't run if this is not a watched platform"""
        import_module_mock.side_effect = ValueError("Boom")
        self.assertTrue(
            is_importable_on_platform(
                ".foo", "bar", ImportError, "darwin", "Some message"
            )
        )

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_sad_on_handled_err_type_on_sad_os(self, import_module_mock):
        """Ensure we don't import or raise if something is watched and not importable"""
        import_module_mock.side_effect = ImportError("This one is handled!")
        self.assertFalse(
            is_importable_on_platform(
                ".foo", "bar", ImportError, "darwin", "Some message"
            )
        )

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_handles_multiple_os_targets(self, import_module_mock):
        """Ensure we don't import or raise if something is watched and not importable"""
        import_module_mock.side_effect = ImportError("This one is handled!")
        self.assertFalse(
            is_importable_on_platform(
                ".foo", "bar", ImportError, ["darwin", "linux"], "Some message"
            )
        )

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_validates_outer_arg_types(self, import_module_mock):
        """Ensure that if we replace any arg with a bad type, we raise a TypeError before import"""
        import_module_mock.side_effect = Exception("This will never happen!")
        happy_args = [".foo", "bar", ImportError, "darwin", "Some message"]
        for arg_idx in range(len(happy_args)):
            sad_args = happy_args.copy()
            sad_args[arg_idx] = None
            with self.assertRaises(TypeError):
                is_importable_on_platform(*sad_args)

    @patch("importlib.import_module")
    @patch("sys.platform", "darwin")
    def test_is_importable_validates_iterable_arg_types(self, import_module_mock):
        """Ensure we replace our iterable arg with bad types, our handling is still happy"""
        import_module_mock.side_effect = Exception("This will never happen!")
        with self.assertRaises(TypeError):
            is_importable_on_platform(
                ".foo", "bar", ImportError, [1, 2, 3, 4], "Some message"
            )
